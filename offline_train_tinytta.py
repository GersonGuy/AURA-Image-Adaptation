"""
Offline self-ensemble stage for TinyTTA (paper Sec. 3.4, Eq. 5 + Eq. 6).

MANDATORY, per the paper's own Limitations section, verbatim: "TinyTTA requires retraining the model
via self-ensemble." Sec. 3.4: "we introduce the concurrent training of all models on the training
dataset. Given the source data, we pass them through both the pre-trained model and the submodules
combined with early exit branches and compute the losses to optimize the models."

    L1 = sum_k CE(z~_k, y)                     (paper Eq. 5, summed over exits)
    L2 = sum_k || z~_k - z ||_1                (paper Eq. 5; z = the frozen model's final logits)
    L  = lambda * L1 + (1 - lambda) * L2       (paper Eq. 6)

Everything about the recipe is a guess (A15): epochs, optimizer, lr, schedule, batch size and
augmentation are all absent from the paper. App. D's "50 epochs" is the MCUNet BACKBONE fine-tune,
not this stage. Defaults follow TinyTL [31] as cited: SGD, momentum 0.9, lr 0.05, cosine, batch 128.

DATA: CLEAN labelled source train split (TINYTTA.OFFLINE_SOURCE="clean"). This is a deliberate
divergence from offline_train_aura.py, which trains on CORRUPTED gradual-mixture streams: TinyTTA's
heads exist to reproduce the SOURCE function at intermediate depths, and training them on the test
corruption pool would be a leak. "corrupted_gradual" is not implemented here -- it is reachable only
by extending this file, and the checkpoint's `source_kind` field makes any such artifact
un-loadable by a "clean"-configured run.

Also runs the gamma calibration (methods/tinytta.py::calibrate_gammas) on a HELD-OUT slice of the
clean source TRAIN split, so the exit thresholds are out-of-sample and never touch test data.

Writes exit-head tensors + gammas + a topology signature + PER-EXIT CLEAN ACCURACY. The last of
these is the cheapest guard against the failure mode that looks like success: an untrained head has
entropy ~ln C, never fires, and silently reduces TinyTTA to entropy-SGD on the original classifier.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python offline_train_tinytta.py --cfg cfgs/cifar10_c/tinytta.yaml
    # short sanity pass:
    ... --cfg cfgs/cifar10_c/tinytta.yaml TINYTTA.OFFLINE_EPOCHS 1 SOURCE.NUM_SAMPLES 10000
"""

import logging
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from conf import cfg, load_cfg_from_args, get_num_classes
from datasets.data_loading import get_source_loader
from methods.tinytta import (calibrate_gammas, default_offline_ckpt_path)
from models.model import get_model
from utils.registry import ADAPTATION_REGISTRY

logger = logging.getLogger(__name__)


def _augmented_transform(base_transform):
    """Standard CIFAR augmentation prepended to the host's transform.

    The host's source transform for CIFAR is a plain ToTensor() (get_transform falls through for an
    unknown `adaptation`), which is correct: RobustBench's `Standard` expects un-normalized [0,1]
    inputs, and the ResNeXt normalizes internally inside its own forward (reproduced by the
    EcoTTA shim's stem). Do NOT add a Normalize here.
    """
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        base_transform,
    ])


@torch.no_grad()
def evaluate_exits(net, loader, device, num_classes):
    """Clean accuracy of every exit head, evaluated on ALL samples (no early exiting).

    Stored in the checkpoint and re-checked by TinyTTA._load_offline_ckpt: any exit at or near
    chance is a hard error there, because such a head can never fire.
    """
    net.eval()
    correct = [0 for _ in range(net.K)]
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        _, _, _, evaluated = net.run_exits(imgs, count=False, force_full=True)
        for k in range(net.K):
            rows, z = evaluated[k]
            assert rows.numel() == imgs.shape[0]
            correct[k] += int((z.argmax(1) == labels).sum().item())
        total += imgs.shape[0]
    return [c / max(1, total) for c in correct]


def train_offline():
    load_cfg_from_args("TinyTTA offline self-ensemble training (paper Sec. 3.4)")

    if cfg.TINYTTA.OFFLINE_SOURCE != "clean":
        raise NotImplementedError(
            f"TINYTTA.OFFLINE_SOURCE={cfg.TINYTTA.OFFLINE_SOURCE!r} is not implemented in this "
            f"entrypoint. Only 'clean' (the paper-faithful clean labelled source train split) is "
            f"supported; a corrupted-source variant would be a declared leak and needs its own "
            f"code path."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)

    base_model, model_preprocess = get_model(cfg, num_classes, device)
    base_model.model_preprocess = model_preprocess

    # Instantiate through the registry with LOAD_OFFLINE/REQUIRE_OFFLINE forced off. This reuses
    # TinyTTA.configure_model verbatim, so the offline stage and the online run see an IDENTICAL
    # module tree and identical state_dict keys by construction. Forcing LOAD_OFFLINE off matters:
    # without it a nominally fresh run would silently RESUME from whatever artifact sits at the
    # auto-derived path (observed during the EcoTTA port).
    cfg.defrost()
    cfg.TINYTTA.LOAD_OFFLINE = False
    cfg.TINYTTA.REQUIRE_OFFLINE = False
    cfg.TINYTTA.GAMMA_MODE = "scalar"          # placeholder during training; the exit rule is unused
    cfg.TINYTTA.GAMMA_SCALAR = 1e-9            # ... and cannot fire, so nothing depends on it
    cfg.freeze()

    method = ADAPTATION_REGISTRY.get("tinytta")(cfg=cfg, model=base_model,
                                                num_classes=num_classes)
    net = method.model
    logger.info(f"Topology signature: {net.topology_signature()}")

    # ---- data: clean labelled source TRAIN split -------------------------------------------
    workers = min(cfg.TINYTTA.OFFLINE_WORKERS, os.cpu_count() or 1)
    train_dataset, _ = get_source_loader(
        dataset_name=cfg.CORRUPTION.DATASET,
        adaptation="tinytta",
        preprocess=model_preprocess,
        data_root_dir=cfg.DATA_DIR,
        batch_size=cfg.TINYTTA.OFFLINE_BATCH_SIZE,
        use_clip=False,
        train_split=True,
        num_samples=cfg.SOURCE.NUM_SAMPLES,
        percentage=cfg.SOURCE.PERCENTAGE,
        workers=workers,
    )
    base_transform = train_dataset.transform
    aug_transform = _augmented_transform(base_transform) if cfg.TINYTTA.OFFLINE_AUGMENT \
        else base_transform

    def set_transform(augment: bool):
        """Toggle augmentation on the ONE shared dataset object.

        Deliberately not a second get_source_loader call: when SOURCE.NUM_SAMPLES / PERCENTAGE
        reduce the dataset, get_source_loader subsamples with the global `random`, so a second call
        would return a DIFFERENT subset and the fit/calib index split below would no longer refer to
        the same images. (Note also a pre-existing host quirk: `percentage < 1` alone trips
        `assert num_samples > 0` in data_loading.py, so use SOURCE.NUM_SAMPLES to subsample.)
        """
        train_dataset.transform = aug_transform if augment else base_transform

    # Held-out calibration slice of the TRAIN split: the heads are fitted on `fit_idx`, the gammas
    # are chosen on `cal_idx`, and the TEST split is never touched.
    n = len(train_dataset)
    g = torch.Generator().manual_seed(cfg.RNG_SEED)
    perm = torch.randperm(n, generator=g).tolist()
    n_cal = int(round(cfg.TINYTTA.CALIB_SPLIT * n))
    cal_idx, fit_idx = perm[:n_cal], perm[n_cal:]
    logger.info(f"Source train split: {n} samples -> fit {len(fit_idx)} / calib {len(cal_idx)} "
                f"(TINYTTA.CALIB_SPLIT={cfg.TINYTTA.CALIB_SPLIT}, seed {cfg.RNG_SEED})")

    fit_loader = DataLoader(Subset(train_dataset, fit_idx),
                            batch_size=cfg.TINYTTA.OFFLINE_BATCH_SIZE, shuffle=True,
                            num_workers=workers, drop_last=False)
    cal_loader = DataLoader(Subset(train_dataset, cal_idx),
                            batch_size=cfg.TINYTTA.OFFLINE_BATCH_SIZE, shuffle=False,
                            num_workers=workers)

    # ---- what is trained -------------------------------------------------------------------
    if not cfg.TINYTTA.OFFLINE_FREEZE_BACKBONE:
        raise NotImplementedError(
            "TINYTTA.OFFLINE_FREEZE_BACKBONE=False (Eq. 6's literal min_theta) is not implemented. "
            "Sec. 3.1/3.3 say the submodules are frozen, and freezing is what makes L2 a well-posed "
            "distillation target."
        )
    clf = net.tail_classifier()
    if not cfg.TINYTTA.OFFLINE_TRAIN_EXIT_K and clf is not None:
        clf.requires_grad_(False)
    params = [p for p in net.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in net.parameters())
    logger.info(f"Offline trainable: {n_train:,} / {n_total:,} params "
                f"({100 * n_train / n_total:.3f}%); per-exit head params {net.head_params()}")

    optimizer = torch.optim.SGD(params, lr=cfg.TINYTTA.OFFLINE_LR,
                                momentum=cfg.TINYTTA.OFFLINE_MOMENTUM,
                                weight_decay=cfg.TINYTTA.OFFLINE_WD)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.TINYTTA.OFFLINE_EPOCHS))
        if cfg.TINYTTA.OFFLINE_SCHEDULE == "cosine" else None)

    lam = cfg.TINYTTA.OFFLINE_LAMBDA
    use_l2 = cfg.TINYTTA.OFFLINE_DISTILL_TARGET != "none"
    num_new = len(net.exits)
    logged_scales = False

    best = dict(mean_acc=-1.0)
    ckpt_path = cfg.TINYTTA.OFFLINE_CKPT or default_offline_ckpt_path(cfg)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    for epoch in range(1, cfg.TINYTTA.OFFLINE_EPOCHS + 1):
        # The frozen backbone stays in eval() throughout: its BN buffers must NOT drift, or the
        # heads would be fitted against normalization statistics that no longer exist at test time
        # (the EcoTTA warm-up/online mismatch). The heads contain no BN, so eval() costs nothing.
        net.eval()
        set_transform(cfg.TINYTTA.OFFLINE_AUGMENT)
        t0 = time.perf_counter()
        run_loss = run_ce = run_l2 = 0.0
        seen = 0
        for imgs, labels in fit_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            # One full forward with no early exiting: every exit is evaluated on every sample,
            # which is what "concurrent training of all models" (Sec. 3.4) requires.
            _, _, _, evaluated = net.run_exits(imgs, count=False, force_full=True)

            # z in L2: the frozen model's own final logits = exit K's logits, which are already
            # computed and (with the classifier frozen) parameter-independent. A17.
            with torch.no_grad():
                z_teacher = evaluated[net.K - 1][1].detach()

            ce = imgs.new_zeros(())
            l2 = imgs.new_zeros(())
            l2_sum_scale = imgs.new_zeros(())
            ks = range(num_new) if cfg.TINYTTA.OFFLINE_SUM_OVER_EXITS else [num_new - 1]
            for k in ks:
                z = evaluated[k][1]
                ce = ce + F.cross_entropy(z, labels)
                if use_l2:
                    diff = (z - z_teacher).abs()
                    l2_sum_scale = l2_sum_scale + diff.sum()
                    if cfg.TINYTTA.OFFLINE_L2_REDUCTION == "mean":
                        l2 = l2 + diff.mean()
                    elif cfg.TINYTTA.OFFLINE_L2_REDUCTION == "sum_over_classes_mean_over_batch":
                        l2 = l2 + diff.sum(1).mean()
                    else:                                        # "sum" -- the literal ||.||_1
                        l2 = l2 + diff.sum()

            loss = lam * ce + (1.0 - lam) * l2 if use_l2 else ce
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if not logged_scales:
                logged_scales = True
                # Eq. 5 prints ||.||_1. Log every reduction's scale once, because the EcoTTA port
                # showed that picking the wrong one is a silent, order-of-magnitude mis-weighting.
                logger.info(
                    f"L2 reduction scales on the first batch: mean={float(l2.item()):.6f} "
                    f"(active: {cfg.TINYTTA.OFFLINE_L2_REDUCTION}), literal_sum="
                    f"{float(l2_sum_scale.item()):.2f}, CE={float(ce.item()):.6f}, "
                    f"lambda={lam} -> lam*CE={lam * float(ce.item()):.6f}, "
                    f"(1-lam)*L2={(1 - lam) * float(l2.item()):.6f}"
                )

            bs = imgs.shape[0]
            run_loss += float(loss.item()) * bs
            run_ce += float(ce.item()) * bs
            run_l2 += float(l2.item()) * bs
            seen += bs

        if scheduler is not None:
            scheduler.step()

        set_transform(False)      # calibration / clean eval must not see augmentation
        accs = evaluate_exits(net, cal_loader, device, num_classes)
        mean_acc = sum(accs[:num_new]) / max(1, num_new)
        logger.info(
            f"epoch {epoch:3d}/{cfg.TINYTTA.OFFLINE_EPOCHS}  loss={run_loss / seen:.4f}  "
            f"CE={run_ce / seen:.4f}  L2={run_l2 / seen:.4f}  "
            f"calib_exit_acc={[round(a, 4) for a in accs]}  mean(new heads)={mean_acc:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.5f}  {time.perf_counter() - t0:.1f}s"
        )

        # Divergence guard. MEASURED: the shipped default OFFLINE_LR 0.05 (transplanted from
        # TinyTL [31] as App. D cites, A15) DIVERGES for these heads -- 1 epoch on 2,048 CIFAR-10
        # samples went from CE 9.64 at init to CE 914,483, with all exits at chance. lr 5e-3 on the
        # same data is stable (CE 10.63, exit accs 0.22/0.13/0.43, gammas 1.80/1.19/0.98). The
        # cause is structural: per-output-channel WS forces the exit conv's weight to unit variance,
        # so the head amplifies by ~sqrt(fan_in) and is already over-confident at init (CE 9.64 vs
        # ln 10 = 2.30). The default is deliberately NOT retuned here -- see the report -- but a
        # diverged run must not be mistaken for a trained one.
        if run_ce / seen > 3.0 * math.log(num_classes):
            logger.warning(
                f"OFFLINE STAGE LOOKS DIVERGED: epoch-{epoch} CE = {run_ce / seen:.2f} vs "
                f"ln C = {math.log(num_classes):.2f}. TINYTTA.OFFLINE_LR={cfg.TINYTTA.OFFLINE_LR} "
                f"is a GUESS (A15: the paper states no offline recipe at all) and 0.05 is measured "
                f"to diverge for WS exit heads. The resulting checkpoint will have chance-level "
                f"per_exit_clean_acc and will be REFUSED at load time by design."
            )

        if mean_acc > best["mean_acc"]:
            best = dict(
                mean_acc=mean_acc,
                epoch=epoch,
                accs=accs,
                exit_state={k: v.detach().cpu().clone()
                            for k, v in net.state_dict().items() if k.startswith("exits.")},
                tail_state=({k: v.detach().cpu().clone()
                             for k, v in net.state_dict().items() if k.startswith("tail.")}
                            if cfg.TINYTTA.OFFLINE_TRAIN_EXIT_K else {}),
            )

    # Restore the best epoch's heads before calibrating -- gammas must match the weights we ship.
    # (The EcoTTA review's MINOR-7 was exactly this: saving the last epoch while reporting the best.)
    net.load_state_dict(best["exit_state"], strict=False)
    if best["tail_state"]:
        net.load_state_dict(best["tail_state"], strict=False)
    logger.info(f"Selected epoch {best['epoch']} (mean new-head calib acc {best['mean_acc']:.4f})")

    set_transform(False)
    gammas, achieved, pcts = calibrate_gammas(
        net, cal_loader, device,
        target_fractions=list(cfg.TINYTTA.GAMMA_TARGET_FRACTIONS),
        max_samples=cfg.TINYTTA.CALIB_NUM_SAMPLES,
    )
    logger.info(f"Calibrated gammas (CLEAN held-out source train slice, nats): "
                f"{[round(g, 4) for g in gammas]}")
    logger.info(f"Achieved exit fractions on the calibration split: "
                f"{[round(a, 4) for a in achieved]}  (targets "
                f"{list(cfg.TINYTTA.GAMMA_TARGET_FRACTIONS) or ['uniform 1/K'] })")
    for k, p in pcts.items():
        logger.info(f"  exit {k} entropy percentiles [1,5,25,50,75,95,99]: "
                    f"{[round(v, 4) for v in p]}  (ln C = {torch.log(torch.tensor(float(num_classes))).item():.4f})")
    logger.warning(
        "Calibration is on CLEAN data; the test stream is corrupted, where entropies are "
        "systematically HIGHER. Expect the achieved test-time early-exit fractions to be BELOW "
        "these targets. That is the leak-free behaviour, not a defect -- check the runtime "
        "EXIT-DIST lines."
    )

    payload = dict(
        exit_state=best["exit_state"],
        tail_state=best["tail_state"],
        gammas=gammas,
        gamma_target_frac=list(cfg.TINYTTA.GAMMA_TARGET_FRACTIONS),
        gamma_achieved_frac=achieved,
        calib_entropy_pcts={int(k): v for k, v in pcts.items()},
        per_exit_clean_acc=best["accs"],
        topology_signature=net.topology_signature(),
        arch=cfg.MODEL.ARCH,
        dataset=cfg.CORRUPTION.DATASET,
        num_classes=num_classes,
        K=cfg.TINYTTA.K,
        partition=list(net.partition),
        taps=list(net.taps),
        source_kind=cfg.TINYTTA.OFFLINE_SOURCE,
        ws_mode=cfg.TINYTTA.WS_MODE,
        ws_eps=cfg.TINYTTA.WS_EPS,
        ws_eps_in_sqrt=cfg.TINYTTA.WS_EPS_IN_SQRT,
        offline_epochs=cfg.TINYTTA.OFFLINE_EPOCHS,
        offline_best_epoch=best["epoch"],
        offline_lr=cfg.TINYTTA.OFFLINE_LR,
        offline_lambda=lam,
        offline_l2_reduction=cfg.TINYTTA.OFFLINE_L2_REDUCTION,
        offline_batch_size=cfg.TINYTTA.OFFLINE_BATCH_SIZE,
        freeze_backbone=cfg.TINYTTA.OFFLINE_FREEZE_BACKBONE,
        train_exit_k=cfg.TINYTTA.OFFLINE_TRAIN_EXIT_K,
        source_percentage=cfg.SOURCE.PERCENTAGE,
        calib_split=cfg.TINYTTA.CALIB_SPLIT,
        rng_seed=cfg.RNG_SEED,
        cfg_dump=str(cfg),
    )
    torch.save(payload, ckpt_path)
    size_mb = os.path.getsize(ckpt_path) / 1e6
    logger.info(f"Saved TinyTTA offline checkpoint ({size_mb:.2f} MB) to:\n    {ckpt_path}")
    logger.info("Consume it by leaving TINYTTA.OFFLINE_CKPT empty (the path is auto-derived), or "
                f"paste:\n    OFFLINE_CKPT: {ckpt_path}")


if __name__ == "__main__":
    train_offline()
