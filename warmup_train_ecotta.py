"""
EcoTTA warm-up: pre-train the meta networks on SOURCE data before test-time adaptation.

Paper (arXiv 2303.01904, Sec. 3.2 + App. D): the meta networks are randomly initialised, so
they are first trained on the source dataset with plain cross-entropy while the pre-trained
encoder stays frozen. CIFAR: 10 epochs, SGD lr 5e-2. Photometric augmentation
(ColorJitter(0.4) / GaussianBlur / RandomGrayscale) is applied so the meta networks do not
overfit the clean source distribution.

This is NOT optional. EcoTTA's meta networks are summed into the frozen feature path, so a
cold-started EcoTTA is near-random rather than merely worse.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python warmup_train_ecotta.py --cfg cfgs/cifar10_c/ecotta.yaml

    # short smoke-test warm-up
    CUDA_VISIBLE_DEVICES=0 python warmup_train_ecotta.py --cfg cfgs/cifar10_c/ecotta.yaml \
        ECOTTA.WARMUP_EPOCHS 1

The checkpoint path is auto-derived by methods.ecotta.default_warmup_path(cfg) -- the same
helper EcoTTA.configure_model() reads -- so a matching online run picks it up with
ECOTTA.WARMUP_CKPT left empty.
"""

import os
import time
import logging

import torch
import torch.nn.functional as F
from torchvision import transforms

import methods  # noqa: F401  (populates ADAPTATION_REGISTRY)
from conf import cfg, load_cfg_from_args, get_num_classes
from datasets.data_loading import get_source_loader
from models.model import get_model
from utils.registry import ADAPTATION_REGISTRY
from methods.ecotta import default_warmup_path

logger = logging.getLogger(__name__)


def build_warmup_transform(base_transform, augment: bool):
    """Prepend the paper's photometric augmentations to the source transform.

    The host's CIFAR source transform is a plain Compose([ToTensor()]) (see
    datasets/data_loading.py get_transform). No Normalize is added: RobustBench `Standard`
    consumes un-normalized [0, 1] inputs (it has no ImageNormalizer wrapper).
    """
    if not augment:
        return base_transform

    augs = [
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.4),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.2),
        transforms.RandomGrayscale(p=0.1),
    ]
    tail = list(base_transform.transforms) if isinstance(base_transform, transforms.Compose) \
        else [base_transform]
    return transforms.Compose(augs + tail)


@torch.no_grad()
def evaluate_clean(method, loader, device):
    """Clean source-test accuracy in the SAME mode configuration used online.

    (frozen BNs per ECOTTA.FROZEN_BN_MODE, meta BNs in train mode / test-batch statistics),
    so the number is directly predictive of the online run rather than of a mode EcoTTA
    never uses.

    The meta BNs stay in train mode, which means this pass would otherwise fold CIFAR-10 TEST
    statistics into their running buffers -- buffers that then get persisted in the checkpoint.
    Harmless online (train mode never reads them), but it is a genuine test-set leak if anyone
    ever evaluates with the meta BNs in eval mode. So snapshot and restore them around the pass.
    """
    method.model.set_compute_reg(False)

    meta_bns = [bn for meta in method.model.meta_modules()
                for bn in (meta.meta_bn, meta.conv_bn)]
    saved = [(bn.running_mean.clone(), bn.running_var.clone(), bn.num_batches_tracked.clone())
             for bn in meta_bns]

    correct = total = 0
    for batch in loader:
        imgs, labels = batch[0].to(device), batch[1].to(device)
        preds = method.model(imgs).argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.numel()

    for bn, (rm, rv, nbt) in zip(meta_bns, saved):
        bn.running_mean.copy_(rm)
        bn.running_var.copy_(rv)
        bn.num_batches_tracked.copy_(nbt)

    return correct / max(total, 1)


def main():
    load_cfg_from_args("EcoTTA meta-network warm-up on source data.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)

    save_path = default_warmup_path(cfg)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # The method under construction must neither demand nor LOAD the checkpoint we are about to
    # create: warm-up has to start from the random meta-net init. Without LOAD_WARMUP=False a
    # re-run would silently resume from the existing file and restart the lr schedule on top of
    # it, so "10 epochs" would really be "previous epochs + 10".
    cfg.defrost()
    cfg.ECOTTA.REQUIRE_WARMUP = False
    cfg.ECOTTA.LOAD_WARMUP = False
    cfg.ECOTTA.WARMUP_CKPT = ""
    cfg.freeze()

    base_model, model_preprocess = get_model(cfg, num_classes, device)
    base_model.model_preprocess = model_preprocess

    # Build through the registry so warm-up and the online run share EXACTLY the same
    # group-splitting code and therefore identical state_dict keys.
    method = ADAPTATION_REGISTRY.get("ecotta")(cfg=cfg, model=base_model, num_classes=num_classes)
    logger.info(f"Meta-network geometry (c_in, c_out, stride) per group: {method.model.geometry()}")
    logger.info(f"Topology: {method.model.topology_signature()}")

    # ---- source data ----------------------------------------------------------
    train_dataset, train_loader = get_source_loader(
        dataset_name=cfg.CORRUPTION.DATASET,
        adaptation=cfg.MODEL.ADAPTATION,
        preprocess=model_preprocess,
        data_root_dir=cfg.DATA_DIR,
        batch_size=cfg.ECOTTA.WARMUP_BATCH_SIZE,
        use_clip=False,
        train_split=True,
        ckpt_path=cfg.MODEL.CKPT_PATH,
        num_samples=cfg.SOURCE.NUM_SAMPLES,
        percentage=cfg.SOURCE.PERCENTAGE,
        workers=min(cfg.SOURCE.NUM_WORKERS, os.cpu_count()),
    )
    train_dataset.transform = build_warmup_transform(train_dataset.transform, cfg.ECOTTA.WARMUP_AUGMENT)
    logger.info(f"Warm-up train transform:\n{train_dataset.transform}")

    _, test_loader = get_source_loader(
        dataset_name=cfg.CORRUPTION.DATASET,
        adaptation=cfg.MODEL.ADAPTATION,
        preprocess=model_preprocess,
        data_root_dir=cfg.DATA_DIR,
        batch_size=cfg.ECOTTA.WARMUP_BATCH_SIZE,
        use_clip=False,
        train_split=False,
        ckpt_path=cfg.MODEL.CKPT_PATH,
        workers=min(cfg.SOURCE.NUM_WORKERS, os.cpu_count()),
    )

    if cfg.ECOTTA.WARMUP_IDENTITY_INIT:
        from methods.ecotta import calibrate_identity_init
        calibrate_identity_init(method.model, train_loader, device, num_batches=50)
        apply_bn_modes_fn = __import__("methods.ecotta", fromlist=["apply_bn_modes"]).apply_bn_modes
        apply_bn_modes_fn(method.model, cfg.ECOTTA.WARMUP_BN_MODE)
        acc0 = evaluate_clean(method, test_loader, device)
        logger.info(f"Identity init: clean test acc BEFORE any warm-up step = {acc0:.4f} "
                    f"(the source model's own clean accuracy is the target)")

    # ---- optimizer / schedule (paper: SGD 5e-2, momentum 0.9, MultiStepLR) ----
    optimizer = torch.optim.SGD(method.params,
                                lr=cfg.ECOTTA.WARMUP_LR,
                                momentum=cfg.ECOTTA.WARMUP_MOMENTUM)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(cfg.ECOTTA.WARMUP_LR_MILESTONES),
        gamma=cfg.ECOTTA.WARMUP_LR_GAMMA,
    )
    iters_per_epoch = len(train_loader)
    linear_iters = cfg.ECOTTA.WARMUP_LINEAR_EPOCHS * iters_per_epoch

    num_meta = sum(p.numel() for p in method.params)
    num_total = sum(p.numel() for p in method.model.parameters())
    logger.info(f"Warm-up: {cfg.ECOTTA.WARMUP_EPOCHS} epoch(s), bs={cfg.ECOTTA.WARMUP_BATCH_SIZE}, "
                f"lr={cfg.ECOTTA.WARMUP_LR}, {iters_per_epoch} iters/epoch, "
                f"trainable meta params {num_meta:,}/{num_total:,}")

    # Warm-up trains the meta networks with plain cross-entropy; no regularization term.
    # Detaching MUST stay off: cross-entropy is computed on the final logits, so with
    # Algorithm 1's `Detach_parts` active the gradient would reach only the deepest meta
    # network and every shallower one would keep its random initialisation.
    method.model.set_compute_reg(False)
    method.model.set_detach(False)

    global_iter = 0
    best_acc = -1.0
    best_state = None
    best_epoch = -1
    for epoch in range(1, cfg.ECOTTA.WARMUP_EPOCHS + 1):
        # Re-assert modes every epoch: evaluate_clean() does not change them, but be explicit.
        from methods.ecotta import apply_bn_modes
        apply_bn_modes(method.model, cfg.ECOTTA.WARMUP_BN_MODE)
        method.model.set_compute_reg(False)

        running_loss, running_correct, running_seen = 0.0, 0, 0
        t_epoch = time.time()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            if global_iter < linear_iters:  # linear lr ramp (reference WarmUpLR)
                lr = cfg.ECOTTA.WARMUP_LR * (global_iter + 1) / linear_iters
                for group in optimizer.param_groups:
                    group["lr"] = lr

            with torch.enable_grad():
                logits = method.model(imgs)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * labels.numel()
            running_correct += (logits.argmax(1) == labels).sum().item()
            running_seen += labels.numel()
            global_iter += 1

            if global_iter % 100 == 0:
                logger.info(f"epoch {epoch} iter {global_iter} "
                            f"loss {running_loss / running_seen:.4f} "
                            f"train_acc {running_correct / running_seen:.4f} "
                            f"lr {optimizer.param_groups[0]['lr']:.5f}")

        if global_iter >= linear_iters:
            scheduler.step()

        clean_acc = evaluate_clean(method, test_loader, device)
        # Real best-checkpoint selection: previously `src_acc` advertised the max over epochs
        # while the tensors saved were the FINAL epoch's, so the logged warm-up quality could
        # describe weights the checkpoint did not contain.
        if clean_acc > best_acc:
            best_acc, best_epoch = clean_acc, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in method.model.state_dict().items()}
        logger.info(f"[epoch {epoch}/{cfg.ECOTTA.WARMUP_EPOCHS}] "
                    f"train_loss {running_loss / running_seen:.4f} "
                    f"train_acc {running_correct / running_seen:.4f} "
                    f"clean_test_acc {clean_acc:.4f} "
                    f"({time.time() - t_epoch:.1f}s)")

    # ---- save (meta tensors only: ~14 MB for K=5, vs. 583 MB for a full state dict) ----
    # Save the BEST epoch's weights, which is what `src_acc` then genuinely describes.
    full_sd = best_state if best_state is not None else method.model.state_dict()
    meta_state = {k: v.detach().cpu().clone() for k, v in full_sd.items() if ".meta." in k}
    logger.info(f"Selected epoch {best_epoch} (clean_test_acc {best_acc:.4f}) for the checkpoint.")

    frozen_bn_buffers = {}
    if cfg.ECOTTA.WARMUP_BN_MODE == "train":
        # The frozen BN running statistics drifted during warm-up. Persist them, otherwise the
        # online run would reload pristine RobustBench statistics and the meta networks would be
        # operating against normalization they were never trained against.
        frozen_bn_buffers = {
            k: v.detach().cpu().clone() for k, v in full_sd.items()
            if ".meta." not in k and k.split(".")[-1] in
            ("running_mean", "running_var", "num_batches_tracked")
        }
        logger.warning(f"WARMUP_BN_MODE='train': persisting {len(frozen_bn_buffers)} frozen BN "
                       f"buffers alongside the meta weights.")

    torch.save({
        "meta_state": meta_state,
        "frozen_bn_buffers": frozen_bn_buffers,
        "arch": cfg.MODEL.ARCH,
        "dataset": cfg.CORRUPTION.DATASET,
        "K": cfg.ECOTTA.K,
        "num_classes": num_classes,
        "bn_placement": cfg.ECOTTA.BN_PLACEMENT,
        # Enforced on load: a warm-up artifact is only valid for the topology it was trained on.
        "topology_signature": method.model.topology_signature(),
        "best_epoch": best_epoch,
        "warmup_epochs": cfg.ECOTTA.WARMUP_EPOCHS,
        "warmup_lr": cfg.ECOTTA.WARMUP_LR,
        "warmup_batch_size": cfg.ECOTTA.WARMUP_BATCH_SIZE,
        "warmup_bn_mode": cfg.ECOTTA.WARMUP_BN_MODE,
        "warmup_augment": cfg.ECOTTA.WARMUP_AUGMENT,
        "src_acc": best_acc,
        "rng_seed": cfg.RNG_SEED,
        "num_meta_params": num_meta,
    }, save_path)

    size_mb = os.path.getsize(save_path) / 1e6
    logger.info(f"Saved EcoTTA warm-up checkpoint to {save_path} ({size_mb:.1f} MB), "
                f"best clean test acc {best_acc:.4f}")


if __name__ == "__main__":
    main()
