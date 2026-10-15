"""Build a SHIFT-ALIGNED basis for ELaTTA and save it in the method's own checkpoint format.

NOT THE PAPER'S METHOD. ELaTTA's basis is the PCA of CLEAN source features; the offline phase
consumes no target data at all. This script instead builds the basis from CORRUPTED data, which
breaks that source-only premise. It exists to answer a specific measured question, and anything
produced with it must be labelled "ELaTTA + corrupted-data basis", never as a reproduction.

Why: `docs/elatta/_capacity_analysis.txt` shows the latent-offset family is PROXY-limited, not
capacity-limited. The clean-PCA subspace can express a large fix (+22.7 accuracy points at k=8 with
a generous norm budget) but only at large ||p||, where the entropy objective degenerates; and raising
k to reach that capacity makes entropy strictly worse (`_sweep_k8_bigN.txt`). So the search has to
stay low-dimensional, which means the few dimensions it gets must be the USEFUL ones.

Prop. 3.1 quantifies "useful": only the component of the shift xi = z_corrupt - z_clean lying inside
span(V_k) is correctable. The clean-PCA basis captures just 16 % of shift energy at k=2. A basis
built from the shift vectors themselves maximizes that fraction per dimension by construction.

Basis modes:
  shift   -- SVD of the raw (UNCENTERED) matrix of shift vectors. Uncentered on purpose: the MEAN
             shift is itself the single most useful correction direction, and centering would throw
             it away into the discarded mean.
  corrupt -- PCA of centered CORRUPTED features (the weaker variant: where corrupted features vary,
             not where they moved).

Leakage control: the basis is built on a corruption pool DISJOINT from the evaluation pool, mirroring
AURA's own `OFFLINE.CORRUPTION_POOL` domain-generalization hook (offline_train_aura.py). Held-out
corruption types are the honest headline. Note the underlying IMAGES do overlap with the evaluation
set, because CIFAR-10-C is generated from the CIFAR-10 test split and this box has no corruption
generator to apply to the train split -- this is the same position AURA's offline phase is in, so the
comparison is fair, but it is transductive in the inputs and must be stated. No LABELS are used.
"""
import argparse
import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from conf import cfg, load_cfg_from_args  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# The repo's two established pools, used here as a disjoint offline/online split.
BASIS_POOL = ["zoom_blur", "snow", "frost", "fog", "brightness",
              "contrast", "elastic_transform", "pixelate", "jpeg_compression"]
EVAL_POOL = ["gaussian_noise", "shot_noise", "impulse_noise",
             "defocus_blur", "glass_blur", "motion_blur"]


@torch.no_grad()
def features(encoder, imgs, device, bs=200):
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(imgs[i:i + bs]).float().div_(255.).permute(0, 3, 1, 2).to(device)
        out.append(encoder(x).flatten(1).double().cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8, help="max k to store; the basis is nested so a "
                                                     "smaller k is just the first columns")
    ap.add_argument("--mode", choices=["shift", "corrupt"], default="shift")
    ap.add_argument("--per-corruption", type=int, default=2000)
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sys.argv = ["build", "--cfg", "cfgs/cifar10_c/elatta.yaml"]
    load_cfg_from_args()
    cfg.defrost()
    cfg.ELATTA.LOG_EVERY = 10 ** 9
    cfg.freeze()

    from models.model import get_model
    from methods.elatta import ELaTTA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = get_model(cfg, num_classes=10, device=device)
    method = ELaTTA(cfg, model, num_classes=10)
    encoder = method.encoder

    from torchvision.datasets import CIFAR10
    ds = CIFAR10(root=cfg.DATA_DIR, train=False, download=True)
    clean_imgs = ds.data[:args.per_corruption]
    z_clean = features(encoder, clean_imgs, device)

    lo = (args.severity - 1) * 10000
    rows = []
    for corr in BASIS_POOL:
        arr = np.load(f"data/CIFAR-10-C/{corr}.npy")[lo:lo + args.per_corruption]
        z_c = features(encoder, arr, device)
        rows.append(z_c - z_clean if args.mode == "shift" else z_c)
        logger.info(f"  {corr}: mean ||xi|| = {float((z_c - z_clean).norm(dim=1).mean()):.3f}")
    M = torch.cat(rows, dim=0)                                   # [N_eff, D]
    N_eff, D = M.shape

    if args.mode == "corrupt":
        M = M - M.mean(0)                                        # PCA of corrupted features
    # 'shift' stays UNCENTERED so the mean shift direction survives as the top component.
    _, S, Vh = torch.linalg.svd(M, full_matrices=False)
    k = min(args.k, Vh.shape[0])
    V_k = Vh[:k].T.contiguous()
    S_k = S[:k].contiguous()
    coeff_std = S_k / np.sqrt(max(1, N_eff - 1))

    payload = {
        "V_k": V_k.float(),
        "singular_values": S_k.float(),
        "coeff_std": coeff_std.float(),
        "sigma0_auto_orthonormal": float(coeff_std.mean().item()),
        "mu_s": M.mean(0).float(),
        "D": int(D), "k": int(k), "N": int(N_eff),
        "arch": cfg.MODEL.ARCH, "dataset": cfg.CORRUPTION.DATASET,
        "seed": int(cfg.RNG_SEED),
        # Self-documenting: anyone loading this must see it is not the paper's basis.
        "source_split": f"NON-PAPER-{args.mode}-basis",
        "sample_order": "all",
        "sample_indices": [],
        "explained_variance_ratio": float(((S_k ** 2).sum() / (S ** 2).sum()).item()),
        "basis_pool": BASIS_POOL,
        "eval_pool_heldout": EVAL_POOL,
        "note": ("NOT the paper's clean-source basis. Built from CORRUPTED data on a corruption "
                 "pool disjoint from the evaluation pool. No labels used. Inputs overlap the test "
                 "images (CIFAR-10-C is generated from the CIFAR-10 test split), so this is "
                 "transductive in the inputs -- the same position AURA's offline phase is in."),
    }

    out = args.out or f"ckpt/elatta/SHIFTBASIS_{args.mode}__cifar10_c__{cfg.MODEL.ARCH}__k{k}.pth"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(payload, out)
    logger.info(f"\nmode={args.mode} D={D} k={k} N={N_eff} "
                f"explained_var={payload['explained_variance_ratio'] * 100:.2f}% "
                f"coeff_std mean={float(coeff_std.mean()):.4f}")

    # Prop 3.1 on the HELD-OUT pool: how much of an unseen corruption's shift does this basis span?
    logger.info("\nProp 3.1 alignment on the HELD-OUT eval pool (unseen corruption types):")
    V_full = V_k.double()
    hdr = f"{'k':>3s} " + " ".join(f"{c[:9]:>10s}" for c in EVAL_POOL) + f" {'mean':>8s}"
    logger.info(hdr)
    lines = [hdr]
    for kk in [1, 2, 4, 8]:
        if kk > k:
            break
        fr = []
        for corr in EVAL_POOL:
            arr = np.load(f"data/CIFAR-10-C/{corr}.npy")[lo:lo + args.per_corruption]
            xi = (features(encoder, arr, device) - z_clean).double()
            Vk = V_full[:, :kk]
            fr.append(float((((xi @ Vk) ** 2).sum(1) / (xi ** 2).sum(1).clamp_min(1e-12)).mean()))
        row = f"{kk:3d} " + " ".join(f"{v:10.4f}" for v in fr) + f" {np.mean(fr):8.4f}"
        logger.info(row)
        lines.append(row)

    with open("docs/elatta/_shift_basis_alignment.txt", "a") as f:
        f.write(f"\n## mode={args.mode} k={k} N={N_eff} basis_pool={BASIS_POOL}\n")
        f.write("\n".join(lines) + "\n")
    logger.info(f"\nsaved basis to {out}")


if __name__ == "__main__":
    main()
