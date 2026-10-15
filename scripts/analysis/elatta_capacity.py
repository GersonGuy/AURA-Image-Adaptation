"""Is ELaTTA's small CIFAR-10 gain a CAPACITY limit or a PROXY limit?

Two measurements, both on CIFAR-10-C severity 5 / WRN-28-10 (D=640, C=10):

  (1) ORACLE CEILING. Replace the entropy objective with the TRUE LABEL and find the best
      achievable p under a norm budget ||p|| <= r. This upper-bounds ANY search strategy over the
      latent-offset family z -> z + V_k p, entropy-based or otherwise. If the ceiling sits just
      above source accuracy, the k-dim subspace has no correction capacity and no objective can do
      better. If it sits far above, the entropy proxy is the bottleneck instead.

      The UNCONSTRAINED oracle is vacuous and is therefore not reported: p may be scaled without
      limit, so whenever some direction u satisfies (A u)_y > (A u)_c for all c != y (A = W V_k), a
      large enough multiple of u dominates the initial logits and the sample is "correctable"
      regardless of its features. The norm budget is what makes the question meaningful, so r is
      swept in units of the source coefficient std.

  (2) PROP. 3.1 ALIGNMENT. The fraction of each sample's corruption-induced feature shift that lies
      inside span(V_k):  ||P_Vk xi||^2 / ||xi||^2  with  xi = z_corrupt - z_clean.  Only the
      in-subspace component is correctable, so this is the geometric reason a clean-variance basis
      may or may not be able to undo a corruption. CIFAR-10-C is generated from the CIFAR-10 test
      set in index order, which is what makes the pairing exact.

Two shortcuts keep this cheap:
  * The PCA basis is NESTED -- V_k is the first k columns of the same SVD -- so one basis at
    k=K_MAX is built and sliced, rather than rebuilding per k.
  * The oracle is solved by BATCHED projected gradient ascent on the true-class margin, all samples
    at once on the GPU, instead of one LP per sample. This is a LOWER bound on the true oracle
    (a local optimum of a max-min of linear functions over a ball), so any headroom it reports is
    real; it cannot overstate the ceiling. Reported alongside a random-restart check.

Standalone analysis script: imports the method's own basis builder and model split so the geometry
is identical to what methods/elatta.py uses. Modifies nothing.
"""
import argparse
import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from conf import cfg, load_cfg_from_args  # noqa: E402  (repo-local import order)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CORRUPTIONS = ["gaussian_noise", "motion_blur", "pixelate", "contrast", "fog"]
K_VALUES = [1, 2, 4, 8, 16, 32, 64]
K_MAX = 64
N_PER_CORRUPTION = 2000
SEVERITY = 5
ORACLE_STEPS = 300


def build(cfg_path="cfgs/cifar10_c/elatta.yaml"):
    """Model + encoder/classifier split + nested basis, all via the method's own code paths."""
    sys.argv = ["analysis", "--cfg", cfg_path, "ELATTA.K", str(K_MAX)]
    load_cfg_from_args()   # freezes cfg
    cfg.defrost()
    cfg.ELATTA.K = K_MAX
    cfg.ELATTA.LOG_EVERY = 10 ** 9
    cfg.freeze()

    from models.model import get_model
    from methods.elatta import ELaTTA

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _preprocess = get_model(cfg, num_classes=10, device=device)
    method = ELaTTA(cfg, model, num_classes=10)
    return method


@torch.no_grad()
def features(method, imgs, bs=200):
    """Pre-logit features for a uint8 NHWC array, using the method's own encoder."""
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(imgs[i:i + bs]).float().div_(255.).permute(0, 3, 1, 2).to(method.device)
        out.append(method.encoder(x).flatten(1))
    return torch.cat(out)


def oracle_accuracy(A, logits0, y, r, steps=ORACLE_STEPS, restarts=2):
    """Best achievable accuracy for logits0 + A p subject to ||p|| <= r.

    A: [C, k]  logits0: [B, C]  y: [B]
    Maximizes the true-class margin by projected gradient ascent, batched over all B samples.
    A LOWER bound on the true oracle, so reported headroom is real.
    """
    B, C = logits0.shape
    k = A.shape[1]
    best = torch.zeros(B, dtype=torch.bool, device=logits0.device)
    for rs in range(restarts):
        p = (torch.zeros(B, k, device=logits0.device) if rs == 0
             else torch.randn(B, k, device=logits0.device) * (r / max(1, k) ** 0.5))
        p.requires_grad_(True)
        opt = torch.optim.Adam([p], lr=max(r / 10, 1e-4))
        for _ in range(steps):
            opt.zero_grad()
            lg = logits0 + p @ A.T                      # [B, C]
            true = lg.gather(1, y[:, None]).squeeze(1)
            other = lg.masked_fill(
                torch.zeros_like(lg, dtype=torch.bool).scatter_(1, y[:, None], True), -1e9).max(1).values
            (-(true - other)).sum().backward()          # maximize the margin
            opt.step()
            with torch.no_grad():                       # project onto the ball ||p|| <= r
                nrm = p.norm(dim=1, keepdim=True).clamp_min(1e-12)
                p.mul_(torch.clamp(r / nrm, max=1.0))
        with torch.no_grad():
            best |= ((logits0 + p @ A.T).argmax(1) == y)
    return best.float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/elatta/_capacity_analysis.txt")
    args = ap.parse_args()

    method = build()
    W = method.classifier.weight.detach()                        # [C, D]
    V_full = method.V.detach().float()                           # [D, K_MAX]
    V_full = V_full / V_full.norm(dim=0, keepdim=True)           # force orthonormal cols
                                                                # (the cfg may ship 'whitened')
    coeff_std = float(method.coeff_std.mean())
    device = method.device
    logger.info(f"D={V_full.shape[0]} K_MAX={V_full.shape[1]} C={W.shape[0]} "
                f"mean source coeff std={coeff_std:.4f}")

    labels_all = np.load("data/CIFAR-10-C/labels.npy")
    # Clean CIFAR-10 TEST images, in the same index order CIFAR-10-C was generated from -- that
    # ordering is what makes the xi = z_corrupt - z_clean pairing exact.
    from torchvision.datasets import CIFAR10
    _ds = CIFAR10(root=cfg.DATA_DIR, train=False, download=True)
    clean = _ds.data                                             # uint8 NHWC, same order as -C
    assert (np.asarray(_ds.targets)[:N_PER_CORRUPTION]
            == labels_all[:N_PER_CORRUPTION]).all(), \
        "clean/corrupted label order mismatch -- the index pairing assumption is wrong"

    lines = []

    def emit(s):
        logger.info(s)
        lines.append(s)

    emit("# ELaTTA capacity vs proxy analysis -- CIFAR-10-C severity 5, WRN-28-10")
    emit(f"# {N_PER_CORRUPTION} samples/corruption, corruptions={CORRUPTIONS}")
    emit(f"# basis: clean-source PCA, N={method.basis_meta.get('N','?')}, orthonormal, nested slices of "
         f"k={K_MAX}")
    emit(f"# oracle = batched projected gradient ascent on the TRUE-class margin, ||p||<=r, "
         f"{ORACLE_STEPS} steps x 2 restarts (a LOWER bound on the true oracle)")
    emit(f"# r reported in units of the mean source coefficient std ({coeff_std:.4f})")
    emit("")

    # rank(W V_k) governs whether entropy is exactly minimizable (surjectivity onto R^C).
    emit("## rank(W V_k) -- surjectivity onto R^C=10 is what makes the entropy objective degenerate")
    for k in K_VALUES:
        r_ = torch.linalg.matrix_rank(W @ V_full[:, :k]).item()
        emit(f"  k={k:3d}  rank(W V_k) = {r_:2d}   {'SURJECTIVE (degenerate)' if r_ >= 10 else ''}")
    emit("")

    per_corr = {}
    for corr in CORRUPTIONS:
        arr = np.load(f"data/CIFAR-10-C/{corr}.npy")
        lo = (SEVERITY - 1) * 10000
        imgs = arr[lo:lo + N_PER_CORRUPTION]
        y = torch.from_numpy(labels_all[lo:lo + N_PER_CORRUPTION]).long().to(device)
        z = features(method, imgs)
        logits0 = method.classifier(z).detach()
        src_acc = (logits0.argmax(1) == y).float().mean().item()

        row = {"source_acc": src_acc, "oracle": {}, "align": {}}

        # Prop 3.1 alignment, using the exact index pairing between CIFAR-10-C and clean test.
        if clean is not None:
            z_clean = features(method, clean[:N_PER_CORRUPTION])
            xi = (z - z_clean).double()
            xi_sq = (xi ** 2).sum(1).clamp_min(1e-12)
            for k in K_VALUES:
                Vk = V_full[:, :k].double()
                proj_sq = ((xi @ Vk) ** 2).sum(1)
                row["align"][k] = (proj_sq / xi_sq).mean().item()

        A = (W @ V_full[:, :K_MAX]).detach()
        for k in K_VALUES:
            Ak = A[:, :k].contiguous()
            for mult in (0.5, 1.0, 2.0, 5.0):
                row["oracle"][(k, mult)] = oracle_accuracy(Ak, logits0, y, r=mult * coeff_std)
        per_corr[corr] = row
        emit(f"  [{corr}] source acc {src_acc:.4f}")

    emit("")
    emit("## ORACLE CEILING -- accuracy, mean over corruptions (source in the r=0 column)")
    src_mean = np.mean([per_corr[c]["source_acc"] for c in CORRUPTIONS])
    emit(f"{'k':>4s} {'r=0 (source)':>13s} {'r=0.5sd':>9s} {'r=1sd':>9s} {'r=2sd':>9s} {'r=5sd':>9s}")
    for k in K_VALUES:
        cells = [np.mean([per_corr[c]["oracle"][(k, m)] for c in CORRUPTIONS])
                 for m in (0.5, 1.0, 2.0, 5.0)]
        emit(f"{k:4d} {src_mean:13.4f} " + " ".join(f"{v:9.4f}" for v in cells))

    if clean is not None:
        emit("")
        emit("## PROP 3.1 -- mean per-sample fraction of shift energy inside span(V_k)")
        emit(f"{'k':>4s} " + " ".join(f"{c[:9]:>10s}" for c in CORRUPTIONS) + f" {'mean':>8s}")
        for k in K_VALUES:
            vals = [per_corr[c]["align"][k] for c in CORRUPTIONS]
            emit(f"{k:4d} " + " ".join(f"{v:10.4f}" for v in vals) + f" {np.mean(vals):8.4f}")
    else:
        emit("")
        emit("## PROP 3.1 -- SKIPPED: no clean test array found for exact index pairing.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
