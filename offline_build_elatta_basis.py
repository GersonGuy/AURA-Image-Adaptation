"""
Offline phase for ELaTTA (paper Sec. "Offline phase", steps 1-4).

MANDATORY: the paper has no cold-start variant -- without source data there is no V_k
(docs/elatta/paper_spec.md hazard #9). It is also almost free: N = 20 clean source images, one
forward pass, no labels, no gradients, no training. Roughly:

    Z_s    = stack([Enc(x_i) for x_i in N clean source TRAIN images])   # [N, D]
    Z_c    = Z_s - Z_s.mean(0)
    _,S,Vh = svd(Z_c, full_matrices=False)
    V_k    = Vh[:k].T                                                  # [D, k], orthonormal

The cached payload holds the ORTHONORMAL V_k plus the singular values, so ELATTA.BASIS_SCALING
("orthonormal" vs "whitened") is applied at load time and switching it never needs a rebuild. The
mean mu_s is stored for diagnostics only: the paper uses it solely for the centering inside the SVD
and never online (step 4).

This entrypoint exists for the repo's convention (cf. offline_train_aura.py,
offline_train_tinytta.py, warmup_train_ecotta.py) and to make the source-data touch explicit in a
run log. methods/elatta.py also auto-builds the same artifact on a cache miss
(ELATTA.AUTO_BUILD_BASIS, default True), through the SAME build_basis() call -- there is one code
path, not two.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python offline_build_elatta_basis.py --cfg cfgs/cifar10_c/elatta.yaml
    # a different k / N / seed, written to its own cache file:
    ... --cfg cfgs/cifar10_c/elatta.yaml ELATTA.K 32 ELATTA.N_SOURCE_SAMPLES 200 RNG_SEED 2
    # overwrite an existing cache:
    ... --cfg cfgs/cifar10_c/elatta.yaml --force
"""

import logging
import os
import sys

import torch

from conf import cfg, load_cfg_from_args, get_num_classes
from methods.elatta import build_basis, default_basis_path
from models.model import get_model, split_up_model

logger = logging.getLogger(__name__)


def main():
    # `--force` is consumed here so it never reaches yacs' opts list.
    force = "--force" in sys.argv
    if force:
        sys.argv.remove("--force")
    load_cfg_from_args("Build and cache the ELaTTA latent subspace basis V_k.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)
    base_model, model_preprocess = get_model(cfg, num_classes, device)
    base_model.model_preprocess = model_preprocess
    base_model.eval()
    base_model.requires_grad_(False)

    # The SAME cut the online method uses (methods/elatta.py D1): Enc = backbone minus the final
    # linear layer, Cls = that linear layer. Going through split_up_model rather than reimplementing
    # the cut is what guarantees the cached D matches the classifier's in_features at load time.
    encoder, classifier = split_up_model(base_model, cfg.MODEL.ARCH, cfg.CORRUPTION.DATASET)
    encoder.eval()
    encoder.requires_grad_(False)

    path = cfg.ELATTA.BASIS_CKPT or default_basis_path(cfg)
    if os.path.isfile(path) and not force:
        logger.info(f"ELaTTA basis already exists at {path} -- nothing to do (pass --force to "
                    f"rebuild).")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        logger.info(f"  D={payload['D']} k={payload['k']} N={payload['N']} "
                    f"split={payload.get('source_split')} order={payload.get('sample_order')} "
                    f"explained_var={payload.get('explained_variance_ratio')}")
        return

    payload = build_basis(cfg, encoder, device, preprocess=model_preprocess)
    if payload["D"] != classifier.in_features:
        raise RuntimeError(f"Encoder produced D={payload['D']} but the classifier expects "
                           f"{classifier.in_features} input features -- the cut is wrong.")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
    size_mb = os.path.getsize(path) / 1e6
    model_mb = sum(p.numel() * p.element_size() for p in base_model.parameters()) / 1e6
    logger.info(
        f"Wrote {path} ({size_mb:.3f} MB). V_k alone is "
        f"{payload['D'] * payload['k'] * 4 / 1e6:.3f} MB vs {model_mb:.2f} MB of model weights "
        f"= {payload['D'] * payload['k'] * 4 / 1e6 / model_mb * 100:.4f}% "
        f"(the paper quotes 0.05 MB / 330.28 MB = 0.015%)."
    )
    logger.info(f"sigma0 auto-derivation: orthonormal -> {payload['sigma0_auto_orthonormal']:.6g}, "
                f"whitened -> 1.0")


if __name__ == "__main__":
    main()
