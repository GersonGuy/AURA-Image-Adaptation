#!/usr/bin/env python
"""
Architecture smoke test — no dataset required.

Builds the model + TTA method exactly as test_time.py does, then runs a few
adaptation steps on synthetic data. Use it to validate a new backbone (or a new
machine / GPU) before launching a real run.

Examples
--------
  # ViT-B/16 + AURA (fc + deepest LayerNorm group)
  python scripts/smoke_test_arch.py --cfg cfgs/imagenet_c/aura_vit.yaml

  # regression check: the CNN path must keep working
  python scripts/smoke_test_arch.py --cfg cfgs/cifar10_c/aura.yaml

  # override anything on the CLI, as with test_time.py
  python scripts/smoke_test_arch.py --cfg cfgs/imagenet_c/aura_vit.yaml AURA.LATENT_DIM 32
"""
import argparse
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import methods  # noqa: F401  (populates ADAPTATION_REGISTRY)
from conf import cfg, get_num_classes
from models.model import get_model
from utils.eval_utils import _is_supervised_tta
from utils.registry import ADAPTATION_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("smoke_test")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg", required=True, help="Path to a yaml config (same files test_time.py takes)")
    parser.add_argument("--batches", type=int, default=2, help="Number of synthetic batches to push through")
    parser.add_argument("--batch-size", type=int, default=0, help="Override TEST.BATCH_SIZE (0 = use the config)")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Config overrides, e.g. AURA.LATENT_DIM 32")
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    if args.opts:
        cfg.merge_from_list(args.opts)
    # A smoke test should not pretend to be a timing measurement
    cfg.TIMING.ENABLED = False
    cfg.freeze()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)
    logger.info(f"device={device} arch={cfg.MODEL.ARCH} adaptation={cfg.MODEL.ADAPTATION} "
                f"dataset={cfg.CORRUPTION.DATASET} num_classes={num_classes}")

    base_model, preprocess = get_model(cfg, num_classes, device)
    base_model.model_preprocess = preprocess
    model = ADAPTATION_REGISTRY.get(cfg.MODEL.ADAPTATION)(cfg=cfg, model=base_model, num_classes=num_classes)

    # Report what the method actually latched onto (the part that breaks on a new arch)
    if hasattr(model, "block_names"):
        logger.info(f"AURA blocks: {model.block_names}")
        for name, phi in zip(model.block_names, model.phi_list):
            logger.info(f"  block '{name}': {phi.numel():,} params")
        logger.info(f"graph splicing active: {getattr(model, 'can_slice', False)}")
    if getattr(model, "params", None):
        logger.info(f"trainable tensors: {len(model.params)}")

    img_size = model.img_size if isinstance(model.img_size, (tuple, list)) else (model.img_size, model.img_size)
    channels = 1 if "mnist" in cfg.CORRUPTION.DATASET else 3
    batch_size = args.batch_size or cfg.TEST.BATCH_SIZE

    torch.manual_seed(cfg.RNG_SEED)
    for i in range(args.batches):
        imgs = torch.rand(batch_size, channels, *img_size, device=device)
        labels = torch.randint(0, num_classes, (batch_size,), device=device)
        # Same dispatch as utils/eval_utils.get_accuracy
        if _is_supervised_tta(model):
            outputs = model.forward_and_adapt([imgs], labels=labels)
        else:
            outputs = model([imgs])
        assert outputs.shape == (batch_size, num_classes), \
            f"unexpected output shape {tuple(outputs.shape)}, expected {(batch_size, num_classes)}"
        assert torch.isfinite(outputs).all(), "non-finite logits — the adaptation step diverged"
        logger.info(f"batch {i + 1}/{args.batches}: logits {tuple(outputs.shape)} "
                    f"mean={outputs.mean().item():.4f} std={outputs.std().item():.4f}")

    if device == "cuda":
        logger.info(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    logger.info("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
