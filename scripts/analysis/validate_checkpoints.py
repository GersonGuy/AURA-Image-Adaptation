import os
import sys
import json
import torch
import logging
import argparse

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)

def find_params_file(ckpt_dir):
    """Finds last_params.pth or best_params.pth in ckpt_dir or any subdirectory."""
    for filename in ["last_params.pth", "best_params.pth"]:
        path = os.path.join(ckpt_dir, filename)
        if os.path.exists(path):
            return path
            
    for root, _, files in os.walk(ckpt_dir):
        for filename in ["last_params.pth", "best_params.pth"]:
            if filename in files:
                return os.path.join(root, filename)
    return None

def validate_checkpoint(ckpt_dir, expected_dataset=None, expected_latent_dim=None,
                        expected_lifting_type=None, expected_regime="gradual_mixed_corruption"):
    """
    Validates a Stage 1 checkpoint against expected topology and metadata.
    Returns (is_valid: bool, status: str, metadata: dict).
    """
    metadata_path = os.path.join(ckpt_dir, "metadata.json")
    params_path = find_params_file(ckpt_dir)

    if not params_path:
        return False, "MISSING_WEIGHTS", {}

    if not os.path.exists(metadata_path):
        return False, "MISSING_METADATA", {}

    try:
        with open(metadata_path, "r") as f:
            meta = json.load(f)
    except Exception as e:
        return False, f"CORRUPT_METADATA ({e})", {}

    # Check status
    if meta.get("status") != "COMPLETED":
        return False, "INCOMPLETE", meta

    # Check regime
    if meta.get("regime") != expected_regime:
        return False, f"MISMATCH_REGIME (expected {expected_regime}, got {meta.get('regime')})", meta

    # Check dataset if specified
    if expected_dataset and meta.get("dataset") != expected_dataset:
        return False, f"MISMATCH_DATASET (expected {expected_dataset}, got {meta.get('dataset')})", meta

    # Check latent dimension if specified
    if expected_latent_dim and meta.get("latent_dim") != expected_latent_dim:
        return False, f"MISMATCH_LATENT_DIM (expected {expected_latent_dim}, got {meta.get('latent_dim')})", meta

    # Check lifting type if specified
    if expected_lifting_type and meta.get("lifting_type") != expected_lifting_type:
        return False, f"MISMATCH_LIFTING_TYPE (expected {expected_lifting_type}, got {meta.get('lifting_type')})", meta

    # Verify PyTorch state dict loadability
    try:
        state = torch.load(params_path, map_location="cpu")
        if "A" not in state and "A_list" not in state and "blocks" not in state:
            return False, "INVALID_STATE_DICT_STRUCTURE", meta
    except Exception as e:
        return False, f"CORRUPT_WEIGHTS ({e})", meta

    return True, "COMPLETED", meta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 Checkpoint Validator")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to experiment checkpoint directory")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--lifting_type", type=str, default=None)
    parser.add_argument("--regime", type=str, default="gradual_mixed_corruption")

    args = parser.parse_args()

    is_valid, status, meta = validate_checkpoint(
        ckpt_dir=args.ckpt_dir,
        expected_dataset=args.dataset,
        expected_latent_dim=args.latent_dim,
        expected_lifting_type=args.lifting_type,
        expected_regime=args.regime
    )

    if is_valid:
        logger.info(f"VALIDATED: {args.ckpt_dir} -> {status} (metadata: {meta})")
        sys.exit(0)
    else:
        logger.error(f"VALIDATION FAILED: {args.ckpt_dir} -> {status}")
        sys.exit(1)
