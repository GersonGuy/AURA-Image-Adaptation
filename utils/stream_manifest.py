import os
import json
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

DEFAULT_CORRUPTIONS = [
    "pixelate",
    "contrast",
    "zoom_blur",
    "fog",
    "jpeg_compression"
]

def generate_stream_manifest(dataset_name: str, seed: int, corruptions_seq=None,
                             max_updates_per_block: int = 50, adapt_samples_per_update: int = 15,
                             eval_samples_per_block: int = 1000, save_dir: str = "data/stream_manifests"):
    """
    Precomputes and saves a deterministic stream manifest for a given dataset and seed.
    Guarantees that evaluation samples (1000 per corruption block) are strictly disjoint
    from all adaptation samples used across updates.
    """
    if corruptions_seq is None:
        corruptions_seq = DEFAULT_CORRUPTIONS

    os.makedirs(save_dir, exist_ok=True)
    manifest_path = os.path.join(save_dir, f"stream_manifest_{dataset_name}_seed{seed}.json")

    # If already computed with matching parameters, load and return
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            if (manifest.get("eval_samples_per_block") == eval_samples_per_block and 
                manifest.get("max_updates_per_block", 0) >= max_updates_per_block and 
                manifest.get("corruptions") == corruptions_seq):
                logger.info(f"Loaded existing stream manifest from {manifest_path}")
                return manifest
        except Exception:
            pass

    rng = np.random.RandomState(seed)
    num_total_images_per_corruption = 10000  # Standard Cifar-C corruption set size
    required_adapt = max_updates_per_block * adapt_samples_per_update

    assert (required_adapt + eval_samples_per_block) <= num_total_images_per_corruption, \
        f"Required adapt ({required_adapt}) + eval ({eval_samples_per_block}) exceeds available images ({num_total_images_per_corruption})"

    manifest = {
        "dataset": dataset_name,
        "seed": seed,
        "corruptions": corruptions_seq,
        "max_updates_per_block": max_updates_per_block,
        "adapt_samples_per_update": adapt_samples_per_update,
        "eval_samples_per_block": eval_samples_per_block,
        "blocks": {}
    }

    for c_idx, corruption in enumerate(corruptions_seq):
        # Permute indices deterministically for this corruption and seed
        c_rng = np.random.RandomState(seed * 1000 + c_idx)
        permuted_indices = c_rng.permutation(num_total_images_per_corruption).tolist()

        # Split into adaptation indices pool and fixed evaluation subset
        adapt_pool = permuted_indices[:required_adapt]
        eval_indices = permuted_indices[required_adapt:required_adapt + eval_samples_per_block]

        # Group adapt_pool into chunks of 15 per update
        adapt_chunks = [
            adapt_pool[u * adapt_samples_per_update : (u + 1) * adapt_samples_per_update]
            for u in range(max_updates_per_block)
        ]

        manifest["blocks"][corruption] = {
            "block_index": c_idx + 1,
            "adapt_chunks": adapt_chunks,
            "eval_indices": eval_indices
        }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Saved stream manifest to {manifest_path}")
    return manifest
