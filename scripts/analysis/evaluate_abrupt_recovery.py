import os
import sys
import time
import json
import csv
import logging
import argparse
import numpy as np
import torch

# Ensure repository root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from conf import cfg, load_cfg_from_args, get_num_classes
from models.model import get_model
from utils.registry import ADAPTATION_REGISTRY
import methods
from datasets.corruptions_datasets import create_cifarc_dataset
from utils.stream_manifest import generate_stream_manifest

logger = logging.getLogger(__name__)

DEFAULT_CORRUPTIONS = [
    "pixelate",
    "contrast",
    "zoom_blur",
    "fog",
    "jpeg_compression"
]

def load_corruption_dataset(dataset_name, corruption, severity, data_dir):
    """Loads raw dataset for a specific corruption and severity."""
    if "cifar100" in dataset_name:
        ds_name = "cifar100_c"
    elif "cifar10" in dataset_name:
        ds_name = "cifar10_c"
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")
    
    return create_cifarc_dataset(
        dataset_name=ds_name,
        severity=severity,
        data_dir=data_dir,
        corruption=corruption,
        transform=None
    )

def evaluate_subset(model, dataset, indices, device, batch_size=200):
    """Evaluates classification error on a fixed subset of indices without model adaptation."""
    model.eval()
    num_correct = 0
    total = len(indices)
    
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    with torch.no_grad():
        for batch in loader:
            imgs, labels = batch[0], batch[1]
            imgs_in = imgs.to(device)
            labels_in = labels.to(device)
            if hasattr(model, "forward_eval"):
                logits = model.forward_eval(imgs_in)
            else:
                logits = model.model(imgs_in)
            preds = logits.argmax(dim=1)
            num_correct += (preds == labels_in).sum().item()
            
    acc = num_correct / float(total)
    err = 1.0 - acc
    return err, acc

def run_recovery_evaluation(dataset_name, latent_dim, lifting_type, init_cov, obs_cov,
                            seed, checkpoint_path, output_csv, update_interval=2,
                            max_updates_per_block=20, eval_samples_count=1000,
                            cov_config_id="cov5", is_smoke=False):
    """
    Executes the abrupt-recovery evaluation loop.
    For each corruption block:
      - Evaluates error at u=0 on 1000 held-out samples.
      - Performs updates in chunks of `update_interval` (15 labeled samples/update).
      - Evaluates error at u=2, 4, 6, ... on the SAME 1000 held-out samples.
      - Preserves state continuously across corruption boundaries.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = get_num_classes(dataset_name)

    # 1. Base model
    base_model, model_preprocess = get_model(cfg, num_classes, device)
    base_model.model_preprocess = model_preprocess

    # 2. Configure AURA method
    aura_cls = ADAPTATION_REGISTRY.get("aura")
    model = aura_cls(cfg=cfg, model=base_model, num_classes=num_classes)

    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading offline checkpoint: {checkpoint_path}")
        model._load_and_validate_checkpoint(checkpoint_path)
    else:
        logger.warning(f"No checkpoint provided or path missing: '{checkpoint_path}'. Initializing from scratch.")

    # Derive exact lifting architecture string
    if lifting_type == "linear":
        lifting_arch = "linear_map"
    else:
        h_dim = cfg.AURA.LIFTING.HIDDEN_DIMS[0] if hasattr(cfg.AURA, "LIFTING") and len(cfg.AURA.LIFTING.HIDDEN_DIMS) > 0 else latent_dim
        act = getattr(cfg.AURA.LIFTING, "ACTIVATION", "gelu")
        lifting_arch = f"MLP_h{h_dim}_{act}"

    # 3. Stream manifest
    manifest = generate_stream_manifest(
        dataset_name=dataset_name,
        seed=seed,
        corruptions_seq=DEFAULT_CORRUPTIONS if not is_smoke else DEFAULT_CORRUPTIONS[:1],
        max_updates_per_block=max_updates_per_block,
        adapt_samples_per_update=15,
        eval_samples_per_block=eval_samples_count
    )

    # Output CSV setup
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    file_exists = os.path.exists(output_csv)
    csv_file = open(output_csv, "a", newline="")
    writer = csv.writer(csv_file)

    if not file_exists:
        writer.writerow([
            "dataset", "seed", "latent_dim", "lifting_type", "lifting_arch",
            "offline_checkpoint", "offline_regime", "init_cov", "obs_cov", "cov_config_id",
            "previous_corruption", "current_corruption", "block_index", "severity",
            "global_online_step", "updates_since_shift", "num_labeled_samples_per_update",
            "num_eval_samples", "classification_error", "accuracy",
            "runtime_preceding_2_updates_ms", "evaluation_runtime_ms", "is_shift_boundary"
        ])

    global_step = 0
    prev_corruption = "none"
    corruptions_to_run = DEFAULT_CORRUPTIONS if not is_smoke else DEFAULT_CORRUPTIONS[:1]

    logger.info(f"Starting abrupt recovery stream for dataset={dataset_name}, seed={seed}, latent_dim={latent_dim}, lifting={lifting_type} ({lifting_arch}), cov_id={cov_config_id}")

    for block_idx, corruption in enumerate(corruptions_to_run):
        block_info = manifest["blocks"][corruption]
        eval_indices = block_info["eval_indices"]
        adapt_chunks = block_info["adapt_chunks"]

        # Load raw dataset for this corruption
        dataset = load_corruption_dataset(dataset_name, corruption, severity=5, data_dir=cfg.DATA_DIR)

        # Initial evaluation at u = 0 (before updates on new corruption)
        t_eval_start = time.time()
        err_0, acc_0 = evaluate_subset(model, dataset, eval_indices, device)
        eval_time_ms = (time.time() - t_eval_start) * 1000.0

        global_step += 1
        writer.writerow([
            dataset_name, seed, latent_dim, lifting_type, lifting_arch,
            os.path.basename(checkpoint_path) if checkpoint_path else "none",
            "gradual_mixed_corruption", init_cov, obs_cov, cov_config_id,
            prev_corruption, corruption, block_idx + 1, 5,
            global_step, 0, 15, len(eval_indices),
            round(err_0 * 100.0, 4), round(acc_0 * 100.0, 4),
            0.0, round(eval_time_ms, 2), True
        ])
        csv_file.flush()

        logger.info(f"  [Block {block_idx+1}/{len(corruptions_to_run)}: {corruption}] u=0 | Error: {err_0*100.0:.2f}%")

        # In smoke mode: log initial state tracking (z, P, logits)
        if is_smoke and hasattr(model, "z_list"):
            z_initial_norm = sum(z.norm().item() for z in model.z_list)
            cov_initial_norm = sum(c.norm().item() for c in model.cov_list)
            logger.info(f"  [Smoke Tracker u=0] Latent z_norm: {z_initial_norm:.4f} | Covariance P_norm: {cov_initial_norm:.4f}")

        # Perform updates in intervals of update_interval (e.g. 2 updates)
        chunk_idx = 0
        while chunk_idx < max_updates_per_block and chunk_idx < len(adapt_chunks):
            t_adapt_start = time.time()
            
            # Execute `update_interval` adaptation updates
            for _ in range(update_interval):
                if chunk_idx >= max_updates_per_block or chunk_idx >= len(adapt_chunks):
                    break
                adapt_indices = adapt_chunks[chunk_idx]
                adapt_subset = torch.utils.data.Subset(dataset, adapt_indices)
                adapt_loader = torch.utils.data.DataLoader(adapt_subset, batch_size=len(adapt_indices), shuffle=False)
                
                for batch in adapt_loader:
                    imgs, labels = batch[0], batch[1]
                    imgs_in = imgs.to(device)
                    labels_in = labels.to(device)
                    model.train()
                    _ = model.forward_and_adapt([imgs_in] if not isinstance(imgs_in, list) else imgs_in, labels=labels_in)
                
                chunk_idx += 1

            adapt_time_ms = (time.time() - t_adapt_start) * 1000.0

            # Evaluate at u = chunk_idx (e.g. u = 2, 4, 6, ...)
            t_eval_start = time.time()
            err_u, acc_u = evaluate_subset(model, dataset, eval_indices, device)
            eval_time_ms = (time.time() - t_eval_start) * 1000.0

            global_step += 1
            writer.writerow([
                dataset_name, seed, latent_dim, lifting_type, lifting_arch,
                os.path.basename(checkpoint_path) if checkpoint_path else "none",
                "gradual_mixed_corruption", init_cov, obs_cov, cov_config_id,
                prev_corruption, corruption, block_idx + 1, 5,
                global_step, chunk_idx, 15, len(eval_indices),
                round(err_u * 100.0, 4), round(acc_u * 100.0, 4),
                round(adapt_time_ms, 2), round(eval_time_ms, 2), False
            ])
            csv_file.flush()

            logger.info(f"  [Block {block_idx+1}: {corruption}] u={chunk_idx} | Error: {err_u*100.0:.2f}% | Adapt time: {adapt_time_ms:.1f}ms")

            if is_smoke and hasattr(model, "z_list"):
                z_new_norm = sum(z.norm().item() for z in model.z_list)
                cov_new_norm = sum(c.norm().item() for c in model.cov_list)
                logger.info(f"  [Smoke Tracker u={chunk_idx}] Latent z_norm: {z_new_norm:.4f} | Covariance P_norm: {cov_new_norm:.4f} | State Updated = {z_new_norm != z_initial_norm}")

        prev_corruption = corruption

    csv_file.close()
    logger.info(f"Completed evaluation. Results appended to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Abrupt Shift Recovery Evaluation Runner")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
    parser.add_argument("--dataset", type=str, default="cifar10_c")
    parser.add_argument("--latent_dim", type=int, default=500)
    parser.add_argument("--lifting_type", type=str, default="nonlinear")
    parser.add_argument("--init_cov", type=float, default=0.001)
    parser.add_argument("--obs_cov", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--max_updates", type=int, default=20)
    parser.add_argument("--cov_config_id", type=str, default="cov5")
    parser.add_argument("--smoke", action="store_true", help="Enable smoke test tracking mode")
    
    args, unknown = parser.parse_known_args()
    
    # Load config overrides
    sys.argv = [sys.argv[0], "--cfg", args.cfg] + unknown
    load_cfg_from_args("Abrupt Recovery Runner")

    # Override config nodes
    cfg.defrost()
    cfg.CORRUPTION.DATASET = args.dataset
    cfg.AURA.LATENT_DIM = args.latent_dim
    cfg.AURA.LIFTING.TYPE = args.lifting_type
    cfg.AURA.INIT_COV_SCALE = args.init_cov
    cfg.AURA.OBS_COV_SCALE = args.obs_cov
    cfg.RNG_SEED = args.seed
    
    run_recovery_evaluation(
        dataset_name=args.dataset,
        latent_dim=args.latent_dim,
        lifting_type=args.lifting_type,
        init_cov=args.init_cov,
        obs_cov=args.obs_cov,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        output_csv=args.output_csv,
        update_interval=2,
        max_updates_per_block=args.max_updates,
        eval_samples_count=1000,
        cov_config_id=args.cov_config_id,
        is_smoke=args.smoke
    )
