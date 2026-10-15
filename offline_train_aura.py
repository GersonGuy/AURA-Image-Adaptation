import os
import json
import datetime
import logging
import numpy as np
import torch
import torch.nn.functional as F
import methods  

from conf import cfg, load_cfg_from_args, get_num_classes
from datasets.data_loading import get_source_loader
from models.model import get_model
from utils.registry import ADAPTATION_REGISTRY
from collections import deque

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# IndexedDataset wrapper
# Augments any map-style dataset to also return the sample index.
# Allows exact per-episode index logging for dataset reproducibility.
# -----------------------------------------------------------------------

class IndexedDataset(torch.utils.data.Dataset):
    """Wraps a dataset to additionally return the integer sample index."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        out = self.dataset[idx]
        if len(out) == 4:
            img, label, domain, alphas = out
            return img, label, idx, alphas
        else:
            img, label = out
            return img, label, idx, None

    def __len__(self):
        return len(self.dataset)


# -----------------------------------------------------------------------
# Save directory name construction
# Format: {dataset}__{arch}__{latent_dim}d__{corruptions}__{severity}__{ts}
# -----------------------------------------------------------------------

def _build_save_dir(cfg, timestamp: str) -> str:
    if hasattr(cfg, "SAVE_DIR") and cfg.SAVE_DIR and not cfg.SAVE_DIR.startswith("./output"):
        return cfg.SAVE_DIR

    num_corruptions = len(cfg.CORRUPTION.TYPE)
    
    if num_corruptions == 15:
        corruption_tag = "all_corruptions"
    else:
        joined_corruptions = "+".join(sorted(cfg.CORRUPTION.TYPE))
        if len(joined_corruptions) > 60:
            corruption_tag = "big_dataset"
        else:
            corruption_tag = joined_corruptions

    severity_tag   = "sev" + "_".join(str(s) for s in sorted(cfg.CORRUPTION.SEVERITY))
    # Pull latent dim from whichever section the active adaptation uses.
    latent_dim = cfg.AURA.LATENT_DIM if cfg.MODEL.ADAPTATION == "aura" else cfg.AURA_FC.LATENT_DIM
    dir_name = (
        f"{cfg.CORRUPTION.DATASET}"
        f"__{cfg.MODEL.ARCH}"
        f"__{latent_dim}d"
        f"__{corruption_tag}"
        f"__{severity_tag}"
        f"__{timestamp}"
    )
    return os.path.join(cfg.CKPT_DIR, "offline_aura_fc", dir_name)


# -----------------------------------------------------------------------
# Checkpoint dict builder
# -----------------------------------------------------------------------

def _build_checkpoint(aura_fc_module, cfg, num_classes: int) -> dict:
    """
    Constructs the full checkpoint dictionary.
    Contains: learned parameters + structural metadata + scenario definition.
    """
    return {
        # ---- Learned parameters ----
        "A":              aura_fc_module.A.data.cpu(),
        "phi":            aura_fc_module.phi.data.cpu(),   # fixed, but saved for consistency

        # ---- Structural metadata ----
        "latent_dim":     cfg.AURA_FC.LATENT_DIM,
        "num_fc_params":  aura_fc_module.num_fc_params,
        "num_classes":    num_classes,
        "in_features":    aura_fc_module.in_features,

        # ---- Kalman hyperparameters ----
        "gamma":          cfg.AURA_FC.GAMMA,
        "init_cov_scale": cfg.AURA_FC.INIT_COV_SCALE,
        "obs_cov_scale":  cfg.AURA_FC.OBS_COV_SCALE,

        # ---- Scenario definition (must match online config at load time) ----
        "dataset":          cfg.CORRUPTION.DATASET,
        "model_arch":       cfg.MODEL.ARCH,
        "corruption_types": list(cfg.CORRUPTION.TYPE),
        "severity":         list(cfg.CORRUPTION.SEVERITY),

        # ---- Full config snapshot (human-readable serialized string) ----
        "cfg_dump": str(cfg),
    }

# -----------------------------------------------------------------------
# Checkpoint builder for AURA
# -----------------------------------------------------------------------

def _build_aura_checkpoint(aura_fc_module, cfg, num_classes: int) -> dict:
    blocks_meta = {}
    for i, b_name in enumerate(aura_fc_module.block_names):
        block_data = {
            "A": aura_fc_module.A_list[i].data.cpu(),
            "phi": aura_fc_module.phi_list[i].data.cpu(),
        }
        # Save nonlinear correction state_dict if present
        if getattr(aura_fc_module, '_use_nonlinear', False) and len(aura_fc_module.correction_list) > 0:
            block_data["correction"] = {
                k: v.cpu() for k, v in aura_fc_module.correction_list[i].state_dict().items()
            }
        blocks_meta[b_name] = block_data

    return {
        "blocks":            blocks_meta,
        "block_names":       aura_fc_module.block_names,
        "latent_dim":        cfg.AURA.LATENT_DIM,
        "lifting_type":      cfg.AURA.LIFTING.TYPE.lower() if hasattr(cfg.AURA, "LIFTING") else getattr(cfg.AURA, "LIFTING_TYPE", "linear").lower(),
        "lifting_h":         int(cfg.AURA.LIFTING.HIDDEN_DIMS[0]) if (hasattr(cfg.AURA, "LIFTING") and len(cfg.AURA.LIFTING.HIDDEN_DIMS) > 0) else int(getattr(cfg.AURA, "LIFTING_H", 0)),
        "num_classes":       num_classes,
        "gamma":             cfg.AURA.GAMMA,
        "dataset":           cfg.CORRUPTION.DATASET,
        "model_arch":        cfg.MODEL.ARCH,
        "corruption_types":  list(cfg.CORRUPTION.TYPE),
        "severity":          list(cfg.CORRUPTION.SEVERITY),
        "cfg_dump":          str(cfg),
    }


# -----------------------------------------------------------------------
# Dataset reconstruction metadata builder
# -----------------------------------------------------------------------

def _build_dataset_meta(cfg, base_dataset_name: str) -> dict:
    return {
        "dataset":        base_dataset_name,
        "train_split":    True,
        "data_root_dir":  cfg.DATA_DIR,
        "batch_size":     1 + cfg.OFFLINE.Q_SAMPLES,
        "rng_seed":       cfg.RNG_SEED,
        "num_episodes":   cfg.OFFLINE.NUM_EPISODES,
        "q_samples":      cfg.OFFLINE.Q_SAMPLES,
        "note": (
            "episode_indices.npy contains shape [num_episodes, 1+q_samples] "
            "with the exact per-episode sample indices from the source dataset. "
            "Re-run with the same rng_seed and the same index log to reproduce "
            "the exact offline training sequence."
        )
    }


# -----------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------

def evaluate_offline():
    load_cfg_from_args("Offline AURA_FC Training")
    logger.info(f"Starting Offline Training over {cfg.OFFLINE.NUM_EPISODES} episodes.")
    logger.info(f"  A-only training: phi is FIXED at source FC parameters.")
    logger.info(f"  z/cov evolve continuously (TBPTT detach between episodes).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Clean source dataset (strip "_c" suffix: "cifar10_c" → "cifar10")
    base_dataset_name = cfg.CORRUPTION.DATASET.replace("_c", "")

    # ---- Data loader with index tracking ----
    if cfg.SETTING == "gradual_mixed_corruption":
        from datasets.corruptions_datasets import (
            create_gradual_mixed_cifarc_dataset,
            create_gradual_mixed_imagenetc_dataset,
            create_gradual_mixed_mnistc_dataset,
        )

        #batch size for multiple updates
        super_batch_size = (1 + cfg.OFFLINE.Q_SAMPLES) * cfg.OFFLINE.NUM_UPDATES_PER_EPISODE

        # Domain-generalization hook: if OFFLINE.CORRUPTION_POOL is non-empty, use it as the
        # offline (M) corruption pool — otherwise fall back to MIXED_GRADUAL.CORRUPTION_TYPES
        # (legacy behavior where offline and online share the same pool).
        offline_corruption_pool = (
            list(cfg.OFFLINE.CORRUPTION_POOL)
            if len(cfg.OFFLINE.CORRUPTION_POOL) > 0
            else list(cfg.MIXED_GRADUAL.CORRUPTION_TYPES)
        )
        if len(cfg.OFFLINE.CORRUPTION_POOL) > 0:
            logger.info(
                f"Offline domain-gen mode: training pool (M) = {offline_corruption_pool} "
                f"(overrides MIXED_GRADUAL.CORRUPTION_TYPES = {list(cfg.MIXED_GRADUAL.CORRUPTION_TYPES)})"
            )
        else:
            logger.info(
                f"Offline pool (M) inherited from MIXED_GRADUAL.CORRUPTION_TYPES = {offline_corruption_pool}"
            )

        if cfg.CORRUPTION.DATASET == "mnist_c":
            import torchvision.transforms as T
            mnist_transform = T.Compose([T.ToTensor()])
            mnist_data_dir = os.path.join(cfg.DATA_DIR, "mnist_c")
            raw_dataset = create_gradual_mixed_mnistc_dataset(
                data_dir=mnist_data_dir,
                corruptions_seq=offline_corruption_pool,
                transform=mnist_transform,
                batch_size=super_batch_size,
                total_frames=cfg.OFFLINE.NUM_EPISODES,
                rng_seed=cfg.RNG_SEED,
                schedule_type=cfg.MIXED_GRADUAL.SCHEDULE_TYPE,
                random_max_step=getattr(cfg.MIXED_GRADUAL, "RANDOM_MAX_STEP", 0.05),
                alpha_seed=getattr(cfg.MIXED_GRADUAL, "ALPHA_SEED", -1),
                waypoint_speed=getattr(cfg.MIXED_GRADUAL, "WAYPOINT_SPEED", 0.03),
                waypoint_arrival_tol=getattr(cfg.MIXED_GRADUAL, "WAYPOINT_ARRIVAL_TOL", 0.02),
                waypoint_concentration=getattr(cfg.MIXED_GRADUAL, "WAYPOINT_CONCENTRATION", 0.5),
                waypoint_noise_scale=getattr(cfg.MIXED_GRADUAL, "WAYPOINT_NOISE_SCALE", 0.002),
            )
            should_shuffle = False
        elif cfg.CORRUPTION.DATASET == "imagenet_c":
            # Standard ImageNet preprocessing. ImageFolder returns PIL → need to tensorize + normalize.
            import torchvision.transforms as T
            imnet_transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            imagenet_data_dir = os.path.join(cfg.DATA_DIR, "ImageNet-C")
            raw_dataset = create_gradual_mixed_imagenetc_dataset(
                data_dir=imagenet_data_dir,
                corruptions_seq=offline_corruption_pool,
                severity=cfg.MIXED_GRADUAL.SEVERITY,
                transform=imnet_transform,
                batch_size=super_batch_size,
                total_frames=cfg.OFFLINE.NUM_EPISODES,
                rng_seed=cfg.RNG_SEED,
                schedule_type=cfg.MIXED_GRADUAL.SCHEDULE_TYPE,
                random_max_step=getattr(cfg.MIXED_GRADUAL, "RANDOM_MAX_STEP", 0.05),
                alpha_seed=getattr(cfg.MIXED_GRADUAL, "ALPHA_SEED", -1),
                waypoint_speed=getattr(cfg.MIXED_GRADUAL, "WAYPOINT_SPEED", 0.03),
                waypoint_arrival_tol=getattr(cfg.MIXED_GRADUAL, "WAYPOINT_ARRIVAL_TOL", 0.02),
                waypoint_concentration=getattr(cfg.MIXED_GRADUAL, "WAYPOINT_CONCENTRATION", 0.5),
                waypoint_noise_scale=getattr(cfg.MIXED_GRADUAL, "WAYPOINT_NOISE_SCALE", 0.002),
            )
            should_shuffle = False
        else:
            raw_dataset = create_gradual_mixed_cifarc_dataset(
                dataset_name=base_dataset_name + "_c",
                severity=cfg.MIXED_GRADUAL.SEVERITY,
                data_dir=cfg.DATA_DIR,
                corruptions_seq=offline_corruption_pool,
                transform=None,
                batch_size=super_batch_size,
                total_frames=cfg.OFFLINE.NUM_EPISODES,
                rng_seed=cfg.RNG_SEED,
                schedule_type=cfg.MIXED_GRADUAL.SCHEDULE_TYPE,
                random_max_step=getattr(cfg.MIXED_GRADUAL, "RANDOM_MAX_STEP", 0.05),
                alpha_seed=getattr(cfg.MIXED_GRADUAL, "ALPHA_SEED", -1),
                waypoint_speed=getattr(cfg.MIXED_GRADUAL, "WAYPOINT_SPEED", 0.03),
                waypoint_arrival_tol=getattr(cfg.MIXED_GRADUAL, "WAYPOINT_ARRIVAL_TOL", 0.02),
                waypoint_concentration=getattr(cfg.MIXED_GRADUAL, "WAYPOINT_CONCENTRATION", 0.5),
                waypoint_noise_scale=getattr(cfg.MIXED_GRADUAL, "WAYPOINT_NOISE_SCALE", 0.002)
            )
            should_shuffle = False
    else:
        raw_dataset, _ = get_source_loader(
            dataset_name=base_dataset_name,
            adaptation=cfg.MODEL.ADAPTATION,
            preprocess=None,
            data_root_dir=cfg.DATA_DIR,
            batch_size=1 + cfg.OFFLINE.Q_SAMPLES,
            use_clip=cfg.MODEL.USE_CLIP,
            train_split=True,
            workers=0    # single-threaded for determinism
        )
        should_shuffle = True

    indexed_dataset = IndexedDataset(raw_dataset)
    train_loader = torch.utils.data.DataLoader(
        indexed_dataset,
        batch_size=1 + cfg.OFFLINE.Q_SAMPLES,
        shuffle=should_shuffle,
        num_workers=0,
        drop_last=False
    )

    num_classes = get_num_classes(cfg.CORRUPTION.DATASET)

    # ---- Base model (encoder frozen inside AURA_FC.__init__) ----
    base_model, _ = get_model(cfg, num_classes=num_classes, device=device)
    base_model = base_model.to(device)

    # ---- Wrap inside Adaptation Method ----
    aura_fc_module = ADAPTATION_REGISTRY.get(cfg.MODEL.ADAPTATION.lower())(cfg=cfg, model=base_model, num_classes=num_classes)
    aura_fc_module.to(device)

    if cfg.MODEL.ADAPTATION == "aura_fc":
        aura_fc_module.A.requires_grad_(True)
        aura_fc_module.phi.requires_grad_(False)
        params_to_train = [aura_fc_module.A]
        z = torch.zeros(aura_fc_module.latent_dim, device=device)
        cov = torch.eye(aura_fc_module.latent_dim, device=device) * cfg.AURA_FC.INIT_COV_SCALE
        offline_lr = cfg.OFFLINE.LR
    else: # aura
        for A in aura_fc_module.A_list:
            A.requires_grad_(True)
        for phi in aura_fc_module.phi_list:
            phi.requires_grad_(False)
        params_to_train = list(aura_fc_module.A_list)
        # If nonlinear lifting is active, also train the correction networks
        if getattr(aura_fc_module, '_use_nonlinear', False) and len(aura_fc_module.correction_list) > 0:
            for corr in aura_fc_module.correction_list:
                corr.requires_grad_(True)
                corr.train()
            params_to_train = params_to_train + list(aura_fc_module.correction_list.parameters())
            logger.info(
                f"Offline: training A_list + NonlinearCorrections, "
                f"total correction params={sum(p.numel() for p in aura_fc_module.correction_list.parameters())}"
            )
        z = [z_t.clone() for z_t in aura_fc_module.z_list]
        cov = [c_t.clone() for c_t in aura_fc_module.cov_list]
        offline_lr = cfg.OFFLINE.LR

    opt_name = getattr(cfg.OFFLINE, "OPTIMIZER", "Adam")
    if opt_name == "SGD":
        optimizer = torch.optim.SGD(params_to_train, lr=offline_lr,
                                    momentum=getattr(cfg.OFFLINE, "MOMENTUM", 0.9))
        logger.info(f"Offline optimizer: SGD(lr={offline_lr}, momentum={getattr(cfg.OFFLINE, 'MOMENTUM', 0.9)})")
    else:
        optimizer = torch.optim.Adam(params_to_train, lr=offline_lr)
        logger.info(f"Offline optimizer: Adam(lr={offline_lr})")

    # ---- Directories and metadata ----
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = _build_save_dir(cfg, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Saving artifacts to: {save_dir}")

    # Config text dump (human-readable, at start)
    with open(os.path.join(save_dir, "offline_config.txt"), "w") as f:
        f.write(str(cfg))

    # Dataset reconstruction metadata JSON
    dataset_meta = _build_dataset_meta(cfg, base_dataset_name)
    with open(os.path.join(save_dir, "dataset_meta.json"), "w") as f:
        json.dump(dataset_meta, f, indent=2)

    # ---- Tracking ----
    running_loss = deque(maxlen=cfg.OFFLINE.PRINT_FREQ)
    running_acc  = deque(maxlen=cfg.OFFLINE.PRINT_FREQ)
    best_loss = float('inf')
    early_stop_patience = int(getattr(cfg.OFFLINE, "EARLY_STOP_PATIENCE", 0))
    no_improve_windows = 0
    if early_stop_patience > 0:
        logger.info(f"Offline: EARLY_STOP_PATIENCE={early_stop_patience} PRINT_FREQ windows without improvement.")

    for name, p in aura_fc_module.named_parameters():
        if p.requires_grad:
            print(name, p.shape)

    # Episode index log: shape will be [num_episodes, 1+Q_SAMPLES]
    episode_index_log = []

    aura_fc_module.train()
    iterator = iter(train_loader)

    # Fix 1: optionally reset z/cov at each episode boundary to prevent cross-episode drift.
    # Fix 2: optionally clip ||z|| per block during the inner loop to bound intra-episode drift.
    reset_per_ep = bool(getattr(cfg.OFFLINE, "RESET_STATE_PER_EPISODE", False))
    z_clip = float(getattr(cfg.OFFLINE, "Z_NORM_CLIP", 0.0))
    if reset_per_ep:
        logger.info("Offline: RESET_STATE_PER_EPISODE=True (z/cov reset at each episode start).")
    if z_clip > 0:
        logger.info(f"Offline: Z_NORM_CLIP={z_clip} (clip ||z|| per block within each update).")

    for episode in range(1, cfg.OFFLINE.NUM_EPISODES + 1):

        optimizer.zero_grad()
        episode_loss = 0.0
        episode_acc = 0.0

        if reset_per_ep:
            # Fix 1: fresh state at episode start — detached clones of the module's init state.
            if cfg.MODEL.ADAPTATION == "aura_fc":
                z = torch.zeros(aura_fc_module.latent_dim, device=device)
                cov = torch.eye(aura_fc_module.latent_dim, device=device) * cfg.AURA_FC.INIT_COV_SCALE
            else:
                z = [z_t.detach().clone() for z_t in aura_fc_module.z_init_list]
                cov = [c_t.detach().clone() for c_t in aura_fc_module.cov_init_list]

        temp_cov = cov
        temp_z = z
        if cfg.MODEL.ADAPTATION == "aura_fc":
            temp_A = aura_fc_module.A.clone()
            tmp_phi = aura_fc_module.phi.clone()

        #  Inner Loop: Process novel sequential batches sharing the identical mathematically frozen Alpha 
        for _ in range(cfg.OFFLINE.NUM_UPDATES_PER_EPISODE):
            try:
                imgs, labels, indices, alphas = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                imgs, labels, indices = next(iterator)

            # Guard against short last batch
            if imgs.shape[0] < (1 + cfg.OFFLINE.Q_SAMPLES):
                iterator = iter(train_loader)
                imgs, labels, indices, alphas = next(iterator)

            imgs   = imgs.to(device)
            labels = labels.to(device)

            # Log sequentially distinct sample indices
            episode_index_log.append(indices.tolist())

            # Split context (1 sample) and query (Q samples)
            ctx_imgs,   ctx_labels = imgs[0:1],  labels[0:1]
            qry_imgs,   qry_labels = imgs[1:],   labels[1:]

            tot_imgs = torch.cat([ctx_imgs, qry_imgs], dim=0)
            tot_labels = torch.cat([ctx_labels, qry_labels], dim=0)

            # ---- Forward single update (stateful: carry z, cov) ----
            logits_q, loss_q, z_new, cov_new = aura_fc_module.forward_offline_step(
                ctx_imgs, ctx_labels, qry_imgs, qry_labels, z, cov
            )
            
            # Carry state forward WITHOUT detach() crossing internal boundaries
            z, cov = z_new, cov_new
            # Fix 2: clip ||z|| per block (aura) or per vector (aura_fc) — keeps z in a bounded regime
            if z_clip > 0:
                if cfg.MODEL.ADAPTATION == "aura_fc":
                    n = z.norm()
                    if n > z_clip:
                        z = z * (z_clip / n)
                else:
                    clipped = []
                    for z_t in z:
                        n = z_t.norm()
                        clipped.append(z_t if n <= z_clip else z_t * (z_clip / n))
                    z = clipped
            episode_loss = episode_loss + loss_q
            episode_acc = episode_acc + (logits_q.argmax(1) == qry_labels).float().mean().item()
        
        
        # Average loss/acc mathematically over updates spanning the batch continuum
        avg_loss = episode_loss / cfg.OFFLINE.NUM_UPDATES_PER_EPISODE
        avg_acc = episode_acc / cfg.OFFLINE.NUM_UPDATES_PER_EPISODE

        # Accumulate gradients across the unrolled episodic loop length
        avg_loss.backward()
        optimizer.step()

        # ---- TBPTT detach: slice analytical graph but retain latent position forward ----
        # Distribution "ticks forward" here after processing inner loops
        if cfg.MODEL.ADAPTATION == "aura_fc":
            z = z.detach()
            cov = cov.detach()
            z_norm_val = z.norm().item()
        else:
            z = [z_t.detach() for z_t in z]
            cov = [c_t.detach() for c_t in cov]
            z_norm_val = sum([z_t.norm().item() for z_t in z]) / len(z)

        running_loss.append(avg_loss.item())
        running_acc.append(avg_acc)
        if avg_acc < 0.87 and cfg.MODEL.ADAPTATION == "aura_fc":
            b = 1
            encod = aura_fc_module.encoder(tot_imgs)
            output = aura_fc_module._obs_fn(z,encod,temp_A,tmp_phi,num_classes,encod.shape[1])
            acc_check = (output.argmax(1) == tot_labels).float().mean().item()
            b = 1
            

        if episode % cfg.OFFLINE.PRINT_FREQ == 0:
            avg_l = sum(running_loss) / len(running_loss)
            avg_a = sum(running_acc)  / len(running_acc)
            logger.info(
                f"Episode {episode:05d}/{cfg.OFFLINE.NUM_EPISODES} | "
                f"z_norm={z_norm_val:.4f} | "
                f"Avg Query Loss={avg_l:.4f} | Avg Query Acc={avg_a:.2%}"
            )

            # Best-loss checkpoint
            improved = avg_l < best_loss
            if improved or cfg.OFFLINE.NUM_EPISODES//2 == episode or cfg.OFFLINE.NUM_EPISODES*(3/4) == episode:
                name = "best_params.pth" if improved else "best_params_2.pth" if cfg.OFFLINE.NUM_EPISODES//2 == episode else "best_params_3.pth"
                best_loss = avg_l if improved else best_loss
                if cfg.MODEL.ADAPTATION == "aura_fc":
                    ckpt = _build_checkpoint(aura_fc_module, cfg, num_classes)
                else:
                    ckpt = _build_aura_checkpoint(aura_fc_module, cfg, num_classes)
                torch.save(ckpt, os.path.join(save_dir, name))
                logger.info(f"  -> New best loss {best_loss:.4f}: saved {name}")

            # Periodic snapshot every 100 episodes (for training-length ablations)
            if episode % 100 == 0:
                if cfg.MODEL.ADAPTATION == "aura_fc":
                    snap = _build_checkpoint(aura_fc_module, cfg, num_classes)
                else:
                    snap = _build_aura_checkpoint(aura_fc_module, cfg, num_classes)
                snap_name = f"params_ep{episode:04d}.pth"
                torch.save(snap, os.path.join(save_dir, snap_name))
                logger.info(f"  -> Periodic snapshot saved: {snap_name}")

            # Early stopping
            if early_stop_patience > 0:
                if improved:
                    no_improve_windows = 0
                else:
                    no_improve_windows += 1
                    if no_improve_windows >= early_stop_patience:
                        logger.info(
                            f"Offline: EARLY STOP at episode {episode} "
                            f"(no improvement for {no_improve_windows} windows of {cfg.OFFLINE.PRINT_FREQ} episodes). "
                            f"best_loss={best_loss:.4f}"
                        )
                        break
        
        
        #visualize image
        """if episode % 250 == 0 or episode == 1:
            import matplotlib.pyplot as plt
            img = qry_imgs[0].detach().cpu().permute(1, 2, 0)  # [H,W,C]
            plt.imshow(img)
            plt.title(f"true={qry_labels[0].item()}, pred={logits_q[0].argmax().item()}")
            plt.axis("off")
            plt.show()"""

    # ---- Final checkpoint ----
    if cfg.MODEL.ADAPTATION == "aura_fc":
        ckpt = _build_checkpoint(aura_fc_module, cfg, num_classes)
    else:
        ckpt = _build_aura_checkpoint(aura_fc_module, cfg, num_classes)
    torch.save(ckpt, os.path.join(save_dir, "last_params.pth"))

    # ---- Episode index log (numpy int32, shape [num_episodes, 1+Q_SAMPLES]) ----
    idx_array = np.array(episode_index_log, dtype=np.int32)
    np.save(os.path.join(save_dir, "episode_indices.npy"), idx_array)
    logger.info(f"Saved episode_indices.npy: shape {idx_array.shape}")
    
    # ---- Data Generation Metadata ----
    if cfg.SETTING == "gradual_mixed_corruption":
        with open(os.path.join(save_dir, "data_generation_metadata.json"), "w") as f:
            json.dump(raw_dataset.metadata, f, indent=4)
        np.save(os.path.join(save_dir, "alpha_schedule.npy"), raw_dataset.alpha_schedule)
        logger.info("Saved data_generation_metadata.json and alpha_schedule.npy")

    logger.info(f"Offline Training complete. Artifacts saved to: {save_dir}")
    logger.info(
        f"To load for online use, set in aura_fc.yaml:\n"
        f"  AURA_FC:\n"
        f"    OFFLINE_CKPT: {os.path.join(save_dir, 'best_params.pth')}"
    )


if __name__ == '__main__':
    evaluate_offline()
