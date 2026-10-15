#!/bin/bash
# R1 (trimmed) — 6 methods x 2 datasets x 3 seeds = 36 runs.
# Methods: source, norm_test, norm_alpha, norm_ema, sgd_base, aura
# (roid dropped: ~65x slower per sample; ekf_fc dropped: not in this fork)
# imagenet_c skipped here: data not extracted AND gradual_mixed_corruption
# isn't routed for imagenet_c in datasets/data_loading.py:144.
# aura warm-starts from locally-trained offline ckpts (shipped paths aren't
# on this box).
#
# GOTCHAS (learned the hard way — do not remove the explicit MIXED_GRADUAL
# overrides below):
#  1. Baseline yamls (source, norm_*, tent, ...) ship with SETTING=continual and
#     have no MIXED_GRADUAL block. Forcing SETTING=gradual_mixed_corruption on
#     the CLI then picks up the conf.py defaults (FRAMES=60 → 6,000 samples)
#     instead of the 100k samples aura/sgd_base yamls bake in. We MUST pass
#     MIXED_GRADUAL.* explicitly to every cell to keep sample counts apples-to-
#     apples.
#  2. aura yamls' OFFLINE_CKPT points to Guy's machine paths; we override
#     per-dataset to the local warm ckpts below.
set -u

SCRIPT_DIR_GUARD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_GUARD="$(cd "$SCRIPT_DIR_GUARD/../.." && pwd)"
cd "$REPO_ROOT_GUARD"

if [[ -f "$REPO_ROOT_GUARD/.venv/bin/activate" ]]; then
  source "$REPO_ROOT_GUARD/.venv/bin/activate"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DRY_RUN="${DRY_RUN:-0}"
seeds=(${seeds:-1 2 3 4 5})
methods=(source norm_test norm_alpha norm_ema sgd_base aura)

# Offline-AURA warm-start ckpts. Default: pick the most recent
# offline_train_aura output for each dataset. Run scripts/prepare_offline.sh
# (DATASET=cifar10 / cifar100) once to produce them. Override with
# AURA_CKPT_C10 / AURA_CKPT_C100 env vars to pin specific checkpoints.
# Set AURA_WARM_START=0 to skip the warm-start path and run AURA cold-start.
AURA_WARM_START="${AURA_WARM_START:-1}"
if [[ "$AURA_WARM_START" = "1" ]]; then
  aura_fc_ckpt_c10="${AURA_CKPT_C10:-$(ls -dt ckpt/offline_aura_fc/cifar10_c__Standard__*d__*__sev5__* 2>/dev/null | head -1)/best_params.pth}"
  aura_fc_ckpt_c100="${AURA_CKPT_C100:-$(ls -dt ckpt/offline_aura_fc/cifar100_c__*__*d__*__sev5__* 2>/dev/null | head -1)/best_params.pth}"
  for p in "$aura_fc_ckpt_c10" "$aura_fc_ckpt_c100"; do
    if [[ ! -f "$p" ]]; then
      echo "[r1] WARN: missing offline ckpt ($p); falling back to AURA cold-start." >&2
      AURA_WARM_START=0
      break
    fi
  done
fi

# Shared gradual_mixed_corruption protocol — must be passed on EVERY cell to
# defeat baselines' missing MIXED_GRADUAL blocks (see gotcha #1 above).
CORR_LIST="['gaussian_noise','shot_noise','impulse_noise','defocus_blur','glass_blur','motion_blur']"
gm_overrides=(
  SETTING gradual_mixed_corruption
  MIXED_GRADUAL.FRAMES 1000
  MIXED_GRADUAL.BATCH_SIZE 100
  MIXED_GRADUAL.SEVERITY 5
  MIXED_GRADUAL.SCHEDULE_TYPE random
  MIXED_GRADUAL.CORRUPTION_TYPES "$CORR_LIST"
  TIMING.ENABLED True
)

run() {
  local cfg="$1"; shift
  [[ -f "$cfg" ]] || { echo "[r1] MISSING CFG: $cfg" >&2; return 0; }
  echo "+ python test_time.py --cfg $cfg $*"
  [[ "$DRY_RUN" = "1" ]] || python test_time.py --cfg "$cfg" "$@"
}

for seed in "${seeds[@]}"; do
  for m in "${methods[@]}"; do
    extra=()
    [[ "$m" = "aura" && "$AURA_WARM_START" = "1" ]] && extra=(AURA.OFFLINE_CKPT "$aura_fc_ckpt_c10")
    run "cfgs/cifar10_c/${m}.yaml" RNG_SEED "$seed" "${gm_overrides[@]}" "${extra[@]}"
  done
  for m in "${methods[@]}"; do
    extra=()
    [[ "$m" = "aura" && "$AURA_WARM_START" = "1" ]] && extra=(AURA.OFFLINE_CKPT "$aura_fc_ckpt_c100")
    run "cfgs/cifar100_c/${m}.yaml" RNG_SEED "$seed" "${gm_overrides[@]}" "${extra[@]}"
  done
done
