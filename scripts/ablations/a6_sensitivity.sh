#!/bin/bash -l
# =============================================================================
# Ablation A6 — OBS_COV_SCALE x GAMMA sensitivity heatmap
#   Grid  : OBS_COV_SCALE ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1}  (5 values)
#           GAMMA         ∈ {0.99, 0.999, 0.9999, 1.0}   (4 values)
#   Cells : 5 x 4 = 20
#   Seeds : 3 per cell
#   TOTAL : 60 RUNS.  At ~1–2 min/run on CIFAR-10-C this is ~1–2 GPU-hours;
#           but TTA loops can be much longer on slower boxes or with dataloader
#           contention. BUDGET ACCORDINGLY before launching.
#   Dataset: CIFAR-10-C, gradual_mixed_corruption, random schedule
#   Mode  : COLD-START (OFFLINE_CKPT="") — the shipped ckpt was trained with
#           GAMMA=0.999 and the prior it encodes is tied to that setting; a
#           cold-start comparison isolates the online hyperparameter effect.
#   Feeds : 2D sensitivity heatmap (error; optionally ms/sample).
# =============================================================================
#                              !!!  WARNING  !!!
#  60 RUNS.  Use DRY_RUN=1 first to sanity-check the full command list before
#  launching.  Consider a narrower seed set (seeds=(1)) for an initial pass
#  and widen to 3 seeds only on promising sub-regions.
# =============================================================================

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate tta || true
fi

set -u
DRY_RUN="${DRY_RUN:-0}"
seeds=(${seeds:-1 2 3 4 5})

obs_scales=(0.0001 0.001 0.01 0.1 1.0)
gammas=(0.99 0.999 0.9999 1.0)

cfg="cfgs/cifar10_c/aura.yaml"

# ---- Pre-flight ----
if [[ ! -f "$cfg" ]]; then
  echo "[A6] MISSING CFG: $cfg" >&2; exit 1
fi

run() {
  echo "+ $*"
  [[ "$DRY_RUN" = "1" ]] || "$@"
}

total=$(( ${#obs_scales[@]} * ${#gammas[@]} * ${#seeds[@]} ))
echo "[A6] Running $total cells (obs=${#obs_scales[@]} x gamma=${#gammas[@]} x seeds=${#seeds[@]})"

for obs in "${obs_scales[@]}"; do
  for g in "${gammas[@]}"; do
    for seed in "${seeds[@]}"; do
      run python test_time.py --cfg "$cfg" \
        RNG_SEED "$seed" \
        AURA.OBS_COV_SCALE "$obs" \
        AURA.GAMMA "$g" \
        AURA.OFFLINE_CKPT "" \
        TIMING.ENABLED True
    done
  done
done
