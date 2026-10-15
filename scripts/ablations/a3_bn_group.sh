#!/bin/bash -l
# =============================================================================
# Ablation A3 — BN grouping mode
#   Varies: AURA.BN_GROUPING_MODE ∈ {shared_all_bn, per_stage_bn}
#   Fixed : AURA.STAGE_SELECTION = [fc, stage_1]  (matches headline config)
#   Dataset: CIFAR-10-C, gradual_mixed_corruption, random schedule
#   Seeds : 3
#   Mode  : COLD-START — the shipped ckpt was trained in per_stage_bn mode so
#           it is not valid for the shared_all_bn cell; cold-start both for
#           a clean comparison.
#   Feeds : Two-row BN-grouping ablation table.
# =============================================================================

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate tta || true
fi

set -u
DRY_RUN="${DRY_RUN:-0}"
seeds=(${seeds:-1 2 3 4 5})

cfg="cfgs/cifar10_c/aura.yaml"
# A2 per-cell yaml already encodes STAGE_SELECTION=[fc,stage_1] + cold-start.
# Reuse it so we don't introduce a parallel stage-selection string anywhere.
cfg_stage="cfgs/cifar10_c/_ablations/a2_fc_stage1.yaml"

# ---- Pre-flight ----
for p in "$cfg" "$cfg_stage"; do
  if [[ ! -f "$p" ]]; then
    echo "[A3] MISSING CFG: $p" >&2; exit 1
  fi
done

run() {
  echo "+ $*"
  [[ "$DRY_RUN" = "1" ]] || "$@"
}

modes=(shared_all_bn per_stage_bn)

total=$(( ${#modes[@]} * ${#seeds[@]} ))
echo "[A3] Running $total cells (modes=${#modes[@]} x seeds=${#seeds[@]})"

for mode in "${modes[@]}"; do
  for seed in "${seeds[@]}"; do
    run python test_time.py --cfg "$cfg_stage" \
      RNG_SEED "$seed" \
      AURA.BN_GROUPING_MODE "$mode" \
      AURA.OFFLINE_CKPT "" \
      TIMING.ENABLED True
  done
done
