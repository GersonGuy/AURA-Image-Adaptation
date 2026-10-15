#!/bin/bash -l
# =============================================================================
# Experiment R1 - Same table, more datasets
# Feeds: main results table (CIFAR-10-C / CIFAR-100-C / ImageNet-C headline)
#        and R3 Pareto plot (error vs ms/sample).
#
# Varies: dataset x method x seed.
# Fixed:  SETTING=gradual_mixed_corruption, SCHEDULE_TYPE=random, TIMING=on.
# Seeds:  override with `seeds="1 2 3 4 5" ./scripts/results/r1_datasets.sh`.
# Dry-run: DRY_RUN=1 ./scripts/results/r1_datasets.sh  (prints commands only)
# =============================================================================

set -euo pipefail

# --- conda activate (don't fail if conda isn't sourced yet) ---
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate tta
else
  echo "[warn] conda not found on PATH; assuming the tta env is already active." >&2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

seeds=(${seeds:-1 2 3 4 5})
DRY_RUN="${DRY_RUN:-0}"

run() {
  local cfg="$1"; shift
  if [[ ! -f "$cfg" ]]; then
    echo "[fatal] missing cfg: $cfg" >&2
    exit 2
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo python test_time.py --cfg "$cfg" "$@"
  else
    python test_time.py --cfg "$cfg" "$@"
  fi
}

# Full comparison set used on CIFAR-10-C and CIFAR-100-C (all 22 yaml'd).
methods_cifar=(
  source norm_test norm_alpha norm_ema tent rpl eta eata
  cotta rotta adacontrast lame sar santa deyo cmf rmt gtta
  roid aura_fc aura sgd_base
)

# ImageNet-C adds tpt + vte on top of the CIFAR set.
methods_imagenet=("${methods_cifar[@]}" tpt vte)

# seed -> method -> dataset for clean resume semantics.
for seed in "${seeds[@]}"; do
  for m in "${methods_cifar[@]}"; do
    run "cfgs/cifar10_c/${m}.yaml" \
        SETTING gradual_mixed_corruption \
        MIXED_GRADUAL.SCHEDULE_TYPE random \
        TIMING.ENABLED True \
        RNG_SEED "$seed"
  done
  for m in "${methods_cifar[@]}"; do
    run "cfgs/cifar100_c/${m}.yaml" \
        SETTING gradual_mixed_corruption \
        MIXED_GRADUAL.SCHEDULE_TYPE random \
        TIMING.ENABLED True \
        RNG_SEED "$seed"
  done
  for m in "${methods_imagenet[@]}"; do
    run "cfgs/imagenet_c/${m}.yaml" \
        SETTING gradual_mixed_corruption \
        MIXED_GRADUAL.SCHEDULE_TYPE random \
        MODEL.ARCH resnet50 \
        TIMING.ENABLED True \
        RNG_SEED "$seed"
  done
done
