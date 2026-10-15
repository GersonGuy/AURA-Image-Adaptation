#!/bin/bash -l
# =============================================================================
# Experiment R2 - Standard continual setting (full 15-corruption sequence)
# Feeds: continual-setting comparison table (apples-to-apples with published
#        TENT / CoTTA / ROID / SAR numbers).
#
# Varies: dataset x method x seed.
# Fixed:  SETTING=continual, TIMING=on. CORRUPTION.TYPE is whatever each cfg
#         already declares (all 15 corruptions in the shipped yamls).
# Seeds:  override with `seeds="1 2 3 4 5" ./scripts/results/r2_continual.sh`.
# Dry-run: DRY_RUN=1 ./scripts/results/r2_continual.sh
# =============================================================================

set -euo pipefail

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

methods_cifar=(
  source norm_test norm_alpha norm_ema tent rpl eta eata
  cotta rotta adacontrast lame sar santa deyo cmf rmt gtta
  roid aura_fc aura sgd_base
)
methods_imagenet=("${methods_cifar[@]}" tpt vte)

settings=(${settings:-continual correlated})

for setting in "${settings[@]}"; do
  for seed in "${seeds[@]}"; do
    for m in "${methods_cifar[@]}"; do
      run "cfgs/cifar10_c/${m}.yaml" \
          SETTING "$setting" \
          TIMING.ENABLED True \
          RNG_SEED "$seed"
    done
    for m in "${methods_cifar[@]}"; do
      run "cfgs/cifar100_c/${m}.yaml" \
          SETTING "$setting" \
          TIMING.ENABLED True \
          RNG_SEED "$seed"
    done
    for m in "${methods_imagenet[@]}"; do
      run "cfgs/imagenet_c/${m}.yaml" \
          SETTING "$setting" \
          MODEL.ARCH resnet50 \
          TIMING.ENABLED True \
          RNG_SEED "$seed"
    done
  done
done
