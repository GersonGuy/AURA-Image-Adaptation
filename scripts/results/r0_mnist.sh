#!/bin/bash -l
# =============================================================================
# Experiment R0 — MNIST-C main-table results.
#
# Methods: source, BN-test / BN-alpha / BN-EMA, sgd_base, roid, AURA.
# Setting: gradual_mixed_corruption with random schedule, severity 1, 6
#          corruption types, 5 seeds.
#
# Prereqs:
#   1. MNIST-C must live under data/mnist_c/<corruption>/{train,test}_{images,labels}.npy
#      (https://github.com/google-research/mnist-c).
#   2. The MNIST ResNet-18 backbone must be trained once:
#        python scripts/train_mnist_resnet18.py
#      which writes ./ckpt/resnet18_mnist.pth (referenced by the cfgs).
#
# Usage:
#   ./scripts/results/r0_mnist.sh
#   seeds="1 2 3" ./scripts/results/r0_mnist.sh
#   DRY_RUN=1     ./scripts/results/r0_mnist.sh
# =============================================================================

set -euo pipefail

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate tta || true
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

seeds=(${seeds:-1 2 3 4 5})
DRY_RUN="${DRY_RUN:-0}"

CORR6='["shot_noise","impulse_noise","motion_blur","glass_blur","fog","brightness"]'

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

methods=(source norm_test norm_alpha norm_ema sgd_base roid aura)

for seed in "${seeds[@]}"; do
  for m in "${methods[@]}"; do
    run "cfgs/mnist_c/${m}.yaml" \
        SETTING gradual_mixed_corruption \
        MIXED_GRADUAL.FRAMES 1000 \
        MIXED_GRADUAL.BATCH_SIZE 100 \
        MIXED_GRADUAL.SEVERITY 1 \
        MIXED_GRADUAL.SCHEDULE_TYPE random \
        MIXED_GRADUAL.CORRUPTION_TYPES "$CORR6" \
        TIMING.ENABLED True \
        RNG_SEED "$seed"
  done
done
