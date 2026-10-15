#!/bin/bash -l
# =============================================================================
# Ablation A7 — Batch-size robustness
#   Varies : MIXED_GRADUAL.BATCH_SIZE ∈ {1, 8, 32, 100}
#   Methods: aura, norm_test, norm_alpha, norm_ema
#   Seeds  : 3
#   Dataset: CIFAR-10-C, gradual_mixed_corruption, random schedule
#   Feeds  : Figure "error vs batch size" — BN-stat baselines are expected
#            to degrade at small batch, aura should hold.
#
# Notes on small-batch behavior — these are known shipped code behaviors, NOT
# bugs to fix in this script. We also set TEST.BATCH_SIZE to match, because
# methods/norm.py reads TEST.BATCH_SIZE (not MIXED_GRADUAL.BATCH_SIZE) to
# decide whether to keep BN1d in train mode:
#
#   * norm_test at batch_size=1: methods/norm.py line 67–69 explicitly keeps
#     BN1d in eval mode (degenerate single-sample stats), but BN2d still runs
#     in train mode — output is almost certainly garbage at bs=1, because
#     BN2d stats over 1 sample are also degenerate. This is the expected
#     collapse the plot wants to show. Does NOT crash.
#   * norm_alpha, norm_ema at batch_size=1: both forward-pass-only; will run
#     but numerics are degenerate for the same reason. Does NOT crash.
#   * aura at batch_size=1: forward_and_adapt slices x[:num_updates] and
#     feeds x[i:i+1] into jacfwd/jacrev one sample at a time anyway, so
#     bs=1 is the normal inner-loop shape — should work unchanged.
#
# Do NOT modify methods/*.py from this script.
# =============================================================================

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate tta || true
fi

set -u
DRY_RUN="${DRY_RUN:-0}"
seeds=(${seeds:-1 2 3 4 5})

batch_sizes=(1 8 32 100)
methods=(aura norm_test norm_alpha norm_ema)

# ---- Pre-flight ----
for m in "${methods[@]}"; do
  cfg="cfgs/cifar10_c/${m}.yaml"
  if [[ ! -f "$cfg" ]]; then
    echo "[A7] MISSING CFG: $cfg" >&2; exit 1
  fi
done

run() {
  echo "+ $*"
  [[ "$DRY_RUN" = "1" ]] || "$@"
}

total=$(( ${#batch_sizes[@]} * ${#methods[@]} * ${#seeds[@]} ))
echo "[A7] Running $total cells (bs=${#batch_sizes[@]} x methods=${#methods[@]} x seeds=${#seeds[@]})"

for m in "${methods[@]}"; do
  cfg="cfgs/cifar10_c/${m}.yaml"
  for bs in "${batch_sizes[@]}"; do
    for seed in "${seeds[@]}"; do
      # Force gradual_mixed_corruption so baselines run in the same setting
      # as aura (norm_*.yaml defaults may be continual).
      run python test_time.py --cfg "$cfg" \
        SETTING gradual_mixed_corruption \
        RNG_SEED "$seed" \
        MIXED_GRADUAL.BATCH_SIZE "$bs" \
        TEST.BATCH_SIZE "$bs" \
        TIMING.ENABLED True
    done
  done
done
