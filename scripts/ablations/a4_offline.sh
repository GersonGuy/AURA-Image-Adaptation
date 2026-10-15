#!/bin/bash -l
# =============================================================================
# Ablation A4 — Offline pretraining on/off
#   Cells  : (a) OFFLINE_CKPT = shipped ckpt for that dataset (warm-start)
#            (b) OFFLINE_CKPT = ""                            (cold-start)
#   Seeds  : 3 per cell
#   Setting: gradual_mixed_corruption, random schedule
#   Feeds  : "does the offline stage matter" table.
#
# LIMITATION: CIFAR-100-C has no shipped offline checkpoint yet
# (cfgs/cifar100_c/aura.yaml ships with OFFLINE_CKPT=""). On that dataset
# only the cold-start cell is runnable — the warm-start cell is skipped with
# a loud log line. Producing a CIFAR-100 offline ckpt is an offline_train_aura.py
# job that must happen before the warm-start cell can run, and is out of scope
# for this script.
# =============================================================================

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate tta || true
fi

set -u
DRY_RUN="${DRY_RUN:-0}"
seeds=(${seeds:-1 2 3 4 5})

datasets=(cifar10_c cifar100_c)

# Warm-start ckpts, per dataset. Default: pick the most recent offline_train_aura
# output. Override with AURA_CKPT_C10 / AURA_CKPT_C100. Run scripts/prepare_offline.sh
# first if you don't have one. Empty string => warm-start cell skipped.
_c10_ckpt="${AURA_CKPT_C10:-$(ls -dt ckpt/offline_aura_fc/cifar10_c__Standard__*d__*__sev5__* 2>/dev/null | head -1)}"
_c100_ckpt="${AURA_CKPT_C100:-$(ls -dt ckpt/offline_aura_fc/cifar100_c__*__*d__*__sev5__* 2>/dev/null | head -1)}"
[[ -d "$_c10_ckpt"  ]] || _c10_ckpt=""
[[ -d "$_c100_ckpt" ]] || _c100_ckpt=""
declare -A WARM_CKPT=(
  [cifar10_c]="$_c10_ckpt"
  [cifar100_c]="$_c100_ckpt"
)

# ---- Pre-flight ----
for ds in "${datasets[@]}"; do
  cfg="cfgs/${ds}/aura.yaml"
  if [[ ! -f "$cfg" ]]; then
    echo "[A4] MISSING CFG: $cfg" >&2; exit 1
  fi
done

run() {
  echo "+ $*"
  [[ "$DRY_RUN" = "1" ]] || "$@"
}

echo "[A4] datasets=${datasets[*]} seeds=${seeds[*]}"

for ds in "${datasets[@]}"; do
  cfg="cfgs/${ds}/aura.yaml"
  warm="${WARM_CKPT[$ds]}"

  # --- Cell (a): warm-start ---
  if [[ -n "$warm" ]]; then
    for seed in "${seeds[@]}"; do
      run python test_time.py --cfg "$cfg" \
        RNG_SEED "$seed" \
        AURA.OFFLINE_CKPT "$warm" \
        TIMING.ENABLED True
    done
  else
    echo "[A4] SKIP warm-start for $ds: no shipped OFFLINE_CKPT — cold-start-only dataset."
  fi

  # --- Cell (b): cold-start ---
  for seed in "${seeds[@]}"; do
    run python test_time.py --cfg "$cfg" \
      RNG_SEED "$seed" \
      AURA.OFFLINE_CKPT "" \
      TIMING.ENABLED True
  done
done
