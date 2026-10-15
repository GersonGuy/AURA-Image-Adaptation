#!/bin/bash
# =============================================================================
# Ablation — Offline labeled-data efficiency
#   Varies: OFFLINE.NUM_EPISODES ∈ {1, 10, 100, 1000}
#   Per budget: 1 offline training run → 3 online eval seeds
#   Total: 4 offline runs + 12 online runs
#   Dataset: CIFAR-10-C, gradual_mixed_corruption, random schedule
#   Feeds : Plot of online error % vs offline budget (log-x).
# =============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  source "$REPO_ROOT/.venv/bin/activate"
fi

DRY_RUN="${DRY_RUN:-0}"
seeds=(${seeds:-1 2 3})
budgets=(${budgets:-1 10 100 1000})

offline_cfg="cfgs/cifar10_c/offline_aura.yaml"
online_cfg="cfgs/cifar10_c/aura.yaml"
for p in "$offline_cfg" "$online_cfg"; do
  [[ -f "$p" ]] || { echo "[a_eff] MISSING CFG: $p" >&2; exit 1; }
done

run() { echo "+ $*"; [[ "$DRY_RUN" = "1" ]] || "$@"; }

total_online=$(( ${#budgets[@]} * ${#seeds[@]} ))
echo "[a_eff] Plan: ${#budgets[@]} offline runs + $total_online online runs"

for B in "${budgets[@]}"; do
  echo "=== [a_eff] offline budget = $B episodes ==="

  # Capture the set of existing offline ckpts before training; we pick the
  # newest one that appears afterwards as this run's output (no rename hook).
  before=$(ls -dt ckpt/offline_aura_fc/cifar10_c__Standard__500d__* 2>/dev/null | head -1 || echo "")

  run python offline_train_aura.py --cfg "$offline_cfg" \
    OFFLINE.NUM_EPISODES "$B"

  if [[ "$DRY_RUN" = "1" ]]; then
    ckpt="<DRY_RUN_CKPT_B${B}>"
  else
    newest=$(ls -dt ckpt/offline_aura_fc/cifar10_c__Standard__500d__* 2>/dev/null | head -1)
    if [[ -z "$newest" || "$newest" = "$before" ]]; then
      echo "[a_eff] ERROR: no new offline ckpt produced for budget=$B" >&2
      exit 2
    fi
    ckpt="$newest"
    echo "[a_eff] budget=$B ckpt=$ckpt"
  fi

  for seed in "${seeds[@]}"; do
    run python test_time.py --cfg "$online_cfg" \
      RNG_SEED "$seed" \
      AURA.OFFLINE_CKPT "$ckpt" \
      TIMING.ENABLED True
  done
done
