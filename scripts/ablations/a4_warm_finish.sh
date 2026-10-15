#!/bin/bash
# Finish A4's missing cifar10_c warm-start cells.
# Step 1: train a local offline ckpt that matches cfgs/cifar10_c/aura.yaml.
# Step 2: run 3 warm-start seeds with that ckpt.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  source "$REPO_ROOT/.venv/bin/activate"
fi

DRY_RUN="${DRY_RUN:-0}"
seeds=(${seeds:-1 2 3 4 5})
offline_cfg="cfgs/cifar10_c/offline_for_a4.yaml"
online_cfg="cfgs/cifar10_c/aura.yaml"

for p in "$offline_cfg" "$online_cfg"; do
  [[ -f "$p" ]] || { echo "[a4_warm] MISSING CFG: $p" >&2; exit 1; }
done

run() { echo "+ $*"; [[ "$DRY_RUN" = "1" ]] || "$@"; }

echo "=== [a4_warm] Step 1: offline training ==="
before=$(ls -dt ckpt/offline_aura_fc/cifar10_c__Standard__*d__*__sev5__* 2>/dev/null | head -1 || echo "")
run python offline_train_aura.py --cfg "$offline_cfg"

if [[ "$DRY_RUN" = "1" ]]; then
  ckpt="<DRY_RUN_CKPT>"
else
  newest=$(ls -dt ckpt/offline_aura_fc/cifar10_c__Standard__*d__*__sev5__* 2>/dev/null | head -1)
  if [[ -z "$newest" || "$newest" = "$before" ]]; then
    echo "[a4_warm] ERROR: no new offline ckpt produced" >&2
    exit 2
  fi
  ckpt="${newest}/best_params.pth"
  if [[ ! -f "$ckpt" ]]; then
    ckpt="${newest}/last_params.pth"
  fi
  if [[ ! -f "$ckpt" ]]; then
    echo "[a4_warm] ERROR: offline training produced no weights in $newest" >&2
    ls -la "$newest" >&2 || true
    exit 3
  fi
  echo "[a4_warm] using ckpt: $ckpt"
fi

echo "=== [a4_warm] Step 2: online warm-start x 3 seeds ==="
for seed in "${seeds[@]}"; do
  run python test_time.py --cfg "$online_cfg" \
    RNG_SEED "$seed" \
    AURA.OFFLINE_CKPT "$ckpt" \
    TIMING.ENABLED True
done
