#!/bin/bash
# Train a local offline ckpt for CIFAR-100-C and run the A4 warm cell (3 seeds).
# Full-recipe offline (1000/10/30) so this cell is comparable to the cifar10
# warm cell in research.md.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  source "$REPO_ROOT/.venv/bin/activate"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "[a4_warm_c100] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

DRY_RUN="${DRY_RUN:-0}"
seeds=(${seeds:-1 2 3 4 5})
offline_cfg="cfgs/cifar100_c/offline_for_a4.yaml"
online_cfg="cfgs/cifar100_c/aura.yaml"

for p in "$offline_cfg" "$online_cfg"; do
  [[ -f "$p" ]] || { echo "[a4_warm_c100] MISSING CFG: $p" >&2; exit 1; }
done

run() { echo "+ $*"; [[ "$DRY_RUN" = "1" ]] || "$@"; }

echo "=== [a4_warm_c100] Step 1: offline training ==="
before=$(ls -dt ckpt/offline_aura_fc/cifar100_c__Hendrycks2020AugMix_ResNeXt__*d__*__sev5__* 2>/dev/null | head -1 || echo "")
run python offline_train_aura.py --cfg "$offline_cfg"

if [[ "$DRY_RUN" = "1" ]]; then
  ckpt="<DRY_RUN_CKPT>"
else
  newest=$(ls -dt ckpt/offline_aura_fc/cifar100_c__Hendrycks2020AugMix_ResNeXt__*d__*__sev5__* 2>/dev/null | head -1)
  if [[ -z "$newest" || "$newest" = "$before" ]]; then
    echo "[a4_warm_c100] ERROR: no new offline ckpt produced" >&2
    exit 2
  fi
  ckpt="${newest}/best_params.pth"
  [[ -f "$ckpt" ]] || ckpt="${newest}/last_params.pth"
  if [[ ! -f "$ckpt" ]]; then
    echo "[a4_warm_c100] ERROR: no weights file in $newest" >&2
    ls -la "$newest" >&2 || true
    exit 3
  fi
  echo "[a4_warm_c100] using ckpt: $ckpt"
fi

echo "=== [a4_warm_c100] Step 2: online warm-start x 3 seeds ==="
for seed in "${seeds[@]}"; do
  run python test_time.py --cfg "$online_cfg" \
    RNG_SEED "$seed" \
    AURA.OFFLINE_CKPT "$ckpt" \
    TIMING.ENABLED True
done
