#!/bin/bash
# A1-c100 — aura latent-dim sweep on CIFAR-100-C at higher capacity.
# FC block on ResNeXt is 102,500 params (16x CIFAR-10's 6,410), so 500d may be
# undercapacity there. Sweep {500, 1000, 5000}, 3 seeds per dim.
# 5000 may OOM offline on 8 GB (A alone is 2 GB; with Adam state ~8 GB).
# If OOM, either skip 5000 or rerun it with OFFLINE.Q_SAMPLES smaller.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  source "$REPO_ROOT/.venv/bin/activate"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DRY_RUN="${DRY_RUN:-0}"
seeds=(${seeds:-1 2 3 4 5})
latent_dims=(${latent_dims:-500 1000 5000})

offline_cfg="cfgs/cifar100_c/offline_for_a4.yaml"
online_cfg="cfgs/cifar100_c/aura.yaml"
for p in "$offline_cfg" "$online_cfg"; do
  [[ -f "$p" ]] || { echo "[A1c100] MISSING CFG: $p" >&2; exit 1; }
done

run() { echo "+ $*"; [[ "$DRY_RUN" = "1" ]] || "$@"; }

echo "[A1c100] latent_dims=${latent_dims[*]} seeds=${seeds[*]}"

for B in "${latent_dims[@]}"; do
  echo "=== [A1c100] latent_dim=$B ==="

  before=$(ls -dt ckpt/offline_aura_fc/cifar100_c__Hendrycks2020AugMix_ResNeXt__*d__*__sev5__* 2>/dev/null | head -1 || echo "")

  # Reduced offline recipe to match A1 cifar10 (500/3/10).
  run python offline_train_aura.py --cfg "$offline_cfg" \
    AURA.LATENT_DIM "$B" \
    OFFLINE.NUM_EPISODES 500 \
    OFFLINE.NUM_UPDATES_PER_EPISODE 3 \
    OFFLINE.Q_SAMPLES 10
  rc=$?

  if [[ "$DRY_RUN" = "1" ]]; then
    ckpt="<DRY_RUN_CKPT_B${B}>"
  else
    newest=$(ls -dt ckpt/offline_aura_fc/cifar100_c__Hendrycks2020AugMix_ResNeXt__*d__*__sev5__* 2>/dev/null | head -1)
    if [[ -z "$newest" || "$newest" = "$before" || $rc -ne 0 ]]; then
      echo "[A1c100] SKIP latent_dim=$B: no new ckpt produced (rc=$rc). Likely OOM." >&2
      continue
    fi
    ckpt="${newest}/best_params.pth"
    if [[ ! -f "$ckpt" ]]; then
      ckpt="${newest}/last_params.pth"
    fi
    if [[ ! -f "$ckpt" ]]; then
      echo "[A1c100] SKIP latent_dim=$B: no weights file in $newest" >&2
      ls -la "$newest" >&2 || true
      continue
    fi
    echo "[A1c100] latent_dim=$B ckpt=$ckpt"
  fi

  for seed in "${seeds[@]}"; do
    run python test_time.py --cfg "$online_cfg" \
      RNG_SEED "$seed" \
      AURA.LATENT_DIM "$B" \
      AURA.OFFLINE_CKPT "$ckpt" \
      TIMING.ENABLED True
  done
done
