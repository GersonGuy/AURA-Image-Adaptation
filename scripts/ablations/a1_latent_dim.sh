#!/bin/bash
# Ablation A1 — LATENT_DIM sweep WITH per-dim offline pretraining.
#   For each B in {16, 32, 64, 128, 256, 500, 1000}:
#     1. Train offline ckpt (offline_for_a4.yaml as template, LATENT_DIM=B)
#     2. Run 3 online seeds using that ckpt
#   Total: 7 offline + 21 online.
#   Dataset: CIFAR-10-C, gradual_mixed_corruption, random schedule.
#   Feeds: error + ms/sample vs latent dim plot.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  source "$REPO_ROOT/.venv/bin/activate"
fi

DRY_RUN="${DRY_RUN:-0}"
seeds=(${seeds:-1 2 3 4 5})
latent_dims=(${latent_dims:-16 32 64 128 256 500 1000})

offline_cfg="cfgs/cifar10_c/offline_for_a4.yaml"
online_cfg="cfgs/cifar10_c/aura.yaml"
for p in "$offline_cfg" "$online_cfg"; do
  [[ -f "$p" ]] || { echo "[A1] MISSING CFG: $p" >&2; exit 1; }
done

run() { echo "+ $*"; [[ "$DRY_RUN" = "1" ]] || "$@"; }

echo "[A1] latent_dims=${latent_dims[*]} seeds=${seeds[*]}"
echo "[A1] Plan: ${#latent_dims[@]} offline + $(( ${#latent_dims[@]} * ${#seeds[@]} )) online"

for B in "${latent_dims[@]}"; do
  echo "=== [A1] latent_dim=$B ==="

  before=$(ls -dt ckpt/offline_aura_fc/cifar10_c__Standard__*d__*__sev5__* 2>/dev/null | head -1 || echo "")
  run python offline_train_aura.py --cfg "$offline_cfg" \
    AURA.LATENT_DIM "$B"

  if [[ "$DRY_RUN" = "1" ]]; then
    ckpt="<DRY_RUN_CKPT_B${B}>"
  else
    newest=$(ls -dt ckpt/offline_aura_fc/cifar10_c__Standard__*d__*__sev5__* 2>/dev/null | head -1)
    if [[ -z "$newest" || "$newest" = "$before" ]]; then
      echo "[A1] ERROR: no new offline ckpt produced for latent_dim=$B" >&2
      exit 2
    fi
    ckpt="${newest}/best_params.pth"
    if [[ ! -f "$ckpt" ]]; then
      ckpt="${newest}/last_params.pth"
    fi
    if [[ ! -f "$ckpt" ]]; then
      echo "[A1] ERROR: offline training produced no weights in $newest (latent_dim=$B)" >&2
      ls -la "$newest" >&2 || true
      exit 3
    fi
    echo "[A1] latent_dim=$B ckpt=$ckpt"
  fi

  for seed in "${seeds[@]}"; do
    run python test_time.py --cfg "$online_cfg" \
      RNG_SEED "$seed" \
      AURA.LATENT_DIM "$B" \
      AURA.OFFLINE_CKPT "$ckpt" \
      TIMING.ENABLED True
  done
done
