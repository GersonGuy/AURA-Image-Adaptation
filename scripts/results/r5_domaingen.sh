#!/bin/bash -l
# =============================================================================
# Experiment R5 - Offline-on-M / online-on-N domain-generalization (CIFAR-10-C)
# Feeds: domain-gen table showing aura transfers across disjoint corruption
#        pools (weather/digital M -> noise/blur N).
#
# Step 1: offline-train the A projection on the weather/digital M pool.
# Step 2: online-eval aura on the noise/blur N pool, loading the ckpt above.
#
# Disjoint pools:
#   M (offline)  = zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform
#   N (online)   = gaussian_noise, shot_noise, impulse_noise, defocus_blur,
#                  glass_blur, motion_blur
# The N pool is already what cfgs/cifar10_c/aura.yaml ships with under
# MIXED_GRADUAL.CORRUPTION_TYPES, so step 2 does NOT need a CLI list override
# (yacs list parsing from bash is fragile; we avoid it by construction).
#
# Seeds: override with `seeds="1 2 3 4 5" ./scripts/results/r5_domaingen.sh`.
# Dry-run: DRY_RUN=1 ./scripts/results/r5_domaingen.sh  (prints commands only)
# Skip offline training (reuse latest matching ckpt): SKIP_OFFLINE=1 ...
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
SKIP_OFFLINE="${SKIP_OFFLINE:-0}"

offline_cfg="cfgs/cifar10_c/offline_aura_domaingen_weather.yaml"
online_cfg="cfgs/cifar10_c/aura.yaml"

for cfg in "$offline_cfg" "$online_cfg"; do
  if [[ ! -f "$cfg" ]]; then
    echo "[fatal] missing cfg: $cfg" >&2
    exit 2
  fi
done

# --------- Step 1: offline train on M (weather/digital) ---------
if [[ "$SKIP_OFFLINE" == "1" ]]; then
  echo "[info] SKIP_OFFLINE=1 -> reusing newest existing M-pool ckpt"
else
  if [[ "$DRY_RUN" == "1" ]]; then
    echo python offline_train_aura.py --cfg "$offline_cfg"
  else
    python offline_train_aura.py --cfg "$offline_cfg"
  fi
fi

# --------- Resolve the ckpt dir produced by step 1 ---------
# _build_save_dir format:
#   ckpt/offline_aura_fc/<dataset>__<arch>__<latent>d__<corruption_tag>__<sev>__<ts>
# For the 7-corruption weather pool the joined string is >60 chars so the tag
# becomes "big_dataset". We pin arch=Standard, latent=500d, sev=sev5 and take
# the newest match. If the glob matches nothing or if multiple recent runs
# could be confused, we error instructively.
shopt -s nullglob
matches=(ckpt/offline_aura_fc/cifar10_c__Standard__*d__*__sev5__*)
shopt -u nullglob

if [[ "$DRY_RUN" == "1" && "$SKIP_OFFLINE" != "1" ]]; then
  latest_ckpt_dir="<resolved-after-offline-run>"
else
  if [[ ${#matches[@]} -eq 0 ]]; then
    echo "[fatal] no offline-aura_fc ckpt dir matched" >&2
    echo "        expected: ckpt/offline_aura_fc/cifar10_c__Standard__*d__*__sev5__*" >&2
    echo "        run step 1 first (drop SKIP_OFFLINE=1) or check _build_save_dir." >&2
    exit 3
  fi
  latest_ckpt_dir=$(ls -td ckpt/offline_aura_fc/cifar10_c__Standard__*d__*__sev5__* | head -1)
fi

latest_ckpt="${latest_ckpt_dir}/best_params.pth"
echo "[info] using offline ckpt: $latest_ckpt"

if [[ "$DRY_RUN" != "1" && ! -f "$latest_ckpt" ]]; then
  echo "[fatal] expected checkpoint file not found: $latest_ckpt" >&2
  echo "        the offline run may have crashed before the first 'best' save;" >&2
  echo "        inspect ${latest_ckpt_dir} for last_params.pth as a fallback." >&2
  exit 4
fi

# --------- Step 2: online eval on N (noise/blur) ---------
# cfgs/cifar10_c/aura.yaml already declares N under MIXED_GRADUAL.CORRUPTION_TYPES,
# so we only override OFFLINE_CKPT + timing + seed.
run_online() {
  local seed="$1"
  local cmd=(
    python test_time.py --cfg "$online_cfg"
    SETTING gradual_mixed_corruption
    MIXED_GRADUAL.SCHEDULE_TYPE random
    AURA.OFFLINE_CKPT "$latest_ckpt"
    TIMING.ENABLED True
    RNG_SEED "$seed"
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

for seed in "${seeds[@]}"; do
  run_online "$seed"
done
