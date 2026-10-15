#!/bin/bash
# R4 - Schedule-type robustness.
# Varies: MIXED_GRADUAL.SCHEDULE_TYPE in {random, linear_overlap, waypoint}
#         x method x seed.
# Fixed:  dataset=cifar10_c, SETTING=gradual_mixed_corruption, TIMING=on.
# Methods: aura, norm_test, norm_alpha, norm_ema, roid.
# Seeds:   override with `seeds="1 2 3 4 5" ./scripts/results/r4_schedules.sh`.
# Dry-run: DRY_RUN=1 ./scripts/results/r4_schedules.sh
#
# GOTCHA: baselines ship without MIXED_GRADUAL blocks — must pass FRAMES etc.
# explicitly on every cell or sample counts drop to 6k from conf.py defaults.
set -u

SCRIPT_DIR_GUARD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_GUARD="$(cd "$SCRIPT_DIR_GUARD/../.." && pwd)"
cd "$REPO_ROOT_GUARD"

if [[ -f "$REPO_ROOT_GUARD/.venv/bin/activate" ]]; then
  source "$REPO_ROOT_GUARD/.venv/bin/activate"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

seeds=(${seeds:-1 2 3 4 5})
DRY_RUN="${DRY_RUN:-0}"

run() {
  local cfg="$1"; shift
  [[ -f "$cfg" ]] || { echo "[r4] MISSING CFG: $cfg" >&2; exit 2; }
  echo "+ python test_time.py --cfg $cfg $*"
  [[ "$DRY_RUN" = "1" ]] || python test_time.py --cfg "$cfg" "$@"
}

methods=(aura norm_test norm_alpha norm_ema roid)
schedules=(random linear_overlap waypoint)

CORR_LIST="['gaussian_noise','shot_noise','impulse_noise','defocus_blur','glass_blur','motion_blur']"
gm_base=(
  SETTING gradual_mixed_corruption
  MIXED_GRADUAL.FRAMES 1000
  MIXED_GRADUAL.BATCH_SIZE 100
  MIXED_GRADUAL.SEVERITY 5
  MIXED_GRADUAL.CORRUPTION_TYPES "$CORR_LIST"
  TIMING.ENABLED True
)

# Pick the most recent offline-AURA ckpt; override with AURA_CKPT_C10. If absent, AURA cold-starts.
aura_fc_ckpt_c10="${AURA_CKPT_C10:-$(ls -dt ckpt/offline_aura_fc/cifar10_c__Standard__*d__*__sev5__* 2>/dev/null | head -1)/best_params.pth}"

for seed in "${seeds[@]}"; do
  for m in "${methods[@]}"; do
    extra=()
    [[ "$m" = "aura" && -f "$aura_fc_ckpt_c10" ]] && extra=(AURA.OFFLINE_CKPT "$aura_fc_ckpt_c10")
    for sch in "${schedules[@]}"; do
      run "cfgs/cifar10_c/${m}.yaml" "${gm_base[@]}" MIXED_GRADUAL.SCHEDULE_TYPE "$sch" RNG_SEED "$seed" "${extra[@]}"
    done
  done
done
