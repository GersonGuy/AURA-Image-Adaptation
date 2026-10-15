#!/bin/bash -l
# =============================================================================
# Ablation A2 — Stage selection sweep
#   Varies: AURA.STAGE_SELECTION over:
#             [fc]
#             [fc, stage_1]
#             [fc, stage_1, stage_2]
#             [fc, stage_1, stage_2, stage_3]
#             [stage_3]                (no fc; BN-only on deepest stage)
#   Dataset: CIFAR-10-C, gradual_mixed_corruption, random schedule
#   Seeds : 3 (override via $seeds)
#   Mode  : COLD-START (OFFLINE_CKPT="") — block topology changes per cell, so
#           the shipped [fc, stage_1]-500d ckpt can only legally be reused in
#           the a2_fc_stage1 cell. For cross-cell parity we cold-start ALL
#           cells.
#   Feeds : Stage-selection ablation table.
#
# YACS list-overrides on the CLI are fragile (they're parsed with literal_eval
# after splitting on whitespace). We author one small yaml per cell under
# cfgs/cifar10_c/_ablations/ instead. Each yaml is a full copy-override of the
# main aura.yaml that sets only OFFLINE_CKPT="" and STAGE_SELECTION.
# =============================================================================

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate tta || true
fi

set -u
DRY_RUN="${DRY_RUN:-0}"
seeds=(${seeds:-1 2 3 4 5})

cells=(
  "cfgs/cifar10_c/_ablations/a2_fc.yaml"
  "cfgs/cifar10_c/_ablations/a2_fc_stage1.yaml"
  "cfgs/cifar10_c/_ablations/a2_fc_stage12.yaml"
  "cfgs/cifar10_c/_ablations/a2_fc_stage123.yaml"
  "cfgs/cifar10_c/_ablations/a2_stage3.yaml"
)

# ---- Pre-flight ----
for cfg in "${cells[@]}"; do
  if [[ ! -f "$cfg" ]]; then
    echo "[A2] MISSING CFG: $cfg" >&2; exit 1
  fi
done

run() {
  echo "+ $*"
  [[ "$DRY_RUN" = "1" ]] || "$@"
}

total=$(( ${#cells[@]} * ${#seeds[@]} ))
echo "[A2] Running $total cells (stage_sels=${#cells[@]} x seeds=${#seeds[@]})"

for cfg in "${cells[@]}"; do
  for seed in "${seeds[@]}"; do
    run python test_time.py --cfg "$cfg" \
      RNG_SEED "$seed" \
      TIMING.ENABLED True
  done
done
