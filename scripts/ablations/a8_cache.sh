#!/bin/bash -l
# =============================================================================
# Ablation A8 — Partial-backbone cache speedup (graph splicing on/off)
#   Intended: compare ms/sample with and without `can_slice` graph splicing at
#             matched accuracy.
#   Dataset: CIFAR-10-C, gradual_mixed_corruption, random schedule
#   Seeds : 3
#   Feeds : Systems table supporting the ~30 ms headline number.
#
# -----------------------------------------------------------------------------
# BLOCKED — requires a new cfg flag before this ablation can run.
# -----------------------------------------------------------------------------
# See methods/aura.py _setup_slicing (~lines 212–309):
#
#   * `self.can_slice` is decided purely from block topology: set to True iff
#     the earliest adapted stage index > 0 (i.e. stage_1 or deeper is frozen).
#   * There is no cfg knob to FORCE can_slice=False while keeping the same
#     STAGE_SELECTION. Flipping STAGE_SELECTION to [fc, stage_1, ...] to
#     disable slicing changes the method's adapted params too — that's a
#     confound, not an isolation of the cache optimization.
#
# Required change (not made here, do NOT edit aura.py as part of this
# script):
#
#   conf.py:
#       _C.AURA.FORCE_NO_SLICE = False   # new flag
#
#   methods/aura.py, at the end of _setup_slicing (after `self.can_slice`
#   is set but before `self.head_fn/tail_fn` are installed):
#       if getattr(self.cfg.AURA, "FORCE_NO_SLICE", False):
#           self.can_slice = False
#           return
#
# Once that flag exists, uncomment the block at the bottom of this file and
# re-run. The structure is already correct; only the flag is missing.
# =============================================================================

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate tta || true
fi

set -u
DRY_RUN="${DRY_RUN:-0}"
seeds=(${seeds:-1 2 3 4 5})

# A2 per-cell yaml already fixes STAGE_SELECTION=[fc, stage_1] + cold-start,
# which is the headline configuration and the one where can_slice=True (earliest
# adapted stage is stage_1 > 0). That's the right cell to toggle the cache on
# and off.
cfg="cfgs/cifar10_c/_ablations/a2_fc_stage1.yaml"

# ---- Pre-flight ----
if [[ ! -f "$cfg" ]]; then
  echo "[A8] MISSING CFG: $cfg" >&2; exit 1
fi

echo "[A8] BLOCKED: requires new cfg flag AURA.FORCE_NO_SLICE"
echo "[A8] See the CAVEAT comment at top of this script for the required edit."
echo "[A8] Not running any cells."
exit 2

# ---- Intended command structure (one edit away from running) --------------
# run() {
#   echo "+ $*"
#   [[ "$DRY_RUN" = "1" ]] || "$@"
# }
#
# for force_no_slice in False True; do
#   for seed in "${seeds[@]}"; do
#     run python test_time.py --cfg "$cfg" \
#       RNG_SEED "$seed" \
#       AURA.FORCE_NO_SLICE "$force_no_slice" \
#       TIMING.ENABLED True
#   done
# done
