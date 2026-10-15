#!/bin/bash -l
# =============================================================================
# Ablation A5 — Pseudo-label vs ground-truth observation
#   Intended: toggle aura between (a) supervised (ground-truth labels) and
#             (b) unsupervised (pseudo-labels via argmax).
#   Dataset: CIFAR-10-C, gradual_mixed_corruption, random schedule
#   Seeds : 3
#   Feeds : One-row "noisy-observation tax" number vs headline.
#
# -----------------------------------------------------------------------------
# BLOCKED — requires a new cfg flag before this ablation can run.
# -----------------------------------------------------------------------------
# See methods/aura.py forward_and_adapt, ~lines 494–580:
#
#   * Dataloader passes `labels` in from ground truth (see utils/eval_utils.py
#     get_accuracy).
#   * Inside forward_and_adapt, the branch is:
#         if labels is not None:  y[labels[i]] = 1.0          # supervised
#         else:                    y[y_pred_probs.argmax()] = 1.0  # pseudo
#   * So the default run today is SUPERVISED (labels are always passed in).
#   * There is no cfg knob to force the pseudo-label path from outside the
#     method.
#
# Required change (not made here, do NOT edit aura.py as part of this
# script):
#
#   conf.py:
#       _C.AURA.USE_PSEUDO_LABELS = False   # new flag
#
#   methods/aura.py, in forward_and_adapt, read the flag and override:
#       if self.cfg.AURA.USE_PSEUDO_LABELS:
#           labels = None
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

cfg="cfgs/cifar10_c/aura.yaml"

# ---- Pre-flight ----
if [[ ! -f "$cfg" ]]; then
  echo "[A5] MISSING CFG: $cfg" >&2; exit 1
fi

echo "[A5] BLOCKED: requires new cfg flag AURA.USE_PSEUDO_LABELS"
echo "[A5] See the CAVEAT comment at top of this script for the required edit."
echo "[A5] Not running any cells."
exit 2

# ---- Intended command structure (one edit away from running) --------------
# run() {
#   echo "+ $*"
#   [[ "$DRY_RUN" = "1" ]] || "$@"
# }
#
# modes=(False True)   # False = supervised (ground-truth); True = pseudo-labels
#
# for pl in "${modes[@]}"; do
#   for seed in "${seeds[@]}"; do
#     run python test_time.py --cfg "$cfg" \
#       RNG_SEED "$seed" \
#       AURA.USE_PSEUDO_LABELS "$pl" \
#       TIMING.ENABLED True
#   done
# done
