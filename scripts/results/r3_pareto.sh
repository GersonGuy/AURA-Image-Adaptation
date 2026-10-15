#!/bin/bash -l
# =============================================================================
# Experiment R3 - Error vs. adaptation-time Pareto plot
# Feeds: the paper's headline figure (one panel per dataset).
#
# No new training runs. Aggregates completed results under
# ckpt/gradual_mixed_results/<method>/<timestamp>/ that already contain
# acc_per_batch.npy + timing.json, then drops figures/pareto.pdf.
#
# Dry-run: DRY_RUN=1 ./scripts/results/r3_pareto.sh
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

DRY_RUN="${DRY_RUN:-0}"
root="${root:-ckpt/gradual_mixed_results}"
out="${out:-figures/pareto.pdf}"
plotter="scripts/plot_pareto.py"

if [[ ! -f "$plotter" ]]; then
  echo "[fatal] missing helper: $plotter" >&2
  exit 2
fi
if [[ ! -d "$root" ]]; then
  echo "[fatal] no results root at $root - run R1 first." >&2
  exit 2
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo python "$plotter" --root "$root" --out "$out"
else
  python "$plotter" --root "$root" --out "$out"
fi
