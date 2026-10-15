#!/bin/bash -l
# =============================================================================
# Offline meta-train AURA on CIFAR-10 (and optionally CIFAR-100).
#
# Run once before evaluating AURA with OFFLINE_CKPT (i.e. before reproducing
# the headline offline-AURA result). The ckpt directory is timestamped, so
# the script prints the produced path at the end — paste it into the
# OFFLINE_CKPT field of the corresponding online cfg
# (cfgs/cifar10_c/aura.yaml or aura_fc.yaml).
#
# Usage:
#   ./scripts/prepare_offline.sh                # cifar10 (default)
#   DATASET=cifar10 ./scripts/prepare_offline.sh
#   VARIANT=aura_fc ./scripts/prepare_offline.sh
#
# Requires: conda env "tta" with the project deps installed; CIFAR-10 fetched
# automatically by RobustBench on first run.
# =============================================================================

set -euo pipefail

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate tta || true
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DATASET:-cifar10}"
VARIANT="${VARIANT:-aura}"   # aura (FC + BN) or aura_fc (FC-only)

case "$DATASET" in
  cifar10) cfg="cfgs/cifar10_c/offline_aura.yaml" ;;
  *) echo "[fatal] unsupported DATASET='$DATASET' (only cifar10 has a packaged offline cfg)"; exit 2 ;;
esac

if [[ ! -f "$cfg" ]]; then
  echo "[fatal] missing offline cfg: $cfg" >&2
  exit 2
fi

echo "=== Offline AURA training: $cfg (variant=$VARIANT) ==="
python offline_train_aura.py --cfg "$cfg" MODEL.ADAPTATION "$VARIANT"

# Surface the most recently written ckpt directory.
latest="$(ls -td ckpt/offline_aura_fc/${DATASET}_c__*/ 2>/dev/null | head -n 1 || true)"
if [[ -n "$latest" ]]; then
  latest="${latest%/}"
  echo
  echo "=== Offline ckpt produced ==="
  echo "$latest"
  echo
  echo "Paste this path into the OFFLINE_CKPT field of"
  echo "  cfgs/${DATASET}_c/${VARIANT}.yaml"
  echo "to run AURA online with the offline-meta-trained representation."
else
  echo "[warn] could not locate produced ckpt under ckpt/offline_aura_fc/" >&2
fi
