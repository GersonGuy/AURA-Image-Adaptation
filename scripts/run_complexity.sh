#!/bin/bash
# Driver: one subprocess per (dataset, method) for CUDA isolation.
# Resumable -- skips cells whose JSON already exists. Aggregates at the end.
set -u

cd /mnt/d/projects/TTA-Latent-Representation
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p logs/complexity_cells

DATASETS=(mnist_c cifar10_c cifar100_c)
METHODS=("Source" "BN-Test" "BN-Alpha" "BN-EMA" "Online GD" "ROID" "EKF-FC" "AURA")

for ds in "${DATASETS[@]}"; do
  for m in "${METHODS[@]}"; do
    fname="logs/complexity_cells/${ds}__${m// /_}.json"
    if [ -f "$fname" ]; then
      echo "skip $ds $m (cached)"
      continue
    fi
    echo "=== $ds $m @ $(date +%H:%M:%S) ==="
    timeout 900 python -u scripts/measure_complexity.py --dataset "$ds" --method "$m" \
      2> >(grep -v -E "FutureWarning|weights_only|checkpoint =|^$" >&2) \
      || echo "  (subprocess returned $?)"
  done
done

echo ""
echo "=== aggregate ==="
python -u scripts/measure_complexity.py --aggregate
