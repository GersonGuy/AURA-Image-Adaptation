#!/bin/bash
# CIFAR-10-C latency / peak-memory / FLOPs driver.
#
# One subprocess per (method, run) for CUDA isolation. STRICTLY SEQUENTIAL --
# these are timing measurements, any concurrent GPU work invalidates them.
# Resumable: skips cells whose JSON already exists.
#
# RUNS>1 because the process-to-process spread (GPU clock ramp, host-side kernel
# launch jitter) is larger than the within-process std for the methods whose
# frame is a long chain of tiny kernels (AURA: 15 sequential batch-1 Jacobians).
set -u

cd /mnt/d/projects/TTA-Latent-Representation
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

mkdir -p logs/efficiency_cells

RUNS="${RUNS:-3}"
METHODS=("Source" "BN-Test" "EcoTTA" "TinyTTA" "AURA" "ELaTTA")
[ $# -gt 0 ] && METHODS=("$@")

for run in $(seq 0 $((RUNS - 1))); do
  for m in "${METHODS[@]}"; do
    sfx=""; [ "$run" -ne 0 ] && sfx="__run${run}"
    fname="logs/efficiency_cells/cifar10_c__${m// /_}${sfx}.json"
    if [ -f "$fname" ]; then
      echo "skip $m run$run (cached)"
      continue
    fi
    echo "=== $m run$run @ $(date +%H:%M:%S) ==="
    timeout 1800 python -u scripts/measure_efficiency.py --method "$m" --run "$run" \
      || echo "  (subprocess returned $?)"
  done
done

echo ""
echo "=== aggregate ==="
python -u scripts/measure_efficiency.py --aggregate
