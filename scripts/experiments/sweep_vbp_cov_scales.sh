#!/usr/bin/env bash
# ==============================================================================
# Sweep VBP across a 3D grid: NUM_UPDATES x INIT_COV x OBS_COV
# Evaluates each configuration across 3 random seeds and outputs a ranked table.
# ==============================================================================

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate tta || true
fi

# Grid hyperparameters
INIT_COVS=(${INIT_COVS:-1e-3 1e-4 1e-5 1e-6})
OBS_COVS=(${OBS_COVS:-0.1 1.0 10.0})
SEEDS=(${SEEDS:-1 2 3})
NUM_UPDATES=(${NUM_UPDATES:-5 10 15})
FRAMES="${FRAMES:-100}"
CFG="${CFG:-cfgs/cifar10_c/vbp_mnist_cnn.yaml}"
EXTRA_OPTS="${EXTRA_OPTS:-}"

echo "=========================================================================="
echo "          VBP Hyperparameter Sweep Benchmark                              "
echo " Config:            ${CFG}"
echo " Updates/Batch:     ${NUM_UPDATES[*]}"
echo " INIT_COV_SCALE:    ${INIT_COVS[*]}"
echo " OBS_COV_SCALE:     ${OBS_COVS[*]}"
echo " Seeds:             ${SEEDS[*]}"
echo " Test Frames/Run:   ${FRAMES} ($((FRAMES * 100)) samples per run)"
echo " Total Eval Runs:   $((${#NUM_UPDATES[@]} * ${#INIT_COVS[@]} * ${#OBS_COVS[@]} * ${#SEEDS[@]}))"
echo "=========================================================================="

TABLE_ENTRIES=()

for ac in "${NUM_UPDATES[@]}"; do
  for ic in "${INIT_COVS[@]}"; do
    for oc in "${OBS_COVS[@]}"; do
      echo ""
      echo ">>> Evaluating [UPDATES: ${ac} | INIT_COV: ${ic} | OBS_COV: ${oc}] across seeds (${SEEDS[*]})..."
      err_vals=()
      lat_vals=()

      for s in "${SEEDS[@]}"; do
        log_file="output/sweep_vbp_n${ac}_ic${ic}_oc${oc}_s${s}.log"
        echo -n "    [Seed ${s}] Running... "

        python test_time.py \
          --cfg "${CFG}" \
          RNG_SEED "${s}" \
          MIXED_GRADUAL.FRAMES "${FRAMES}" \
          VBP.NUM_ADAPT_SAMPLES "${ac}" \
          VBP.INIT_COV_SCALE "${ic}" \
          VBP.OBS_COV_SCALE "${oc}" \
          ${EXTRA_OPTS} \
          > "${log_file}" 2>&1

        err=$(grep -oP "mean error:\s*\K[0-9.]+(?=%)" "${log_file}" | tail -1 || true)
        lat=$(grep -oP "Adaptation latency:\s*\K[0-9.]+(?=\s*ms/sample)" "${log_file}" | tail -1 || true)

        if [ -n "$err" ]; then
          err_vals+=("$err")
          lat_vals+=("${lat:-0.0}")
          echo "Done -> Error: ${err}%, Latency: ${lat:-N/A} ms/sample"
        else
          echo "FAILED (Check ${log_file})"
        fi
      done

      # Compute numerical mean & std for sorting and display
      stats=$(python -c "
import numpy as np, sys
errs = [float(x) for x in sys.argv[1].split() if x]
lats = [float(x) for x in sys.argv[2].split() if x and float(x) > 0.0]
if errs:
    m_err = np.mean(errs)
    s_err = np.std(errs)
    err_str = f'{m_err:.2f}% ± {s_err:.2f}%'
else:
    m_err = 999.0
    err_str = 'FAILED'
lat_str = f'{np.mean(lats):.2f} ± {np.std(lats):.2f} ms/sample' if lats else 'N/A'
print(f'{m_err:.4f}|{err_str}|{lat_str}')
" "${err_vals[*]}" "${lat_vals[*]}")

      mean_num=$(echo "$stats" | cut -d'|' -f1)
      err_fmt=$(echo "$stats" | cut -d'|' -f2)
      lat_fmt=$(echo "$stats" | cut -d'|' -f3)

      TABLE_ENTRIES+=("${mean_num}|${ac}|${ic}|${oc}|${err_fmt}|${lat_fmt}")
    done
  done
done

echo ""
echo "===================================================================================================="
echo "                   Hyperparameter Sweep Results (Ranked by Lowest Error)                            "
echo "===================================================================================================="
printf "%-10s %-10s %-16s %-16s %-24s %-22s\n" "Rank" "Updates" "INIT_COV_SCALE" "OBS_COV_SCALE" "Error (Mean ± Std)" "Latency (Mean ± Std)"
echo "----------------------------------------------------------------------------------------------------"

rank=1
while IFS="|" read -r mean_val ac_val ic_val oc_val err_val lat_val; do
  [ -z "$mean_val" ] && continue
  tag=""
  if [ "$rank" -eq 1 ]; then
    tag=" (Best)"
  fi
  printf "%-10s %-10s %-16s %-16s %-24s %-22s\n" "#${rank}${tag}" "${ac_val}" "${ic_val}" "${oc_val}" "${err_val}" "${lat_val}"
  rank=$((rank + 1))
done < <(printf "%s\n" "${TABLE_ENTRIES[@]}" | sort -t'|' -k1,1n)
echo "===================================================================================================="
