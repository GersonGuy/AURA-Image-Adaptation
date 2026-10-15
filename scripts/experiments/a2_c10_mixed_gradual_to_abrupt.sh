#!/bin/bash -l
# =============================================================================
# A2-C10 — Mixed-gradual training -> abrupt-shift evaluation
#
# Dataset:
#   CIFAR-10-C
#
# Evaluation protocol:
#   - Abrupt corruption transitions evaluated via SETTING continual
#   - 200 images per evaluation batch (TEST.BATCH_SIZE 200)
#
# Supervised methods:
#   - AURA: 15 labeled images per update (AURA.NUM_ADAPT_SAMPLES 15)
#   - EKF-FC: 15 labeled images per update (EKF_FC.NUM_ADAPT_SAMPLES 15)
#
# Unsupervised batch-wise methods:
#   - ROID: all 200 unlabeled images (ROID.SUPERVISED False, ROID.ADAPT_SAMPLES -1)
#   - BN-Test: all 200 unlabeled images (MIXED_GRADUAL.NUM_UPDATES_PER_EPISODE -1)
#   - BN-EMA: all 200 unlabeled images (MIXED_GRADUAL.NUM_UPDATES_PER_EPISODE -1)
#   - BN-Alpha: all 200 unlabeled images (MIXED_GRADUAL.NUM_UPDATES_PER_EPISODE -1)
#
# AURA sweep:
#   latent dimensions: 128, 256, 500
#   INIT_COV_SCALE:    1e-4, 1e-3, 1e-2
#   OBS_COV_SCALE:     1e-4, 1e-3, 1e-2
#   seeds:             1, 2, 3
#
# Baselines:
#   EKF-FC, ROID, BN-Test, BN-EMA, BN-Alpha
#   3 seeds each
#
# Smoke test:
#   SMOKE=1 bash scripts/experiments/a2_c10_mixed_gradual_to_abrupt.sh
#
# Dry run:
#   DRY_RUN=1 bash scripts/experiments/a2_c10_mixed_gradual_to_abrupt.sh
# =============================================================================

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  source "$REPO_ROOT/.venv/bin/activate"
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate tta || true
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# -----------------------------------------------------------------------------
# General settings
# -----------------------------------------------------------------------------

DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

seeds=(${seeds:-1 2 3})

batch_size="${batch_size:-200}"
labeled_samples="${labeled_samples:-15}"
severity="${severity:-5}"

latent_dims=(${latent_dims:-128 256 500})
init_cov_scales=(${init_cov_scales:-1e-4 1e-3 1e-2})
obs_cov_scales=(${obs_cov_scales:-1e-4 1e-3 1e-2})

offline_num_episodes="${offline_num_episodes:-1500}"
offline_num_updates="${offline_num_updates:-3}"
offline_q_samples="${offline_q_samples:-10}"

# Abrupt sequence: 9 standard corruptions evaluated sequentially via SETTING continual
abrupt_corruptions=('zoom_blur' 'snow' 'frost' 'fog' 'brightness' 'contrast' 'elastic_transform' 'pixelate' 'jpeg_compression')
abrupt_corruptions_str='["zoom_blur","snow","frost","fog","brightness","contrast","elastic_transform","pixelate","jpeg_compression"]'

# -----------------------------------------------------------------------------
# Configurations
# -----------------------------------------------------------------------------

offline_cfg="${offline_cfg:-cfgs/cifar10_c/offline_for_a4.yaml}"
aura_cfg="${aura_cfg:-cfgs/cifar10_c/aura.yaml}"
ekf_cfg="${ekf_cfg:-cfgs/cifar10_c/ekf_fc.yaml}"
roid_cfg="${roid_cfg:-cfgs/cifar10_c/roid.yaml}"
bn_test_cfg="${bn_test_cfg:-cfgs/cifar10_c/norm_test.yaml}"
bn_ema_cfg="${bn_ema_cfg:-cfgs/cifar10_c/norm_ema.yaml}"
bn_alpha_cfg="${bn_alpha_cfg:-cfgs/cifar10_c/norm_alpha.yaml}"

# -----------------------------------------------------------------------------
# Smoke mode
# -----------------------------------------------------------------------------

if [[ "$SMOKE" = "1" ]]; then
  seeds=(1)
  latent_dims=(500)
  init_cov_scales=(1e-3)
  obs_cov_scales=(1e-3)

  # In smoke mode, run only the first 2 corruptions with a small num_ex
  abrupt_corruptions=('zoom_blur' 'snow')
  abrupt_corruptions_str='["zoom_blur","snow"]'
  smoke_num_ex="${smoke_num_ex:-400}"

  offline_num_episodes="${smoke_offline_episodes:-5}"
  offline_num_updates="${smoke_offline_updates:-1}"
  offline_q_samples="${smoke_q_samples:-2}"

  CONTINUE_ON_ERROR=0

  echo "================================================================="
  echo "[A2-C10] SMOKE mode enabled."
  echo "  Corruption sequence: ${abrupt_corruptions[*]}"
  echo "  Transition indices (samples): 0, 400"
  echo "  Evaluation batch size: $batch_size"
  echo "  Adaptation batch sizes:"
  echo "    - AURA:    $labeled_samples (labeled)"
  echo "    - EKF-FC:  $labeled_samples (labeled)"
  echo "    - ROID:    $batch_size (unsupervised)"
  echo "    - BN-Test: $batch_size (unsupervised)"
  echo "    - BN-EMA:  $batch_size (unsupervised)"
  echo "    - BN-Alpha: $batch_size (unsupervised)"
  echo "================================================================="
fi

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

timestamp="$(date +%Y%m%d_%H%M%S)"
run_root="${run_root:-logs/a2_c10_abrupt/${timestamp}}"

mkdir -p "$run_root/aura"
mkdir -p "$run_root/baselines"

manifest="$run_root/manifest.tsv"
summary_file="$run_root/final_summary.txt"

printf \
  "method\tlatent_dim\tinit_cov\tobs_cov\tseed\tsupervision\tadapt_samples\tstatus\tresult\tcheckpoint\tlog\n" \
  > "$manifest"

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

run_logged() {
  local log_file="$1"
  shift

  echo "+ $*" | tee -a "$log_file"

  if [[ "$DRY_RUN" = "1" ]]; then
    return 0
  fi

  set +e
  "$@" 2>&1 | tee -a "$log_file"
  local rc=${PIPESTATUS[0]}
  set -e

  return "$rc"
}

extract_result() {
  local log_file="$1"

  if [[ ! -f "$log_file" ]]; then
    echo "no log"
    return
  fi

  local result

  result="$(
    grep -Ei "mean error:" "$log_file" 2>/dev/null \
      | tail -1 \
      | sed 's/^[[:space:]]*//' \
      || true
  )"

  if [[ -z "$result" ]]; then
    result="$(
      grep -Ei "error %|average adaptation time" "$log_file" 2>/dev/null \
        | tail -1 \
        | sed 's/^[[:space:]]*//' \
        || true
    )"
  fi

  if [[ -z "$result" ]]; then
    result="result not found in log"
  fi

  result="${result//$'\t'/ }"
  echo "$result"
}

record_result() {
  local method="$1"
  local latent_dim="$2"
  local init_cov="$3"
  local obs_cov="$4"
  local seed="$5"
  local supervision="$6"
  local adapt_samples="$7"
  local status="$8"
  local result="$9"
  local checkpoint="${10}"
  local log_file="${11}"

  printf \
    "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$method" \
    "$latent_dim" \
    "$init_cov" \
    "$obs_cov" \
    "$seed" \
    "$supervision" \
    "$adapt_samples" \
    "$status" \
    "$result" \
    "$checkpoint" \
    "$log_file" \
    >> "$manifest"
}

find_new_checkpoint() {
  local marker="$1"

  find ckpt/offline_aura_fc -type f \
    \( -name "best_params.pth" -o -name "last_params.pth" \) \
    -newer "$marker" \
    -printf '%T@|%p\n' 2>/dev/null \
    | sort -nr \
    | head -1 \
    | cut -d'|' -f2- \
    || true
}

fail_or_continue() {
  local message="$1"

  echo "$message" >&2

  if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
    exit 1
  fi
}

# -----------------------------------------------------------------------------
# Pre-flight
# -----------------------------------------------------------------------------

required_configs=(
  "$offline_cfg"
  "$aura_cfg"
  "$ekf_cfg"
  "$roid_cfg"
  "$bn_test_cfg"
  "$bn_ema_cfg"
  "$bn_alpha_cfg"
)

for cfg_file in "${required_configs[@]}"; do
  if [[ ! -f "$cfg_file" ]]; then
    echo "[A2-C10] Missing configuration: $cfg_file" >&2
    exit 1
  fi
done

echo "================================================================="
echo "[A2-C10] Mixed-gradual training -> abrupt evaluation"
echo "================================================================="
echo "Images per evaluation batch: $batch_size"
echo "AURA/EKF labeled images:     $labeled_samples per batch"
echo "ROID/BN unlabeled images:    $batch_size per batch"
echo "Latent dimensions:           ${latent_dims[*]}"
echo "INIT covariance:             ${init_cov_scales[*]}"
echo "OBS covariance:              ${obs_cov_scales[*]}"
echo "Seeds:                       ${seeds[*]}"
echo "Corruption sequence:         ${abrupt_corruptions[*]}"
echo "Output:                      $run_root"
echo "================================================================="

{
  echo "timestamp=$timestamp"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "git_branch=$(git branch --show-current 2>/dev/null || echo unknown)"
  echo "python=$(command -v python || true)"
  echo "batch_size=$batch_size"
  echo "labeled_samples=$labeled_samples"
  echo "severity=$severity"
  echo "latent_dims=${latent_dims[*]}"
  echo "init_cov_scales=${init_cov_scales[*]}"
  echo "obs_cov_scales=${obs_cov_scales[*]}"
  echo "seeds=${seeds[*]}"
  echo "corruptions=${abrupt_corruptions[*]}"
} > "$run_root/run_metadata.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$run_root/nvidia_smi.txt" 2>&1 || true
fi

# =============================================================================
# PART 1 — AURA
# Train on mixed-gradual, evaluate on abrupt.
# =============================================================================

for latent_dim in "${latent_dims[@]}"; do
  aura_dim_dir="$run_root/aura/latent_${latent_dim}"
  mkdir -p "$aura_dim_dir"

  offline_log="$aura_dim_dir/offline.log"
  marker="$aura_dim_dir/offline_start.marker"

  touch "$marker"

  echo
  echo "================================================================="
  echo "[A2-C10] AURA offline training: latent_dim=$latent_dim"
  echo "================================================================="

  offline_cmd=(
    python offline_train_aura.py
    --cfg "$offline_cfg"

    RNG_SEED 1
    SETTING gradual_mixed_corruption
    AURA.LATENT_DIM "$latent_dim"

    OFFLINE.NUM_EPISODES "$offline_num_episodes"
    OFFLINE.NUM_UPDATES_PER_EPISODE "$offline_num_updates"
    OFFLINE.Q_SAMPLES "$offline_q_samples"
  )

  if ! run_logged "$offline_log" "${offline_cmd[@]}"; then
    record_result \
      "AURA" \
      "$latent_dim" \
      "-" \
      "-" \
      "1" \
      "offline" \
      "-" \
      "FAILED" \
      "$(extract_result "$offline_log")" \
      "" \
      "$offline_log"

    fail_or_continue \
      "[A2-C10] AURA offline training failed for latent_dim=$latent_dim"

    continue
  fi

  if [[ "$DRY_RUN" = "1" ]]; then
    checkpoint="<DRY_RUN_AURA_D${latent_dim}>"
  else
    checkpoint="$(find_new_checkpoint "$marker")"

    if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
      record_result \
        "AURA" \
        "$latent_dim" \
        "-" \
        "-" \
        "1" \
        "offline" \
        "-" \
        "NO_CHECKPOINT" \
        "checkpoint not found" \
        "" \
        "$offline_log"

      fail_or_continue \
        "[A2-C10] No new AURA checkpoint for latent_dim=$latent_dim"

      continue
    fi
  fi

  echo "$checkpoint" > "$aura_dim_dir/checkpoint_path.txt"

  record_result \
    "AURA" \
    "$latent_dim" \
    "-" \
    "-" \
    "1" \
    "offline" \
    "-" \
    "OK" \
    "checkpoint created" \
    "$checkpoint" \
    "$offline_log"

  # ---------------------------------------------------------------------------
  # AURA abrupt covariance sweep
  # ---------------------------------------------------------------------------

  for init_cov in "${init_cov_scales[@]}"; do
    for obs_cov in "${obs_cov_scales[@]}"; do
      for seed in "${seeds[@]}"; do

        online_log="$aura_dim_dir/init${init_cov}_obs${obs_cov}_seed${seed}.log"

        echo
        echo "-------------------------------------------------------------"
        echo "[A2-C10] AURA abrupt evaluation"
        echo "latent_dim=$latent_dim"
        echo "seed=$seed"
        echo "init_cov=$init_cov"
        echo "obs_cov=$obs_cov"
        echo "labeled=$labeled_samples / batch=$batch_size"
        echo "-------------------------------------------------------------"

        online_cmd=(
          python test_time.py
          --cfg "$aura_cfg"

          RNG_SEED "$seed"
          SETTING continual

          CORRUPTION.TYPE "$abrupt_corruptions_str"
          CORRUPTION.SEVERITY "[$severity]"

          AURA.LATENT_DIM "$latent_dim"
          AURA.OFFLINE_CKPT "$checkpoint"
          AURA.INIT_COV_SCALE "$init_cov"
          AURA.OBS_COV_SCALE "$obs_cov"
          AURA.NUM_ADAPT_SAMPLES "$labeled_samples"

          TEST.BATCH_SIZE "$batch_size"
          TIMING.ENABLED True
        )

        if [[ "$SMOKE" = "1" ]]; then
          online_cmd+=(CORRUPTION.NUM_EX "$smoke_num_ex")
        fi

        if run_logged "$online_log" "${online_cmd[@]}"; then
          status="OK"
        else
          status="FAILED"

          fail_or_continue \
            "[A2-C10] AURA failed: latent=$latent_dim init=$init_cov obs=$obs_cov seed=$seed"
        fi

        record_result \
          "AURA" \
          "$latent_dim" \
          "$init_cov" \
          "$obs_cov" \
          "$seed" \
          "supervised" \
          "$labeled_samples" \
          "$status" \
          "$(extract_result "$online_log")" \
          "$checkpoint" \
          "$online_log"
      done
    done
  done
done

# =============================================================================
# PART 2 — Baselines on the same abrupt stream
# =============================================================================

run_baseline() {
  local method="$1"
  local cfg_file="$2"
  local supervision="$3"
  local adapt_samples="$4"

  shift 4
  local extra_args=("$@")

  local method_dir="$run_root/baselines/$method"
  mkdir -p "$method_dir"

  for seed in "${seeds[@]}"; do
    local log_file="$method_dir/seed${seed}.log"

    echo
    echo "-------------------------------------------------------------"
    echo "[A2-C10] Baseline: $method"
    echo "seed=$seed"
    echo "supervision=$supervision"
    echo "adapt_samples=$adapt_samples"
    echo "batch_size=$batch_size"
    echo "-------------------------------------------------------------"

    baseline_cmd=(
      python test_time.py
      --cfg "$cfg_file"

      RNG_SEED "$seed"
      SETTING continual

      CORRUPTION.TYPE "$abrupt_corruptions_str"
      CORRUPTION.SEVERITY "[$severity]"

      TEST.BATCH_SIZE "$batch_size"
      TIMING.ENABLED True
    )

    if [[ "$SMOKE" = "1" ]]; then
      baseline_cmd+=(CORRUPTION.NUM_EX "$smoke_num_ex")
    fi

    baseline_cmd+=("${extra_args[@]}")

    if run_logged "$log_file" "${baseline_cmd[@]}"; then
      status="OK"
    else
      status="FAILED"

      fail_or_continue \
        "[A2-C10] Baseline failed: method=$method seed=$seed"
    fi

    record_result \
      "$method" \
      "-" \
      "-" \
      "-" \
      "$seed" \
      "$supervision" \
      "$adapt_samples" \
      "$status" \
      "$(extract_result "$log_file")" \
      "-" \
      "$log_file"
  done
}

# EKF-FC — supervised, same 15 labeled images as AURA.
run_baseline \
  "EKF_FC" \
  "$ekf_cfg" \
  "supervised" \
  "$labeled_samples" \
  EKF_FC.NUM_ADAPT_SAMPLES "$labeled_samples"

# ROID — original unsupervised behavior, all 200 images.
run_baseline \
  "ROID" \
  "$roid_cfg" \
  "unsupervised" \
  "$batch_size" \
  ROID.SUPERVISED False \
  ROID.ADAPT_SAMPLES -1

# BatchNorm methods — unsupervised, use the full 200-image batch.
run_baseline \
  "BN_TEST" \
  "$bn_test_cfg" \
  "unsupervised" \
  "$batch_size" \
  MIXED_GRADUAL.NUM_UPDATES_PER_EPISODE -1

run_baseline \
  "BN_EMA" \
  "$bn_ema_cfg" \
  "unsupervised" \
  "$batch_size" \
  MIXED_GRADUAL.NUM_UPDATES_PER_EPISODE -1

run_baseline \
  "BN_ALPHA" \
  "$bn_alpha_cfg" \
  "unsupervised" \
  "$batch_size" \
  MIXED_GRADUAL.NUM_UPDATES_PER_EPISODE -1

# =============================================================================
# Final summary
# =============================================================================

{
  echo "================================================================="
  echo "A2-C10 ABRUPT-SHIFT FINAL SUMMARY"
  echo "Finished: $(date)"
  echo "================================================================="
  echo
  echo "Protocol:"
  echo "  Images per batch:             $batch_size"
  echo "  AURA/EKF labels per batch:    $labeled_samples"
  echo "  ROID/BN samples per batch:    $batch_size unlabeled"
  echo "  Corruptions sequence:         ${abrupt_corruptions[*]}"
  echo

  echo "Status counts:"
  awk -F '\t' '
    NR > 1 {
      count[$1 ":" $8]++
    }
    END {
      for (key in count) {
        print "  " key " = " count[key]
      }
    }
  ' "$manifest" | sort

  echo
  echo "Results:"
  echo

  if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "$manifest"
  else
    cat "$manifest"
  fi

  echo
  echo "Manifest:"
  echo "  $manifest"
  echo
  echo "Logs:"
  echo "  $run_root"
} | tee "$summary_file"

echo
echo "[A2-C10] Experiment completed."
echo "[A2-C10] Summary:  $summary_file"
echo "[A2-C10] Manifest: $manifest"
echo "[A2-C10] Logs:     $run_root"
