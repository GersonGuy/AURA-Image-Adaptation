#!/bin/bash -l
# =============================================================================
# A2-C10 — Nonlinear AURA: compare offline training on
#   (1) mixed-gradual trajectories
#   (2) abrupt/continual trajectories
#
# Both checkpoints are evaluated online on the SAME abrupt stream.
#
# Smoke:
#   SMOKE=1 bash scripts/experiments/a2_c10_abrupt_aura_nonlinear.sh
#
# Dry run:
#   DRY_RUN=1 bash scripts/experiments/a2_c10_abrupt_aura_nonlinear.sh
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

# Stop immediately when a run fails.
# This prevents the script from printing "completed" after failed training.
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

offline_seed="${offline_seed:-1}"
seeds=(${seeds:-1 2 3})

latent_dims=(${latent_dims:-1000 500})

batch_size="${batch_size:-200}"
labeled_samples="${labeled_samples:-15}"
severity="${severity:-5}"

# Expanded online covariance sweep.
init_cov_scales=(${init_cov_scales:-1e-5 1e-4 1e-3 1e-2 1e-1})
obs_cov_scales=(${obs_cov_scales:-1e-5 1e-4 1e-3 1e-2 1e-1})

# Offline training settings shared by both training distributions.
offline_num_episodes="${offline_num_episodes:-1500}"
offline_num_updates="${offline_num_updates:-10}"
offline_q_samples="${offline_q_samples:-10}"
offline_lr="${offline_lr:-0.0003}"

# Mixed-gradual offline settings.
mixed_frames="${mixed_frames:-1000}"
mixed_schedule_type="${mixed_schedule_type:-random}"

# Existing configs. No new YAML files are created.
offline_cfg="${offline_cfg:-cfgs/cifar10_c/offline_aura_nonlinear.yaml}"
aura_cfg="${aura_cfg:-cfgs/cifar10_c/aura_nonlinear.yaml}"

# -----------------------------------------------------------------------------
# Abrupt corruption sequence
#
# This exact sequence is used:
#   1. for abrupt Offline training
#   2. for every Online evaluation
# -----------------------------------------------------------------------------

abrupt_corruptions=(
  zoom_blur
  snow
  frost
  fog
  brightness
  contrast
  elastic_transform
  pixelate
  jpeg_compression
)

abrupt_corruptions_str='["zoom_blur","snow","frost","fog","brightness","contrast","elastic_transform","pixelate","jpeg_compression"]'

# Offline distributions.
# Format:
#   name | SETTING
training_specs=(
  "mixed|gradual_mixed_corruption"
  "abrupt|continual"
)

# -----------------------------------------------------------------------------
# Smoke mode
# -----------------------------------------------------------------------------

if [[ "$SMOKE" = "1" ]]; then
  seeds=(1)
  latent_dims=(500)

  init_cov_scales=(1e-3)
  obs_cov_scales=(1e-3)

  abrupt_corruptions=(
    zoom_blur
    snow
  )

  abrupt_corruptions_str='["zoom_blur","snow"]'

  smoke_num_ex="${smoke_num_ex:-400}"

  offline_num_episodes="${smoke_offline_episodes:-5}"
  offline_num_updates="${smoke_offline_updates:-1}"
  offline_q_samples="${smoke_q_samples:-2}"
  mixed_frames="${smoke_mixed_frames:-100}"

  CONTINUE_ON_ERROR=0

  echo "================================================================="
  echo "[A2-C10] SMOKE mode"
  echo "[A2-C10] Offline trainings: mixed + abrupt"
  echo "[A2-C10] Online evaluations: abrupt for both checkpoints"
  echo "[A2-C10] Corruptions: ${abrupt_corruptions[*]}"
  echo "================================================================="
fi

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

timestamp="$(date +%Y%m%d_%H%M%S)"

run_root="${run_root:-logs/a2_c10_nonlinear_train_mixed_vs_abrupt/${timestamp}}"

mkdir -p "$run_root"

manifest="$run_root/manifest.tsv"
summary_file="$run_root/final_summary.txt"

printf \
  "train_distribution\tphase\tlatent_dim\tinit_cov\tobs_cov\tseed\tstatus\tcheckpoint\tlog\n" \
  > "$manifest"

# -----------------------------------------------------------------------------
# Helpers
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

find_new_checkpoint() {
  local marker="$1"

  find ckpt/offline_aura_fc \
    -type f \
    \( \
      -name "best_params.pth" \
      -o -name "last_params.pth" \
    \) \
    -newer "$marker" \
    -printf '%T@|%p\n' \
    2>/dev/null \
    | sort -nr \
    | head -1 \
    | cut -d'|' -f2- \
    || true
}

record_result() {
  printf \
    "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$@" \
    >> "$manifest"
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

for cfg in "$offline_cfg" "$aura_cfg"; do
  if [[ ! -f "$cfg" ]]; then
    echo "[A2-C10] Missing config: $cfg" >&2
    exit 1
  fi
done

if [[ ! -f "offline_train_aura.py" ]]; then
  echo "[A2-C10] Missing offline_train_aura.py" >&2
  exit 1
fi

if [[ ! -f "test_time.py" ]]; then
  echo "[A2-C10] Missing test_time.py" >&2
  exit 1
fi

# Catch the old YAML key that previously caused:
# KeyError: Non-existent config key: AURA.LIFTING_TYPE
if grep -Eq \
  '^[[:space:]]*LIFTING_TYPE:' \
  "$offline_cfg" \
  "$aura_cfg"; then

  echo "[A2-C10] Old YAML key LIFTING_TYPE was found." >&2
  echo "[A2-C10] Pull the fixed commit or replace it with:" >&2
  echo "[A2-C10] AURA.LIFTING.TYPE" >&2
  exit 2
fi

num_offline=$(( ${#training_specs[@]} * ${#latent_dims[@]} ))

num_online=$(( \
  num_offline \
  * ${#init_cov_scales[@]} \
  * ${#obs_cov_scales[@]} \
  * ${#seeds[@]} \
))

echo "================================================================="
echo "[A2-C10] Nonlinear train-mixed vs train-abrupt"
echo "[A2-C10] Online evaluation is always abrupt"
echo "================================================================="
echo "Offline trainings:  $num_offline"
echo "Online evaluations: $num_online"
echo "Latent dimensions:  ${latent_dims[*]}"
echo "INIT covariance:    ${init_cov_scales[*]}"
echo "OBS covariance:     ${obs_cov_scales[*]}"
echo "Online seeds:       ${seeds[*]}"
echo "Corruptions:        ${abrupt_corruptions[*]}"
echo "Output:             $run_root"
echo "================================================================="

{
  echo "timestamp=$timestamp"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "git_branch=$(git branch --show-current 2>/dev/null || echo unknown)"
  echo "python=$(command -v python || true)"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"

  echo "offline_cfg=$offline_cfg"
  echo "online_cfg=$aura_cfg"

  echo "offline_train_distributions=mixed abrupt"
  echo "online_distribution=abrupt"

  echo "offline_seed=$offline_seed"
  echo "online_seeds=${seeds[*]}"

  echo "latent_dims=${latent_dims[*]}"
  echo "init_cov_scales=${init_cov_scales[*]}"
  echo "obs_cov_scales=${obs_cov_scales[*]}"

  echo "offline_num_episodes=$offline_num_episodes"
  echo "offline_num_updates=$offline_num_updates"
  echo "offline_q_samples=$offline_q_samples"
  echo "offline_lr=$offline_lr"

  echo "batch_size=$batch_size"
  echo "labeled_samples=$labeled_samples"
  echo "severity=$severity"

  echo "abrupt_corruptions=${abrupt_corruptions[*]}"
} > "$run_root/run_metadata.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$run_root/nvidia_smi.txt" 2>&1 || true
fi

# =============================================================================
# Main experiment
# =============================================================================

for spec in "${training_specs[@]}"; do
  IFS='|' read -r train_distribution offline_setting <<< "$spec"

  for latent_dim in "${latent_dims[@]}"; do
    exp_dir="$run_root/train_${train_distribution}/latent_${latent_dim}"

    mkdir -p "$exp_dir"

    offline_log="$exp_dir/offline.log"
    marker="$exp_dir/offline_start.marker"

    touch "$marker"

    echo
    echo "================================================================="
    echo "[A2-C10] OFFLINE TRAINING"
    echo "train_distribution=$train_distribution"
    echo "setting=$offline_setting"
    echo "latent_dim=$latent_dim"
    echo "mapping=nonlinear residual"
    echo "================================================================="

    # Common arguments for both Offline trainings.
    #
    # Both mappings are explicitly nonlinear.
    offline_cmd=(
      python offline_train_aura.py
      --cfg "$offline_cfg"

      RNG_SEED "$offline_seed"

      AURA.LATENT_DIM "$latent_dim"

      AURA.LIFTING.TYPE nonlinear
      AURA.LIFTING.RESIDUAL_LINEAR True
      AURA.LIFTING.HIDDEN_DIMS "[$latent_dim]"
      AURA.LIFTING.ACTIVATION gelu
      AURA.LIFTING.NORMALIZATION none
      AURA.LIFTING.OUTPUT_SCALE 1.0
      AURA.LIFTING.ZERO_INIT_OUTPUT True

      OFFLINE.NUM_EPISODES "$offline_num_episodes"
      OFFLINE.NUM_UPDATES_PER_EPISODE "$offline_num_updates"
      OFFLINE.Q_SAMPLES "$offline_q_samples"
      OFFLINE.OPTIMIZER Adam
      OFFLINE.LR "$offline_lr"
    )

    if [[ "$train_distribution" = "mixed" ]]; then
      # -----------------------------------------------------------
      # Offline checkpoint A:
      # Learn the nonlinear mapping on mixed-gradual trajectories.
      # -----------------------------------------------------------

      offline_cmd+=(
        SETTING gradual_mixed_corruption

        MIXED_GRADUAL.FRAMES "$mixed_frames"
        MIXED_GRADUAL.SEVERITY "$severity"
        MIXED_GRADUAL.SCHEDULE_TYPE "$mixed_schedule_type"
      )
    else
      # -----------------------------------------------------------
      # Offline checkpoint B:
      # Learn the nonlinear mapping on abrupt transitions.
      # -----------------------------------------------------------

      offline_cmd+=(
        SETTING continual

        CORRUPTION.TYPE "$abrupt_corruptions_str"
        CORRUPTION.SEVERITY "[$severity]"

        TEST.BATCH_SIZE "$batch_size"
        AURA.NUM_ADAPT_SAMPLES "$labeled_samples"
      )

      if [[ "$SMOKE" = "1" ]]; then
        offline_cmd+=(
          CORRUPTION.NUM_EX "$smoke_num_ex"
        )
      fi
    fi

    if ! run_logged "$offline_log" "${offline_cmd[@]}"; then
      record_result \
        "$train_distribution" \
        "offline" \
        "$latent_dim" \
        "-" \
        "-" \
        "$offline_seed" \
        "FAILED" \
        "-" \
        "$offline_log"

      fail_or_continue \
        "[A2-C10] Offline failed: train=$train_distribution latent=$latent_dim"

      continue
    fi

    if [[ "$DRY_RUN" = "1" ]]; then
      checkpoint="<DRY_RUN_${train_distribution}_D${latent_dim}>"
    else
      checkpoint="$(find_new_checkpoint "$marker")"

      if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
        record_result \
          "$train_distribution" \
          "offline" \
          "$latent_dim" \
          "-" \
          "-" \
          "$offline_seed" \
          "NO_CHECKPOINT" \
          "-" \
          "$offline_log"

        fail_or_continue \
          "[A2-C10] No checkpoint: train=$train_distribution latent=$latent_dim"

        continue
      fi
    fi

    echo "$checkpoint" > "$exp_dir/checkpoint_path.txt"

    record_result \
      "$train_distribution" \
      "offline" \
      "$latent_dim" \
      "-" \
      "-" \
      "$offline_seed" \
      "OK" \
      "$checkpoint" \
      "$offline_log"

    echo "[A2-C10] Checkpoint: $checkpoint"

    # =========================================================================
    # Online evaluation
    #
    # This part is identical for both Offline checkpoints.
    # Online is ALWAYS abrupt.
    # =========================================================================

    for init_cov in "${init_cov_scales[@]}"; do
      for obs_cov in "${obs_cov_scales[@]}"; do
        for seed in "${seeds[@]}"; do
          online_log="$exp_dir/online_abrupt_init${init_cov}_obs${obs_cov}_seed${seed}.log"

          echo
          echo "-------------------------------------------------------------"
          echo "[A2-C10] ONLINE ABRUPT EVALUATION"
          echo "trained_on=$train_distribution"
          echo "latent_dim=$latent_dim"
          echo "init_cov=$init_cov"
          echo "obs_cov=$obs_cov"
          echo "seed=$seed"
          echo "-------------------------------------------------------------"

          online_cmd=(
            python test_time.py
            --cfg "$aura_cfg"

            RNG_SEED "$seed"

            # Online is always abrupt.
            SETTING continual

            CORRUPTION.TYPE "$abrupt_corruptions_str"
            CORRUPTION.SEVERITY "[$severity]"

            # The online mapping must match the Offline checkpoint.
            AURA.LATENT_DIM "$latent_dim"

            AURA.LIFTING.TYPE nonlinear
            AURA.LIFTING.RESIDUAL_LINEAR True
            AURA.LIFTING.HIDDEN_DIMS "[$latent_dim]"
            AURA.LIFTING.ACTIVATION gelu
            AURA.LIFTING.NORMALIZATION none
            AURA.LIFTING.OUTPUT_SCALE 1.0
            AURA.LIFTING.ZERO_INIT_OUTPUT True

            AURA.OFFLINE_CKPT "$checkpoint"

            AURA.INIT_COV_SCALE "$init_cov"
            AURA.OBS_COV_SCALE "$obs_cov"
            AURA.NUM_ADAPT_SAMPLES "$labeled_samples"

            TEST.BATCH_SIZE "$batch_size"

            TIMING.ENABLED True
          )

          if [[ "$SMOKE" = "1" ]]; then
            online_cmd+=(
              CORRUPTION.NUM_EX "$smoke_num_ex"
            )
          fi

          if run_logged "$online_log" "${online_cmd[@]}"; then
            status="OK"
          else
            status="FAILED"

            fail_or_continue \
              "[A2-C10] Online failed: trained_on=$train_distribution latent=$latent_dim init=$init_cov obs=$obs_cov seed=$seed"
          fi

          record_result \
            "$train_distribution" \
            "online_abrupt" \
            "$latent_dim" \
            "$init_cov" \
            "$obs_cov" \
            "$seed" \
            "$status" \
            "$checkpoint" \
            "$online_log"
        done
      done
    done
  done
done

# -----------------------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------------------

echo
echo "================================================================="
echo "[A2-C10] Experiment completed"
echo "Manifest: $manifest"
echo "Logs:     $run_root"
echo "================================================================="

if command -v column >/dev/null 2>&1; then
  column \
    -t \
    -s $'\t' \
    "$manifest" \
    | tee "$summary_file"
else
  cat "$manifest" | tee "$summary_file"
fi