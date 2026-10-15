#!/bin/bash -l
# =============================================================================
# AURA nonlinear lifting capacity + LEARNING RATE + ONLINE covariance sweep
# (Optimized 5-Hour Run Plan for Latent Dim = 1000)
#
# Dataset:
#   CIFAR-100-C
#
# Stream:
#   gradual_mixed_corruption
#   random schedule
#   severity 5
#   1500 online frames
#
# Latent Dimension:
#   1000
#
# Offline pretraining LR sweep:
#   LR = {0.0001, 0.0003}
#   GELU activation
#   Adam optimizer
#
# Architecture:
#   gelu_1x500|[500]
#
# Online covariance grid (2 seeds per cell):
#   INIT_COV_SCALE = {1e-4, 1e-3}
#   OBS_COV_SCALE  = {1e-4, 1e-3, 1e-2}
#   seeds          = {1, 2}
#
# Post-Sweep Evaluation:
#   Evaluates best LR + covariance setup under 5, 10, 15 labeled samples
#
# Estimated Total Time: ~4 hours 25 minutes
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
RUN_TESTS="${RUN_TESTS:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

MIN_FREE_GB="${MIN_FREE_GB:-10}"

offline_seed="${offline_seed:-1}"
seeds=(${seeds:-1 2})

# Evaluation labeled adaptation samples and seeds
eval_adapt_samples=(${eval_adapt_samples:-5 10 15})
eval_seeds=(${eval_seeds:-1 2})

latent_dim="${latent_dim:-1000}"
learning_rates=(${learning_rates:-0.01 0.001})

episodes="${episodes:-1000}"
updates_per_episode="${updates_per_episode:-10}"
q_samples="${q_samples:-5}"

frames="${frames:-1500}"
severity="${severity:-5}"
schedule_type="${schedule_type:-random}"

offline_cfg="${offline_cfg:-cfgs/cifar100_c/offline_aura_nonlinear.yaml}"
online_cfg="${online_cfg:-cfgs/cifar100_c/aura_nonlinear.yaml}"

# -----------------------------------------------------------------------------
# Architecture specification (1 architecture for 5-hour target)
# -----------------------------------------------------------------------------

architectures=(
  "gelu_1x500|[650]"
)

# -----------------------------------------------------------------------------
# ONLINE covariance sweep (Optimized grid)
# -----------------------------------------------------------------------------

init_cov_scales=(${init_cov_scales:-1e-4 1e-3})
obs_cov_scales=(${obs_cov_scales:-1e-4 1e-3 1e-2})

# -----------------------------------------------------------------------------
# Short smoke mode
# -----------------------------------------------------------------------------

if [[ "$SMOKE" = "1" ]]; then
  architectures=(
    "gelu_1x500|[650]"
  )

  learning_rates=(0.0003)
  seeds=(1)
  eval_seeds=(1 2)
  init_cov_scales=(1e-3)
  obs_cov_scales=(1e-3)

  episodes="${smoke_episodes:-5}"
  updates_per_episode="${smoke_updates:-2}"
  q_samples="${smoke_q_samples:-2}"
  frames="${smoke_frames:-100}"

  CONTINUE_ON_ERROR=0

  echo "[NONLINEAR] SMOKE mode enabled."
fi

# -----------------------------------------------------------------------------
# Output layout
# -----------------------------------------------------------------------------

timestamp="$(date +%Y%m%d_%H%M%S)"
run_root="${run_root:-logs/cifar100_nonlinear_capacity_covariance/${timestamp}}"

mkdir -p "$run_root"

manifest="$run_root/manifest.tsv"
summary_file="$run_root/final_summary.txt"

printf \
  "architecture\tlr\tphase\tseed\tinit_cov\tobs_cov\tstatus\tcheckpoint\tlog\n" \
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

record_result() {
  local architecture="$1"
  local lr="$2"
  local phase="$3"
  local seed="$4"
  local init_cov="$5"
  local obs_cov="$6"
  local status="$7"
  local checkpoint="$8"
  local log_file="$9"

  printf \
    "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$architecture" \
    "$lr" \
    "$phase" \
    "$seed" \
    "$init_cov" \
    "$obs_cov" \
    "$status" \
    "$checkpoint" \
    "$log_file" \
    >> "$manifest"
}

find_new_checkpoint() {
  local marker="$1"

  find ckpt -type f \
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

check_disk_space() {
  local free_kb
  local required_kb

  free_kb="$(
    df -Pk "$REPO_ROOT" \
      | awk 'NR == 2 {print $4}'
  )"

  required_kb=$((MIN_FREE_GB * 1024 * 1024))

  if [[ -z "$free_kb" ]]; then
    echo "[NONLINEAR] WARNING: could not determine free disk space."
    return 0
  fi

  echo "[NONLINEAR] Free disk space: $((free_kb / 1024 / 1024)) GB"

  if (( free_kb < required_kb )); then
    echo "[NONLINEAR] ERROR: less than ${MIN_FREE_GB} GB are free." >&2
    echo "[NONLINEAR] Stopping before creating another checkpoint." >&2
    exit 4
  fi
}

# -----------------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------------

for cfg in "$offline_cfg" "$online_cfg"; do
  if [[ ! -f "$cfg" ]]; then
    echo "[NONLINEAR] Missing configuration: $cfg" >&2
    exit 1
  fi
done

if [[ ! -f "offline_train_aura.py" ]]; then
  echo "[NONLINEAR] Missing offline_train_aura.py" >&2
  exit 1
fi

if [[ ! -f "test_time.py" ]]; then
  echo "[NONLINEAR] Missing test_time.py" >&2
  exit 1
fi

num_architectures=${#architectures[@]}
num_lrs=${#learning_rates[@]}
num_cov_cells=$(( ${#init_cov_scales[@]} * ${#obs_cov_scales[@]} ))
num_offline_runs=$(( num_architectures * num_lrs ))
num_online_runs=$(( num_offline_runs * num_cov_cells * ${#seeds[@]} ))

echo "================================================================="
echo "[NONLINEAR] Repository: $REPO_ROOT"
echo "[NONLINEAR] Output: $run_root"
echo "[NONLINEAR] Latent dimension: $latent_dim"
echo "[NONLINEAR] Activation: GELU"
echo "[NONLINEAR] Optimizer: Adam"
echo "[NONLINEAR] Learning rates: ${learning_rates[*]}"
echo "[NONLINEAR] Architectures: $num_architectures"
echo "[NONLINEAR] INIT covariance: ${init_cov_scales[*]}"
echo "[NONLINEAR] OBS covariance: ${obs_cov_scales[*]}"
echo "[NONLINEAR] Online seeds: ${seeds[*]}"
echo "[NONLINEAR] Offline trainings: $num_offline_runs"
echo "[NONLINEAR] Online evaluations: $num_online_runs"
echo "================================================================="

{
  echo "timestamp=$timestamp"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "git_branch=$(git branch --show-current 2>/dev/null || echo unknown)"
  echo "python=$(command -v python || true)"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "offline_cfg=$offline_cfg"
  echo "online_cfg=$online_cfg"
  echo "latent_dim=$latent_dim"
  echo "learning_rates=${learning_rates[*]}"
  echo "episodes=$episodes"
  echo "updates_per_episode=$updates_per_episode"
  echo "q_samples=$q_samples"
  echo "frames=$frames"
  echo "severity=$severity"
  echo "schedule_type=$schedule_type"
  echo "offline_seed=$offline_seed"
  echo "online_seeds=${seeds[*]}"
  echo "init_cov_scales=${init_cov_scales[*]}"
  echo "obs_cov_scales=${obs_cov_scales[*]}"
  echo "architectures=${architectures[*]}"
} > "$run_root/run_metadata.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$run_root/nvidia_smi.txt" 2>&1 || true
fi

check_disk_space

# -----------------------------------------------------------------------------
# Unit tests
# -----------------------------------------------------------------------------

if [[ "$RUN_TESTS" = "1" ]]; then
  tests_log="$run_root/tests.log"

  if [[ -f "tests/test_nonlinear_lifting.py" ]]; then
    if ! run_logged \
        "$tests_log" \
        env PYTHONPATH=. python tests/test_nonlinear_lifting.py; then

      echo "[NONLINEAR] Nonlinear lifting tests failed." >&2
      exit 2
    fi
  else
    echo \
      "[NONLINEAR] WARNING: tests/test_nonlinear_lifting.py was not found." \
      | tee "$tests_log"
  fi
fi

# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------

for spec in "${architectures[@]}"; do
  IFS='|' read -r architecture hidden_dims <<< "$spec"

  for lr in "${learning_rates[@]}"; do

    check_disk_space

    run_key="${architecture}_lr${lr}"
    architecture_dir="$run_root/$run_key"
    mkdir -p "$architecture_dir"

    offline_log="$architecture_dir/offline.log"
    marker="$architecture_dir/offline_start.marker"

    touch "$marker"

    echo
    echo "================================================================="
    echo "[NONLINEAR] Offline architecture: $architecture | LR: $lr"
    echo "[NONLINEAR] Hidden dimensions: $hidden_dims"
    echo "================================================================="

    offline_cmd=(
      python offline_train_aura.py
      --cfg "$offline_cfg"

      RNG_SEED "$offline_seed"
      SETTING gradual_mixed_corruption

      AURA.LATENT_DIM "$latent_dim"

      AURA.LIFTING.TYPE nonlinear
      AURA.LIFTING.RESIDUAL_LINEAR True
      AURA.LIFTING.HIDDEN_DIMS "$hidden_dims"
      AURA.LIFTING.ACTIVATION gelu
      AURA.LIFTING.NORMALIZATION none
      AURA.LIFTING.OUTPUT_SCALE 1.0
      AURA.LIFTING.ZERO_INIT_OUTPUT True

      OFFLINE.NUM_EPISODES "$episodes"
      OFFLINE.NUM_UPDATES_PER_EPISODE "$updates_per_episode"
      OFFLINE.Q_SAMPLES "$q_samples"
      OFFLINE.OPTIMIZER Adam
      OFFLINE.LR "$lr"

      MIXED_GRADUAL.FRAMES "$frames"
      MIXED_GRADUAL.SEVERITY "$severity"
      MIXED_GRADUAL.SCHEDULE_TYPE "$schedule_type"
    )

    if ! run_logged "$offline_log" "${offline_cmd[@]}"; then
      record_result \
        "$architecture" \
        "$lr" \
        "offline" \
        "$offline_seed" \
        "yaml" \
        "yaml" \
        "FAILED" \
        "" \
        "$offline_log"

      fail_or_continue \
        "[NONLINEAR] Offline training failed for $architecture (LR=$lr)"

      continue
    fi

    if [[ "$DRY_RUN" = "1" ]]; then
      checkpoint="<DRY_RUN_CHECKPOINT_${run_key}>"
    else
      checkpoint="$(find_new_checkpoint "$marker")"

      if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
        record_result \
          "$architecture" \
          "$lr" \
          "offline" \
          "$offline_seed" \
          "yaml" \
          "yaml" \
          "NO_CHECKPOINT" \
          "" \
          "$offline_log"

        fail_or_continue \
          "[NONLINEAR] No checkpoint found for $architecture (LR=$lr)"

        continue
      fi
    fi

    echo "$checkpoint" > "$architecture_dir/checkpoint_path.txt"

    record_result \
      "$architecture" \
      "$lr" \
      "offline" \
      "$offline_seed" \
      "yaml" \
      "yaml" \
      "OK" \
      "$checkpoint" \
      "$offline_log"

    echo "[NONLINEAR] Checkpoint: $checkpoint"

    if [[ "$DRY_RUN" != "1" ]]; then
      checkpoint_size="$(
        du -h "$checkpoint" 2>/dev/null \
          | awk '{print $1}'
      )"

      echo \
        "[NONLINEAR] Checkpoint size: ${checkpoint_size:-unknown}" \
        | tee "$architecture_dir/checkpoint_size.txt"
    fi

    # ---------------------------------------------------------------------------
    # Online covariance sweep
    # ---------------------------------------------------------------------------

    for init_cov in "${init_cov_scales[@]}"; do
      for obs_cov in "${obs_cov_scales[@]}"; do
        for seed in "${seeds[@]}"; do

          online_log="$architecture_dir/online_init${init_cov}_obs${obs_cov}_seed${seed}.log"

          echo
          echo "-------------------------------------------------------------"
          echo "[NONLINEAR] Architecture: $architecture | LR: $lr"
          echo "[NONLINEAR] Seed: $seed"
          echo "[NONLINEAR] INIT_COV_SCALE: $init_cov"
          echo "[NONLINEAR] OBS_COV_SCALE: $obs_cov"
          echo "-------------------------------------------------------------"

          online_cmd=(
            python test_time.py
            --cfg "$online_cfg"

            RNG_SEED "$seed"
            SETTING gradual_mixed_corruption

            AURA.LATENT_DIM "$latent_dim"
            AURA.INIT_COV_SCALE "$init_cov"
            AURA.OBS_COV_SCALE "$obs_cov"

            AURA.LIFTING.TYPE nonlinear
            AURA.LIFTING.RESIDUAL_LINEAR True
            AURA.LIFTING.HIDDEN_DIMS "$hidden_dims"
            AURA.LIFTING.ACTIVATION gelu
            AURA.LIFTING.NORMALIZATION none
            AURA.LIFTING.OUTPUT_SCALE 1.0
            AURA.LIFTING.ZERO_INIT_OUTPUT True

            AURA.OFFLINE_CKPT "$checkpoint"

            MIXED_GRADUAL.FRAMES "$frames"
            MIXED_GRADUAL.SEVERITY "$severity"
            MIXED_GRADUAL.SCHEDULE_TYPE "$schedule_type"

            TIMING.ENABLED True
          )

          if run_logged "$online_log" "${online_cmd[@]}"; then
            record_result \
              "$architecture" \
              "$lr" \
              "online" \
              "$seed" \
              "$init_cov" \
              "$obs_cov" \
              "OK" \
              "$checkpoint" \
              "$online_log"
          else
            record_result \
              "$architecture" \
              "$lr" \
              "online" \
              "$seed" \
              "$init_cov" \
              "$obs_cov" \
              "FAILED" \
              "$checkpoint" \
              "$online_log"

            fail_or_continue \
              "[NONLINEAR] Online run failed: architecture=$architecture lr=$lr seed=$seed init=$init_cov obs=$obs_cov"
          fi
        done
      done
    done
  done
done

# -----------------------------------------------------------------------------
# Phase 2: Post-Sweep Evaluation (5, 10, 15 Labeled Adaptation Samples)
#
# Find the best architecture + LR + covariance scale combination from Phase 1,
# then evaluate across 5, 10, 15 labeled adaptation samples.
# -----------------------------------------------------------------------------

echo
echo "================================================================="
echo "[NONLINEAR] Phase 2: Evaluating Best Configuration on 5, 10, 15 Labeled Samples"
echo "================================================================="

best_config=""
if [[ "$DRY_RUN" = "1" ]]; then
  best_arch="gelu_1x500"
  best_lr="0.0003"
  best_init_cov="1e-3"
  best_obs_cov="1e-3"
  best_avg_err="25.00"
else
  best_config=$(python3 -c "
import glob, os, re, sys

manifest = '$manifest'
results = {}

if not os.path.exists(manifest):
    sys.exit(0)

with open(manifest, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 9 and parts[2] == 'online' and parts[6] == 'OK':
            arch, lr, phase, seed, init_cov, obs_cov, status, ckpt, log_file = parts[:9]
            if os.path.exists(log_file):
                with open(log_file, 'r') as lf:
                    content = lf.read()
                m = re.search(r'mean error:\s*([0-9.]+)%', content)
                if not m:
                    m = re.search(r'error %.*:\s*([0-9.]+)%', content)
                if m:
                    err = float(m.group(1))
                    key = (arch, lr, init_cov, obs_cov)
                    results.setdefault(key, []).append(err)

if results:
    best_key = min(results.keys(), key=lambda k: sum(results[k])/len(results[k]))
    best_avg = sum(results[best_key]) / len(results[best_key])
    print(f'{best_key[0]}|{best_key[1]}|{best_key[2]}|{best_key[3]}|{best_avg:.4f}')
" || true)

  if [[ -n "$best_config" ]]; then
    IFS='|' read -r best_arch best_lr best_init_cov best_obs_cov best_avg_err <<< "$best_config"
  else
    best_arch="gelu_1x500"
    best_lr="0.0003"
    best_init_cov="1e-3"
    best_obs_cov="1e-3"
    best_avg_err="N/A"
  fi
fi

echo "[NONLINEAR] Best Configuration Selection:"
echo "  Architecture:     $best_arch"
echo "  Learning Rate:    $best_lr"
echo "  INIT_COV_SCALE:   $best_init_cov"
echo "  OBS_COV_SCALE:    $best_obs_cov"
echo "  Phase 1 Mean Err: $best_avg_err%"

best_hidden_dims="[500]"
for spec in "${architectures[@]}"; do
  IFS='|' read -r arch hidden_dims <<< "$spec"
  if [[ "$arch" == "$best_arch" ]]; then
    best_hidden_dims="$hidden_dims"
    break
  fi
done

best_run_key="${best_arch}_lr${best_lr}"
best_checkpoint=""
if [[ -f "$run_root/$best_run_key/checkpoint_path.txt" ]]; then
  best_checkpoint="$(cat "$run_root/$best_run_key/checkpoint_path.txt")"
fi
if [[ -z "$best_checkpoint" && "$DRY_RUN" = "1" ]]; then
  best_checkpoint="<DRY_RUN_CHECKPOINT_${best_run_key}>"
fi

if [[ -z "$best_checkpoint" && "$DRY_RUN" != "1" ]]; then
  echo "[NONLINEAR] WARNING: No checkpoint found for best configuration $best_run_key. Skipping Phase 2." >&2
else
  echo "[NONLINEAR] Best Checkpoint: $best_checkpoint"
  echo "[NONLINEAR] Labeled Samples Grid: ${eval_adapt_samples[*]}"
  echo "[NONLINEAR] Seeds for Evaluation: ${eval_seeds[*]}"

  eval_dir="$run_root/few_shot_eval"
  mkdir -p "$eval_dir"

  for adapt_samples in "${eval_adapt_samples[@]}"; do
    for seed in "${eval_seeds[@]}"; do
      eval_log="$eval_dir/adapt_samples${adapt_samples}_seed${seed}.log"

      echo
      echo "-------------------------------------------------------------"
      echo "[NONLINEAR] Evaluating NUM_ADAPT_SAMPLES=$adapt_samples (Seed=$seed)"
      echo "-------------------------------------------------------------"

      eval_cmd=(
        python test_time.py
        --cfg "$online_cfg"

        RNG_SEED "$seed"
        SETTING gradual_mixed_corruption

        AURA.LATENT_DIM "$latent_dim"
        AURA.INIT_COV_SCALE "$best_init_cov"
        AURA.OBS_COV_SCALE "$best_obs_cov"

        AURA.LIFTING.TYPE nonlinear
        AURA.LIFTING.RESIDUAL_LINEAR True
        AURA.LIFTING.HIDDEN_DIMS "$best_hidden_dims"
        AURA.LIFTING.ACTIVATION gelu
        AURA.LIFTING.NORMALIZATION none
        AURA.LIFTING.OUTPUT_SCALE 1.0
        AURA.LIFTING.ZERO_INIT_OUTPUT True

        AURA.NUM_ADAPT_SAMPLES "$adapt_samples"
        AURA.OFFLINE_CKPT "$best_checkpoint"

        MIXED_GRADUAL.FRAMES "$frames"
        MIXED_GRADUAL.SEVERITY "$severity"
        MIXED_GRADUAL.SCHEDULE_TYPE "$schedule_type"

        TIMING.ENABLED True
      )

      if run_logged "$eval_log" "${eval_cmd[@]}"; then
        record_result \
          "$best_arch" \
          "$best_lr" \
          "adapt_${adapt_samples}" \
          "$seed" \
          "$best_init_cov" \
          "$best_obs_cov" \
          "OK" \
          "$best_checkpoint" \
          "$eval_log"
      else
        record_result \
          "$best_arch" \
          "$best_lr" \
          "adapt_${adapt_samples}" \
          "$seed" \
          "$best_init_cov" \
          "$best_obs_cov" \
          "FAILED" \
          "$best_checkpoint" \
          "$eval_log"

        fail_or_continue \
          "[NONLINEAR] Few-shot evaluation failed: adapt_samples=$adapt_samples seed=$seed"
      fi
    done
  done
fi

# -----------------------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------------------

{
  echo "================================================================="
  echo "AURA NONLINEAR CAPACITY + LR + LABELED SAMPLES EVALUATION SUMMARY"
  echo "Finished: $(date)"
  echo "================================================================="
  echo
  echo "Run directory:"
  echo "$run_root"
  echo
  echo "Best Configuration Selected:"
  echo "  Architecture:   ${best_arch:-N/A}"
  echo "  Learning Rate:  ${best_lr:-N/A}"
  echo "  INIT_COV_SCALE: ${best_init_cov:-N/A}"
  echo "  OBS_COV_SCALE:  ${best_obs_cov:-N/A}"
  echo
  echo "Status counts:"
  awk -F '\t' '
    NR > 1 {
      count[$3 ":" $7]++
    }
    END {
      for (key in count) {
        print "  " key " = " count[key]
      }
    }
  ' "$manifest" | sort

  echo
  echo "Checkpoints:"
  find "$run_root" \
    -name checkpoint_path.txt \
    -type f \
    -exec sh -c '
      for file do
        printf "%s: " "$(basename "$(dirname "$file")")"
        cat "$file"
      done
    ' sh {} +

  echo
  echo "================================================================="
  echo "FEW-SHOT LABELED ADAPTATION SAMPLES BENCHMARK SUMMARY"
  echo "================================================================="
  python3 -c "
import glob, os, re, sys
import numpy as np

manifest = '$manifest'
eval_dir = '$run_root/few_shot_eval'

if not os.path.exists(eval_dir):
    print('No few-shot evaluation directory found.')
    sys.exit(0)

print(f'{\"Adapt Samples\":<15} | {\"Mean Error (%)\":<15} | {\"Mean Accuracy (%)\":<18} | {\"Seeds\":<10}')
print('-' * 65)

for s in [5, 10, 15]:
    errs = []
    logs = glob.glob(os.path.join(eval_dir, f'adapt_samples{s}_seed*.log'))
    for log in logs:
        with open(log, 'r') as f:
            content = f.read()
        m = re.search(r'mean error:\s*([0-9.]+)%', content)
        if not m:
            m = re.search(r'error %.*:\s*([0-9.]+)%', content)
        if m:
            errs.append(float(m.group(1)))
    if errs:
        mean_err = np.mean(errs)
        std_err = np.std(errs)
        mean_acc = 100.0 - mean_err
        print(f'{s:<15} | {mean_err:.2f} ± {std_err:.2f}%      | {mean_acc:.2f} ± {std_err:.2f}%       | {len(errs):<10}')
    else:
        print(f'{s:<15} | N/A             | N/A                | 0')
" || true

  echo
  echo "Manifest:"
  echo "$manifest"
} | tee "$summary_file"

echo
echo "[NONLINEAR] Experiment completed."
echo "[NONLINEAR] Logs: $run_root"
echo "[NONLINEAR] Manifest: $manifest"
echo "[NONLINEAR] Summary: $summary_file"