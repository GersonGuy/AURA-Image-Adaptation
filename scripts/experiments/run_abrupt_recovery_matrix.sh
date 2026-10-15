#!/bin/bash -l
# =============================================================================
# Abrupt-Shift Recovery Matrix Runner (3-Stage Architecture)
#
# Environment Variables:
#   DRY_RUN=1               : Print all configs, commands, metadata without executing
#   FORCE_RETRAIN=1         : Force fresh retraining of offline checkpoints
#   SMOKE_TEST=1            : Run single manual smoke test (CIFAR-10-C, d=500, linear, 1 corruption, u=0..4)
#   OFFLINE_NUM_EPISODES=N  : Override offline episode count (default: 1500)
#   MAX_UPDATES=N           : Override max online updates per corruption block (default: 20)
#   STAGE=1|2|3             : Run only specific stage (Stage 1: Offline, Stage 2: Validate, Stage 3: Online)
# =============================================================================

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
else
  PYTHON_BIN="python"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DRY_RUN="${DRY_RUN:-0}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
SMOKE_TEST="${SMOKE_TEST:-0}"
STAGE="${STAGE:-all}"
OFFLINE_NUM_EPISODES="${OFFLINE_NUM_EPISODES:-1500}"
MAX_UPDATES="${MAX_UPDATES:-20}"

EXP_CKPT_BASE="ckpt/abrupt_recovery_experiment"
mkdir -p "$EXP_CKPT_BASE"

timestamp="$(date +%Y%m%d_%H%M%S)"
run_root="logs/abrupt_recovery/${timestamp}"
detailed_csv="$run_root/abrupt_recovery_detailed.csv"

# 8 Required Configurations
datasets=("cifar10_c")
latent_dims=(500 1000)
lifting_types=("linear" "nonlinear")
seeds=(1 2)

# 4 Key Covariance Grid Configurations
cov_inits=(1e-4 1e-3 1e-2 1e-2)
cov_obss=( 1e-4 1e-3 1e-4 1e-2)

get_lifting_arch() {
  local dim="$1"
  local lifting="$2"
  if [[ "$lifting" == "linear" ]]; then
    echo "linear_map"
  else
    echo "MLP_h${dim}_gelu"
  fi
}

get_target_ckpt_dir() {
  local dataset="$1"
  local dim="$2"
  local lifting="$3"
  if [[ "$SMOKE_TEST" == "1" ]]; then
    echo "${EXP_CKPT_BASE}/smoke_${dataset}__d${dim}__${lifting}"
  else
    echo "${EXP_CKPT_BASE}/${dataset}__d${dim}__${lifting}"
  fi
}

check_checkpoint_state() {
  local target_dir="$1"
  local dataset="$2"
  local dim="$3"
  local lifting="$4"

  if [[ ! -d "$target_dir" ]]; then
    echo "MISSING"
    return
  fi

  local meta_file="${target_dir}/metadata.json"
  local pth_file="$(find "$target_dir" -name "last_params.pth" -o -name "best_params.pth" | head -1 || true)"

  if [[ -z "$pth_file" || ! -f "$meta_file" ]]; then
    echo "INCOMPLETE"
    return
  fi

  # Validate metadata JSON
  "$PYTHON_BIN" -c "
import json, sys
try:
    with open('$meta_file', 'r') as f:
        meta = json.load(f)
    if meta.get('status') == 'COMPLETED' and meta.get('dataset') == '$dataset' and int(meta.get('latent_dim')) == $dim and meta.get('lifting_type') == '$lifting':
        sys.exit(0)
    else:
        sys.exit(1)
except Exception:
    sys.exit(1)
" >/dev/null 2>&1 && echo "COMPLETED" || echo "MISMATCHING"
}

# =============================================================================
# DRY RUN MODE (DRY_RUN=1)
# =============================================================================
if [[ "$DRY_RUN" == "1" ]]; then
  echo "================================================================="
  echo "[ABRUPT RECOVERY] DRY-RUN MODE (DRY_RUN=1)"
  echo "  No models will be initialized, trained, or evaluated."
  echo "================================================================="
  echo ""
  echo "--- STAGE 1: 8 REQUIRED FRESH OFFLINE CONFIGURATIONS ---"

  for dataset in "${datasets[@]}"; do
    for dim in "${latent_dims[@]}"; do
      for lifting in "${lifting_types[@]}"; do
        arch="$(get_lifting_arch "$dim" "$lifting")"
        ckpt_dir="$(get_target_ckpt_dir "$dataset" "$dim" "$lifting")"
        state="$(check_checkpoint_state "$ckpt_dir" "$dataset" "$dim" "$lifting")"
        
        if [[ "$dataset" == "cifar10_c" ]]; then
          off_cfg="cfgs/cifar10_c/offline_aura_nonlinear.yaml"
        else
          off_cfg="cfgs/cifar100_c/offline_aura_nonlinear.yaml"
        fi

        echo ""
        echo "Configuration: dataset=$dataset | d=$dim | lifting=$lifting | arch=$arch"
        echo "  - Offline Seed:        1"
        echo "  - Offline Episodes:    $OFFLINE_NUM_EPISODES"
        echo "  - Target Checkpoint:   $ckpt_dir"
        echo "  - Current State:       $state"
        echo "  - Expected Metadata:   {dataset: '$dataset', latent_dim: $dim, lifting_type: '$lifting', lifting_arch: '$arch', regime: 'gradual_mixed_corruption', status: 'COMPLETED'}"
        echo "  - Resolved Command:    python offline_train_aura.py --cfg $off_cfg CORRUPTION.DATASET $dataset AURA.LATENT_DIM $dim AURA.LIFTING.TYPE $lifting AURA.LIFTING.HIDDEN_DIMS \"[$dim]\" OFFLINE.NUM_EPISODES $OFFLINE_NUM_EPISODES SAVE_DIR $ckpt_dir"
      done
    done
  done

  echo ""
  echo "--- STAGE 2: CHECKPOINT VALIDATION COMMANDS ---"
  for dataset in "${datasets[@]}"; do
    for dim in "${latent_dims[@]}"; do
      for lifting in "${lifting_types[@]}"; do
        ckpt_dir="$(get_target_ckpt_dir "$dataset" "$dim" "$lifting")"
        echo "  python scripts/analysis/validate_checkpoints.py --ckpt_dir $ckpt_dir --dataset $dataset --latent_dim $dim --lifting_type $lifting"
      done
    done
  done

  echo ""
  echo "--- STAGE 3: COMPLETE ONLINE COVARIANCE MATRIX ---"
  echo "  Total Online Covariance Configurations: 9 (cov1 .. cov9)"
  for i in "${!cov_inits[@]}"; do
    echo "  cov$((i+1)): AURA.INIT_COV_SCALE=${cov_inits[$i]} | AURA.OBS_COV_SCALE=${cov_obss[$i]}"
  done
  echo "  Total Seeds per Config: 3 (seeds: 1, 2, 3)"
  echo "  Total Online Runs per Dataset: 2 dims x 2 liftings x 9 covs x 3 seeds = 108 runs"
  echo "  Output Directory: $run_root"
  echo "  Detailed Results CSV: $detailed_csv"

  echo ""
  echo "--- MINIMAL MANUAL SMOKE-TEST COMMAND ---"
  echo "  SMOKE_TEST=1 OFFLINE_NUM_EPISODES=5 MAX_UPDATES=4 CUDA_VISIBLE_DEVICES=0 bash scripts/experiments/run_abrupt_recovery_matrix.sh"
  echo "================================================================="
  exit 0
fi

# =============================================================================
# STAGE 1: Fresh Offline Training
# =============================================================================
if [[ "$STAGE" == "all" || "$STAGE" == "1" ]]; then
  echo "================================================================="
  echo "[Stage 1/3] Training Fresh Offline Checkpoints"
  echo "================================================================="

  for dataset in "${datasets[@]}"; do
    for dim in "${latent_dims[@]}"; do
      for lifting in "${lifting_types[@]}"; do
        if [[ "$SMOKE_TEST" == "1" && ("$dataset" != "cifar10_c" || "$dim" != "500" || "$lifting" != "linear") ]]; then
          continue
        fi

        arch="$(get_lifting_arch "$dim" "$lifting")"
        ckpt_dir="$(get_target_ckpt_dir "$dataset" "$dim" "$lifting")"
        state="$(check_checkpoint_state "$ckpt_dir" "$dataset" "$dim" "$lifting")"

        echo "[Offline Check] dataset=$dataset | d=$dim | lifting=$lifting | arch=$arch | state=$state"

        if [[ "$FORCE_RETRAIN" == "1" && "$state" != "MISSING" ]]; then
          echo "  FORCE_RETRAIN=1: Deleting existing checkpoint directory $ckpt_dir"
          rm -rf "$ckpt_dir"
          state="MISSING"
        fi

        if [[ "$state" == "COMPLETED" ]]; then
          echo "  Completed experiment checkpoint found at $ckpt_dir. Reusing."
          continue
        elif [[ "$state" == "MISMATCHING" ]]; then
          echo "  ERROR: Mismatching checkpoint found at $ckpt_dir! Delete or set FORCE_RETRAIN=1."
          exit 1
        fi

        echo "  [Stage 1 Execution] Training fresh offline checkpoint..."
        if [[ "$dataset" == "cifar10_c" ]]; then
          off_cfg="cfgs/cifar10_c/offline_aura_nonlinear.yaml"
        else
          off_cfg="cfgs/cifar100_c/offline_aura_nonlinear.yaml"
        fi

        episodes="$OFFLINE_NUM_EPISODES"
        if [[ "$SMOKE_TEST" == "1" ]]; then
          episodes=5
        fi

        mkdir -p "$ckpt_dir"
        "$PYTHON_BIN" offline_train_aura.py \
          --cfg "$off_cfg" \
          CORRUPTION.DATASET "$dataset" \
          AURA.LATENT_DIM "$dim" \
          AURA.LIFTING.TYPE "$lifting" \
          AURA.LIFTING.HIDDEN_DIMS "[$dim]" \
          OFFLINE.NUM_EPISODES "$episodes" \
          SAVE_DIR "$ckpt_dir"

        # Copy any parameters saved in a timestamped subfolder up to root $ckpt_dir
        sub_last="$(find "$ckpt_dir" -mindepth 2 -name "last_params.pth" | head -1 || true)"
        if [[ -n "$sub_last" && -f "$sub_last" ]]; then
          cp "$sub_last" "${ckpt_dir}/last_params.pth"
        fi
        sub_best="$(find "$ckpt_dir" -mindepth 2 -name "best_params.pth" | head -1 || true)"
        if [[ -n "$sub_best" && -f "$sub_best" ]]; then
          cp "$sub_best" "${ckpt_dir}/best_params.pth"
        fi

        # Write metadata.json upon successful completion
        "$PYTHON_BIN" -c "
import json, datetime
meta = {
    'dataset': '$dataset',
    'latent_dim': $dim,
    'lifting_type': '$lifting',
    'lifting_arch': '$arch',
    'regime': 'gradual_mixed_corruption',
    'offline_seed': 1,
    'num_episodes': $episodes,
    'status': 'COMPLETED',
    'created_at': datetime.datetime.now().isoformat()
}
with open('${ckpt_dir}/metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
"
        echo "  [Stage 1 Confirmation] Successfully trained and saved metadata to ${ckpt_dir}/metadata.json"
      done
    done
  done
fi

# =============================================================================
# STAGE 2: Checkpoint Validation
# =============================================================================
if [[ "$STAGE" == "all" || "$STAGE" == "2" ]]; then
  echo "================================================================="
  echo "[Stage 2/3] Validating Offline Checkpoints"
  echo "================================================================="

  for dataset in "${datasets[@]}"; do
    for dim in "${latent_dims[@]}"; do
      for lifting in "${lifting_types[@]}"; do
        if [[ "$SMOKE_TEST" == "1" && ("$dataset" != "cifar10_c" || "$dim" != "500" || "$lifting" != "linear") ]]; then
          continue
        fi

        ckpt_dir="$(get_target_ckpt_dir "$dataset" "$dim" "$lifting")"
        echo "[Stage 2 Validation] Checking $ckpt_dir ..."
        "$PYTHON_BIN" scripts/analysis/validate_checkpoints.py \
          --ckpt_dir "$ckpt_dir" \
          --dataset "$dataset" \
          --latent_dim "$dim" \
          --lifting_type "$lifting"
      done
    done
  done
fi

# =============================================================================
# STAGE 3: Online Abrupt Evaluation
# =============================================================================
if [[ "$STAGE" == "all" || "$STAGE" == "3" ]]; then
  echo "================================================================="
  echo "[Stage 3/3] Running Online Covariance Sweep"
  echo "================================================================="

  eval_seeds=("${seeds[@]}")
  eval_cov_inits=("${cov_inits[@]}")
  eval_cov_obss=("${cov_obss[@]}")

  if [[ "$SMOKE_TEST" == "1" ]]; then
    eval_seeds=(1)
    eval_cov_inits=(1e-3)
    eval_cov_obss=(1e-3)
  fi

  for dataset in "${datasets[@]}"; do
    if [[ "$dataset" == "cifar10_c" ]]; then
      aura_cfg="cfgs/cifar10_c/aura_nonlinear.yaml"
    else
      aura_cfg="cfgs/cifar100_c/aura_nonlinear.yaml"
    fi

    for dim in "${latent_dims[@]}"; do
      for lifting in "${lifting_types[@]}"; do
        if [[ "$SMOKE_TEST" == "1" && ("$dataset" != "cifar10_c" || "$dim" != "500" || "$lifting" != "linear") ]]; then
          continue
        fi

        ckpt_dir="$(get_target_ckpt_dir "$dataset" "$dim" "$lifting")"
        ckpt_file="$(find "$ckpt_dir" -name "last_params.pth" -o -name "best_params.pth" | head -1 || true)"

        for c_idx in "${!eval_cov_inits[@]}"; do
          cov_id="cov$((c_idx + 1))"
          init_cov="${eval_cov_inits[$c_idx]}"
          obs_cov="${eval_cov_obss[$c_idx]}"

          for seed in "${eval_seeds[@]}"; do
            echo "[Stage 3 Execution] dataset=$dataset | d=$dim | $lifting | $cov_id (init=$init_cov, obs=$obs_cov) | seed=$seed"

            smoke_flag=""
            if [[ "$SMOKE_TEST" == "1" ]]; then
              smoke_flag="--smoke"
            fi

            "$PYTHON_BIN" scripts/analysis/evaluate_abrupt_recovery.py \
              --cfg "$aura_cfg" \
              --dataset "$dataset" \
              --latent_dim "$dim" \
              --lifting_type "$lifting" \
              --init_cov "$init_cov" \
              --obs_cov "$obs_cov" \
              --seed "$seed" \
              --checkpoint "$ckpt_file" \
              --output_csv "$detailed_csv" \
              --max_updates "$MAX_UPDATES" \
              --cov_config_id "$cov_id" \
              $smoke_flag
          done
        done
      done
    done
  done

  # Plot summary results
  if [[ -f "$detailed_csv" ]]; then
    echo "[Stage 3 Plotting] Generating summary tables and plots..."
    "$PYTHON_BIN" scripts/analysis/plot_abrupt_recovery.py --csv "$detailed_csv" --output_dir "$run_root"
  fi
fi

echo "================================================================="
echo "[ABRUPT RECOVERY] Stage Execution Completed Successfully!"
echo "================================================================="
