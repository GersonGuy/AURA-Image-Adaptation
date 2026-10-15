#!/bin/bash -l
# =============================================================================
# LoRA Baseline — Mixed-gradual experiment (fully automated, single command)
#
# Usage:
#   bash scripts/experiments/lora_mixed_gradual.sh cifar10_c
#   bash scripts/experiments/lora_mixed_gradual.sh cifar100_c
#   bash scripts/experiments/lora_mixed_gradual.sh both      # run both datasets
#
# Runs all three phases end-to-end without any manual intervention:
#
#   Phase 1 — LR / rank tuning
#     Scope:  fc_stage1 only
#     Ranks:  2, 4, 8
#     LRs:    1e-4, 1e-3, 1e-2  (one-decade log spacing around SGD baseline 1e-3)
#     Seed:   1
#     → 9 runs
#     → best rank + LR selected automatically by lowest mean error
#
#   Phase 2 — Scope comparison
#     Uses best rank + LR from Phase 1
#     Scopes:  fc, fc_stage1
#     Seed:    1
#     → 2 runs
#     → best scope selected automatically
#
#   Phase 3 — Final evaluation
#     Uses best rank + LR + scope from Phases 1–2
#     Seeds:  1, 2, 3
#     → 3 runs  (seed-1 result re-used from Phase 1 or 2 — not re-run)
#
# Stream: identical to the existing AURA mixed-gradual setup.
#   Same YAML → same FRAMES, BATCH_SIZE, CORRUPTION_TYPES, SEVERITY,
#   SCHEDULE_TYPE, RNG_SEED.  Only LORA.RANK / LORA.LR / LORA.SCOPE
#   are overridden on the command line.
#
# Supervision budget: NUM_ADAPT_SAMPLES=15, NUM_UPDATES_PER_SAMPLE=1
#   (matches AURA and SGD_Base).
#
# Modes:
#   SMOKE=1   — one run only (rank=4, lr=1e-3, fc_stage1, seed=1, NUM_EX=400)
#   DRY_RUN=1 — print all commands, execute nothing
#
# Override sweep values:
#   ranks="2 4" lrs="1e-3 1e-2" final_seeds="1 2" \
#     bash scripts/experiments/lora_mixed_gradual.sh cifar10_c
# =============================================================================

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Dataset argument (positional, required)
# ---------------------------------------------------------------------------
_DATASET_ARG="${1:-}"
if [[ -z "$_DATASET_ARG" ]]; then
  echo "Usage: bash $0 [cifar10_c|cifar100_c|both]" >&2
  exit 1
fi

# If 'both', recurse for each dataset and exit
if [[ "$_DATASET_ARG" == "both" ]]; then
  bash "${BASH_SOURCE[0]}" cifar10_c
  bash "${BASH_SOURCE[0]}" cifar100_c
  exit 0
fi

DATASET="$_DATASET_ARG"
if [[ "$DATASET" != "cifar10_c" && "$DATASET" != "cifar100_c" ]]; then
  echo "Dataset must be 'cifar10_c', 'cifar100_c', or 'both'. Got: '$DATASET'" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Repo root
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  source "$REPO_ROOT/.venv/bin/activate"
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate tta || true
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

ranks=(${ranks:-2 4 8})
lrs=(${lrs:-1e-4 1e-3 1e-2})
tuning_seed=1
scopes=(fc fc_stage1)
final_seeds=(${final_seeds:-1 2 3})

case "$DATASET" in
  cifar10_c)  cfg_file="cfgs/cifar10_c/lora.yaml"  ;;
  cifar100_c) cfg_file="cfgs/cifar100_c/lora.yaml" ;;
esac

if [[ ! -f "$cfg_file" ]]; then
  echo "Config not found: $cfg_file" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Smoke mode
# ---------------------------------------------------------------------------
if [[ "$SMOKE" = "1" ]]; then
  ranks=(4)
  lrs=(1e-3)
  scopes=(fc_stage1)
  final_seeds=(1)
  smoke_num_ex="${smoke_num_ex:-400}"
  echo "================================================================="
  echo "[LoRA] SMOKE mode — $DATASET"
  echo "  rank=4, lr=1e-3, scope=fc_stage1, seed=1, NUM_EX=$smoke_num_ex"
  echo "================================================================="
fi

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
timestamp="$(date +%Y%m%d_%H%M%S)"
run_root="${run_root:-logs/lora_mixed_gradual/${DATASET}/${timestamp}}"
mkdir -p "$run_root"/{phase1_tuning,phase2_scope,phase3_final}

manifest="$run_root/manifest.tsv"
summary_file="$run_root/final_summary.txt"

printf "phase\tdataset\tscope\trank\tlr\tseed\tstatus\tmean_error\tlog\n" > "$manifest"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
run_logged() {
  local log_file="$1"; shift
  echo "+ $*" | tee -a "$log_file"
  if [[ "$DRY_RUN" = "1" ]]; then return 0; fi
  set +e
  "$@" 2>&1 | tee -a "$log_file"
  local rc=${PIPESTATUS[0]}
  set -e
  return "$rc"
}

# Extract mean error as a float (e.g. "45.23%" → "45.23")
# Returns "999" if not found (sorts last when picking minimum)
extract_error_float() {
  local log="$1"
  [[ ! -f "$log" ]] && { echo "999"; return; }
  local pct
  pct="$(grep -Ei "mean error:" "$log" 2>/dev/null | tail -1 \
         | grep -oP '[0-9]+\.[0-9]+(?=%)' | head -1 || true)"
  echo "${pct:-999}"
}

extract_error_display() {
  local log="$1"
  [[ ! -f "$log" ]] && { echo "no_log"; return; }
  local val
  val="$(grep -Ei "mean error:" "$log" 2>/dev/null | tail -1 \
         | grep -oP '[0-9]+\.[0-9]+%' | head -1 || true)"
  echo "${val:-not_found}"
}

record() {
  local phase="$1" scope="$2" rank="$3" lr="$4" seed="$5" \
        status="$6" err="$7" log="$8"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$phase" "$DATASET" "$scope" "$rank" "$lr" "$seed" \
    "$status" "$err" "$log" >> "$manifest"
}

fail_or_continue() {
  echo "$1" >&2
  [[ "$CONTINUE_ON_ERROR" != "1" ]] && exit 1
}

# Build and echo the python test_time.py command for a given config
build_cmd() {
  local _scope="$1" _rank="$2" _lr="$3" _seed="$4"
  local _alpha
  _alpha="$(python3 -c "print(float(2*${_rank}))")"
  local cmd=(
    python test_time.py
    --cfg "$cfg_file"
    RNG_SEED "$_seed"
    SETTING gradual_mixed_corruption
    LORA.RANK "$_rank"
    LORA.ALPHA "$_alpha"
    LORA.LR "$_lr"
    LORA.SCOPE "$_scope"
    TIMING.ENABLED True
  )
  if [[ "$SMOKE" = "1" ]]; then
    cmd+=(CORRUPTION.NUM_EX "$smoke_num_ex")
  fi
  echo "${cmd[@]}"
}

# ---------------------------------------------------------------------------
# Pre-flight banner
# ---------------------------------------------------------------------------
echo "================================================================="
echo "[LoRA] Mixed-gradual baseline — $DATASET  (fully automated)"
echo "================================================================="
echo "Config:       $cfg_file"
echo "Phase 1:      fc_stage1, ranks=${ranks[*]}, lrs=${lrs[*]}, seed=$tuning_seed"
echo "Phase 2:      best rank+LR, scopes=${scopes[*]}, seed=$tuning_seed"
echo "Phase 3:      best config, seeds=${final_seeds[*]}"
echo "Output root:  $run_root"
echo "DRY_RUN:      $DRY_RUN  |  SMOKE: $SMOKE"
echo "================================================================="

{
  echo "timestamp=$timestamp"
  echo "dataset=$DATASET"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "ranks=${ranks[*]}"
  echo "lrs=${lrs[*]}"
  echo "tuning_seed=$tuning_seed"
  echo "final_seeds=${final_seeds[*]}"
} > "$run_root/run_metadata.txt"

# =============================================================================
# PHASE 1 — LR / Rank tuning  (scope=fc_stage1, seed=1)
# =============================================================================
echo
echo "================================================================="
echo "[LoRA] Phase 1: LR/rank tuning (scope=fc_stage1, seed=$tuning_seed)"
echo "================================================================="

# best tracking variables
best_p1_err=999
best_rank="${best_rank:-}"
best_lr="${best_lr:-}"

for rank in "${ranks[@]}"; do
  for lr in "${lrs[@]}"; do
    log="$run_root/phase1_tuning/rank${rank}_lr${lr}.log"

    echo
    echo "-------------------------------------------------------------"
    echo "[LoRA-P1] $DATASET  scope=fc_stage1  rank=$rank  lr=$lr  seed=$tuning_seed"
    echo "-------------------------------------------------------------"

    cmd=($(build_cmd "fc_stage1" "$rank" "$lr" "$tuning_seed"))
    if run_logged "$log" "${cmd[@]}"; then
      status="OK"
    else
      status="FAILED"
      fail_or_continue "[LoRA-P1] Failed: rank=$rank lr=$lr"
    fi

    err_disp="$(extract_error_display "$log")"
    record "phase1" "fc_stage1" "$rank" "$lr" "$tuning_seed" "$status" "$err_disp" "$log"

    # Auto-select best (lowest error) — only when not DRY_RUN
    if [[ "$DRY_RUN" != "1" && "$status" == "OK" ]]; then
      err_float="$(extract_error_float "$log")"
      # python3 for floating-point comparison
      if python3 -c "exit(0 if float('$err_float') < float('$best_p1_err') else 1)" 2>/dev/null; then
        best_p1_err="$err_float"
        best_rank="$rank"
        best_lr="$lr"
      fi
    fi
  done
done

# In DRY_RUN mode, pick a default so phases 2 and 3 print commands too
if [[ "$DRY_RUN" = "1" ]]; then
  best_rank="${best_rank:-4}"
  best_lr="${best_lr:-1e-3}"
fi

echo
echo "================================================================="
echo "[LoRA] Phase 1 complete."
echo "  Best config: rank=$best_rank  lr=$best_lr  error=${best_p1_err}%"
echo "================================================================="

# =============================================================================
# PHASE 2 — Scope comparison  (best rank+LR, seed=1)
# =============================================================================
echo
echo "================================================================="
echo "[LoRA] Phase 2: Scope comparison (rank=$best_rank, lr=$best_lr)"
echo "================================================================="

best_scope="${best_scope:-fc_stage1}"   # default if only 1 scope or DRY_RUN
best_p2_err=999

for scope in "${scopes[@]}"; do
  log="$run_root/phase2_scope/${scope}_rank${best_rank}_lr${best_lr}.log"

  # For fc_stage1 seed=1 the result already exists from Phase 1 — copy the log
  # to avoid a redundant run, then skip.
  if [[ "$scope" == "fc_stage1" && "$DRY_RUN" != "1" ]]; then
    p1_log="$run_root/phase1_tuning/rank${best_rank}_lr${best_lr}.log"
    if [[ -f "$p1_log" ]]; then
      cp "$p1_log" "$log"
      err_disp="$(extract_error_display "$log")"
      echo "[LoRA-P2] fc_stage1 seed=$tuning_seed → reusing Phase 1 result ($err_disp)"
      err_float="$(extract_error_float "$log")"
      record "phase2" "$scope" "$best_rank" "$best_lr" "$tuning_seed" "OK(reuse)" "$err_disp" "$log"
      if python3 -c "exit(0 if float('$err_float') < float('$best_p2_err') else 1)" 2>/dev/null; then
        best_p2_err="$err_float"; best_scope="$scope"
      fi
      continue
    fi
  fi

  echo
  echo "-------------------------------------------------------------"
  echo "[LoRA-P2] $DATASET  scope=$scope  rank=$best_rank  lr=$best_lr  seed=$tuning_seed"
  echo "-------------------------------------------------------------"

  cmd=($(build_cmd "$scope" "$best_rank" "$best_lr" "$tuning_seed"))
  if run_logged "$log" "${cmd[@]}"; then
    status="OK"
  else
    status="FAILED"
    fail_or_continue "[LoRA-P2] Failed: scope=$scope"
  fi

  err_disp="$(extract_error_display "$log")"
  record "phase2" "$scope" "$best_rank" "$best_lr" "$tuning_seed" "$status" "$err_disp" "$log"

  if [[ "$DRY_RUN" != "1" && "$status" == "OK" ]]; then
    err_float="$(extract_error_float "$log")"
    if python3 -c "exit(0 if float('$err_float') < float('$best_p2_err') else 1)" 2>/dev/null; then
      best_p2_err="$err_float"; best_scope="$scope"
    fi
  fi
done

if [[ "$DRY_RUN" = "1" ]]; then
  best_scope="${best_scope:-fc_stage1}"
fi

echo
echo "================================================================="
echo "[LoRA] Phase 2 complete."
echo "  Best scope: $best_scope  (error=${best_p2_err}%)"
echo "================================================================="

# =============================================================================
# PHASE 3 — Final evaluation  (best config, seeds 1–3)
# =============================================================================
echo
echo "================================================================="
echo "[LoRA] Phase 3: Final evaluation"
echo "  rank=$best_rank  lr=$best_lr  scope=$best_scope  seeds=${final_seeds[*]}"
echo "================================================================="

for seed in "${final_seeds[@]}"; do
  log="$run_root/phase3_final/${best_scope}_rank${best_rank}_lr${best_lr}_seed${seed}.log"

  # Seed 1 result may already exist from Phase 1 (fc_stage1) or Phase 2 (fc)
  if [[ "$DRY_RUN" != "1" && "$seed" == "$tuning_seed" ]]; then
    if [[ "$best_scope" == "fc_stage1" ]]; then
      src="$run_root/phase1_tuning/rank${best_rank}_lr${best_lr}.log"
    else
      src="$run_root/phase2_scope/${best_scope}_rank${best_rank}_lr${best_lr}.log"
    fi
    if [[ -f "$src" ]]; then
      cp "$src" "$log"
      err_disp="$(extract_error_display "$log")"
      echo "[LoRA-P3] seed=$seed → reusing earlier result ($err_disp)"
      record "phase3" "$best_scope" "$best_rank" "$best_lr" "$seed" "OK(reuse)" "$err_disp" "$log"
      continue
    fi
  fi

  echo
  echo "-------------------------------------------------------------"
  echo "[LoRA-P3] $DATASET  scope=$best_scope  rank=$best_rank  lr=$best_lr  seed=$seed"
  echo "-------------------------------------------------------------"

  cmd=($(build_cmd "$best_scope" "$best_rank" "$best_lr" "$seed"))
  if run_logged "$log" "${cmd[@]}"; then
    status="OK"
  else
    status="FAILED"
    fail_or_continue "[LoRA-P3] Failed: seed=$seed"
  fi

  err_disp="$(extract_error_display "$log")"
  record "phase3" "$best_scope" "$best_rank" "$best_lr" "$seed" "$status" "$err_disp" "$log"
done

# =============================================================================
# Final summary
# =============================================================================
{
  echo "================================================================="
  echo "LoRA MIXED-GRADUAL FINAL SUMMARY — $DATASET"
  echo "Finished: $(date)"
  echo "================================================================="
  echo
  echo "Best configuration selected automatically:"
  echo "  Rank:  $best_rank"
  echo "  LR:    $best_lr"
  echo "  Scope: $best_scope"
  echo
  echo "All results:"
  echo
  if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "$manifest"
  else
    cat "$manifest"
  fi
  echo
  echo "Manifest: $manifest"
  echo "Logs:     $run_root"
} | tee "$summary_file"

echo
echo "[LoRA] Complete. Summary: $summary_file"
