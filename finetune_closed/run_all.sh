#!/usr/bin/env bash

set -uo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
K=10
MIN_TRACE_LENGTH=1
TEST_SLOT_STRATEGY=all
GPU_IDS=${GPU_IDS:-"0 1 2 3"}
GPU_MAX_USED_MIB=${GPU_MAX_USED_MIB:-512}
GPU_POLL_SECONDS=${GPU_POLL_SECONDS:-10}
read -r -a GPUS <<<"$GPU_IDS"
DATASETS=(df r-precision w-t k-nn)
BASELINES=(awf tmwf ares df tiktok varcnn rf countmamba)
PRETRAINED=(scsm-single netclr swallow-origin swallow-single traverse)
PRETRAIN_SOURCES=(gtt swallow)

usage() {
  printf 'Usage: %s [--k K] [--dataset-name df|r-precision|w-t|k-nn] [--test-slot-strategy fixed|adaptive|ensemble|all] [--dry-run]\n' "$0"
}

SELECTED_DATASET=''
DRY_RUN=0
while (($#)); do
  case "$1" in
    --k) K=$2; shift 2 ;;
    --dataset-name) SELECTED_DATASET=$2; shift 2 ;;
    --test-slot-strategy) TEST_SLOT_STRATEGY=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! [[ $K =~ ^[1-9][0-9]*$ ]]; then
  printf '%s\n' '--k must be a positive integer' >&2; exit 2
fi
if [[ $TEST_SLOT_STRATEGY != fixed && $TEST_SLOT_STRATEGY != adaptive && $TEST_SLOT_STRATEGY != ensemble && $TEST_SLOT_STRATEGY != all ]]; then
  printf '%s\n' '--test-slot-strategy must be fixed, adaptive, ensemble, or all' >&2; exit 2
fi
if [[ -n $SELECTED_DATASET ]]; then
  valid=0
  for dataset in "${DATASETS[@]}"; do [[ $dataset == "$SELECTED_DATASET" ]] && valid=1; done
  if ((valid == 0)); then printf 'Unknown dataset: %s\n' "$SELECTED_DATASET" >&2; exit 2; fi
  DATASETS=("$SELECTED_DATASET")
fi
if ((${#GPUS[@]} == 0)); then printf '%s\n' 'GPU_IDS must contain at least one GPU index' >&2; exit 2; fi
if ! [[ $GPU_MAX_USED_MIB =~ ^[0-9]+$ ]]; then printf '%s\n' 'GPU_MAX_USED_MIB must be non-negative' >&2; exit 2; fi
if ! [[ $GPU_POLL_SECONDS =~ ^[1-9][0-9]*$ ]]; then printf '%s\n' 'GPU_POLL_SECONDS must be positive' >&2; exit 2; fi

cd "$PROJECT_ROOT" || exit 1

PENDING_DATASETS=()
PENDING_MODELS=()
PENDING_SOURCES=()
PENDING_RANDOM_SLOTS=()
PENDING_FREEZES=()
TOTAL_TASKS=0
SKIPPED_TASKS=0

task_suffixes() {
  local model=$1 random_slot=$2 freeze=$3
  RANDOM_SUFFIX=''
  TEST_SUFFIX=''
  FREEZE_SUFFIX=''
  ((random_slot)) && RANDOM_SUFFIX='_randomslot'
  ((freeze)) && FREEZE_SUFFIX='_freeze'
  if [[ $model == scsm-single ]]; then
    if [[ $TEST_SLOT_STRATEGY == all ]]; then TEST_SUFFIX='_alltest'
    elif [[ $TEST_SLOT_STRATEGY == ensemble ]]; then TEST_SUFFIX='_ensemblerange5test'
    else TEST_SUFFIX="_${TEST_SLOT_STRATEGY}test"
    fi
  fi
}

task_output_path() {
  local dataset=$1 model=$2 source=$3 random_slot=$4 freeze=$5
  task_suffixes "$model" "$random_slot" "$freeze"
  printf '%s/finetune_closed/results/%s/%s/%s/k%s_seed0_minlen%s%s%s%s/week01' \
    "$PROJECT_ROOT" "$dataset" "$source" "$model" "$K" "$MIN_TRACE_LENGTH" \
    "$RANDOM_SUFFIX" "$TEST_SUFFIX" "$FREEZE_SUFFIX"
}

task_is_completed() {
  local dataset=$1 model=$2 source=$3 random_slot=$4 freeze=$5 output strategy
  output=$(task_output_path "$dataset" "$model" "$source" "$random_slot" "$freeze")
  [[ -f "$output/final.pt" && -f "$output/test_metrics.json" ]] || return 1
  if [[ $model == scsm-single && $TEST_SLOT_STRATEGY == all ]]; then
    for strategy in fixed adaptive ensemble; do
      [[ -f "$output/test_metrics_${strategy}.json" && -f "$output/predictions_${strategy}.npz" ]] || return 1
    done
  fi
}

queue_task() {
  local dataset=$1 model=$2 source=$3 random_slot=$4 freeze=$5
  TOTAL_TASKS=$((TOTAL_TASKS + 1))
  if task_is_completed "$dataset" "$model" "$source" "$random_slot" "$freeze"; then
    printf '[scan] skip completed: %s %s %s\n' "$dataset" "$source" "$model"
    SKIPPED_TASKS=$((SKIPPED_TASKS + 1)); return
  fi
  PENDING_DATASETS+=("$dataset")
  PENDING_MODELS+=("$model")
  PENDING_SOURCES+=("$source")
  PENDING_RANDOM_SLOTS+=("$random_slot")
  PENDING_FREEZES+=("$freeze")
}

for dataset in "${DATASETS[@]}"; do
  for model in "${BASELINES[@]}"; do queue_task "$dataset" "$model" none 0 0; done
  for source in "${PRETRAIN_SOURCES[@]}"; do
    for model in "${PRETRAINED[@]}"; do
      if [[ $model == scsm-single ]]; then
        queue_task "$dataset" "$model" "$source" 1 0
        queue_task "$dataset" "$model" "$source" 0 0
      elif [[ $model == swallow-origin || $model == swallow-single ]]; then
        queue_task "$dataset" "$model" "$source" 0 1
        queue_task "$dataset" "$model" "$source" 0 0
      elif [[ $model == traverse ]]; then
        queue_task "$dataset" "$model" "$source" 0 1
      else
        queue_task "$dataset" "$model" "$source" 0 0
      fi
    done
  done
done

gpu_is_available() {
  local gpu=$1 used
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null) || return 1
  used=${used%%$'\n'*}
  [[ $used =~ ^[0-9]+$ ]] && ((used <= GPU_MAX_USED_MIB))
}

run_task() {
  local gpu=$1 dataset=$2 model=$3 source=$4 random_slot=$5 freeze=$6
  local output log_dir log variant=''
  local random_args=(--no-random-slot) freeze_args=() test_args=()
  ((random_slot)) && random_args=(--random-slot) && variant+=' random-slot'
  ((freeze)) && freeze_args=(--freeze) && variant+=' freeze'
  [[ $model == scsm-single ]] && test_args=(--test-slot-strategy "$TEST_SLOT_STRATEGY")
  output=$(task_output_path "$dataset" "$model" "$source" "$random_slot" "$freeze")
  task_suffixes "$model" "$random_slot" "$freeze"
  log_dir="$PROJECT_ROOT/finetune_closed/logs/$dataset/$source/$model/k${K}_minlen${MIN_TRACE_LENGTH}${RANDOM_SUFFIX}${TEST_SUFFIX}${FREEZE_SUFFIX}"
  log="$log_dir/week01.log"
  mkdir -p "$log_dir"
  if [[ -e $output ]]; then
    printf '[cuda:%s] incomplete output exists, refusing to overwrite: %s\n' "$gpu" "$output" >&2
    return 1
  fi
  printf '[cuda:%s] start: %s %s %s%s\n' "$gpu" "$dataset" "$source" "$model" "$variant"
  if "$PYTHON_BIN" finetune_closed/main.py \
      --dataset-name "$dataset" --pretrain-source "$source" --model "$model" \
      --k "$K" --min-trace-length "$MIN_TRACE_LENGTH" --weeks 1 --device "cuda:$gpu" \
      "${random_args[@]}" "${test_args[@]}" "${freeze_args[@]}" >"$log" 2>&1; then
    printf '[cuda:%s] done: %s %s %s%s\n' "$gpu" "$dataset" "$source" "$model" "$variant"; return 0
  fi
  printf '[cuda:%s] failed: %s %s %s%s (log: %s)\n' "$gpu" "$dataset" "$source" "$model" "$variant" "$log" >&2
  return 1
}

PENDING_COUNT=${#PENDING_MODELS[@]}
printf 'Found %s tasks; skipped %s completed, scheduling %s across %s GPUs (k=%s)\n' \
  "$TOTAL_TASKS" "$SKIPPED_TASKS" "$PENDING_COUNT" "${#GPUS[@]}" "$K"
if ((DRY_RUN)); then
  for ((index = 0; index < PENDING_COUNT; index++)); do
    printf '[dry-run] %s %s %s random-slot=%s freeze=%s\n' \
      "${PENDING_DATASETS[$index]}" "${PENDING_SOURCES[$index]}" \
      "${PENDING_MODELS[$index]}" "${PENDING_RANDOM_SLOTS[$index]}" \
      "${PENDING_FREEZES[$index]}"
  done
  exit 0
fi
for gpu in "${GPUS[@]}"; do
  if ! nvidia-smi --id="$gpu" --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1; then
    printf 'Cannot query cuda:%s with nvidia-smi\n' "$gpu" >&2; exit 1
  fi
done

declare -A GPU_PIDS=()
NEXT_TASK=0
RUNNING_TASKS=0
FAILED_TASKS=0
WAITING_REPORTED=0
while ((NEXT_TASK < PENDING_COUNT || RUNNING_TASKS > 0)); do
  made_progress=0
  for gpu in "${GPUS[@]}"; do
    if [[ -n ${GPU_PIDS[$gpu]:-} ]]; then
      pid=${GPU_PIDS[$gpu]}
      if kill -0 "$pid" 2>/dev/null; then continue; fi
      status=0; wait "$pid" || status=$?
      ((status == 0)) || FAILED_TASKS=$((FAILED_TASKS + 1))
      unset 'GPU_PIDS[$gpu]'; RUNNING_TASKS=$((RUNNING_TASKS - 1)); made_progress=1
    fi
    if ((NEXT_TASK < PENDING_COUNT)) && [[ -z ${GPU_PIDS[$gpu]:-} ]] && gpu_is_available "$gpu"; then
      run_task "$gpu" "${PENDING_DATASETS[$NEXT_TASK]}" "${PENDING_MODELS[$NEXT_TASK]}" \
        "${PENDING_SOURCES[$NEXT_TASK]}" "${PENDING_RANDOM_SLOTS[$NEXT_TASK]}" \
        "${PENDING_FREEZES[$NEXT_TASK]}" &
      GPU_PIDS[$gpu]=$!; NEXT_TASK=$((NEXT_TASK + 1)); RUNNING_TASKS=$((RUNNING_TASKS + 1))
      made_progress=1; WAITING_REPORTED=0
    fi
  done
  if ((NEXT_TASK < PENDING_COUNT && RUNNING_TASKS == 0 && made_progress == 0 && WAITING_REPORTED == 0)); then
    printf 'All configured GPUs are occupied; polling every %s seconds\n' "$GPU_POLL_SECONDS"; WAITING_REPORTED=1
  fi
  if ((NEXT_TASK < PENDING_COUNT || RUNNING_TASKS > 0)); then sleep "$GPU_POLL_SECONDS"; fi
done
if ((FAILED_TASKS)); then printf 'Experiments finished with %s failed tasks\n' "$FAILED_TASKS" >&2; exit 1; fi
printf '%s\n' 'All experiments completed successfully.'
