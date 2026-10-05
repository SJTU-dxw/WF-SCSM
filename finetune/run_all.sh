#!/usr/bin/env bash

set -uo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
K=10
WEBSITE_COUNT=200
MIN_TRACE_LENGTH=80
TEST_SLOT_STRATEGY=all
GPU_IDS=${GPU_IDS:-"0 1 2 3"}
GPU_MAX_USED_MIB=${GPU_MAX_USED_MIB:-512}
GPU_POLL_SECONDS=${GPU_POLL_SECONDS:-10}
read -r -a GPUS <<<"$GPU_IDS"
MODELS=(
  awf
  tmwf
  ares
  df
  tiktok
  varcnn
  rf
  countmamba
  scsm-single
  netclr
  swallow-origin
  swallow-single
  traverse
)

usage() {
  printf 'Usage: %s [--k K] [--website-count N] [--min-trace-length N] [--test-slot-strategy fixed|adaptive|ensemble|all]\n' "$0"
}

while (($#)); do
  case "$1" in
    --k)
      if (($# < 2)); then
        usage >&2
        exit 2
      fi
      K=$2
      shift 2
      ;;
    --test-slot-strategy)
      if (($# < 2)); then
        usage >&2
        exit 2
      fi
      TEST_SLOT_STRATEGY=$2
      shift 2
      ;;
    --website-count)
      if (($# < 2)); then
        usage >&2
        exit 2
      fi
      WEBSITE_COUNT=$2
      shift 2
      ;;
    --min-trace-length)
      if (($# < 2)); then
        usage >&2
        exit 2
      fi
      MIN_TRACE_LENGTH=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ $K =~ ^[1-9][0-9]*$ ]]; then
  printf '%s\n' '--k must be a positive integer' >&2
  exit 2
fi
if [[ -n $WEBSITE_COUNT ]] && { ! [[ $WEBSITE_COUNT =~ ^[0-9]+$ ]] || ((WEBSITE_COUNT < 2)); }; then
  printf '%s\n' '--website-count must be an integer of at least 2' >&2
  exit 2
fi
if ! [[ $MIN_TRACE_LENGTH =~ ^[1-9][0-9]*$ ]]; then
  printf '%s\n' '--min-trace-length must be a positive integer' >&2
  exit 2
fi
if [[ $TEST_SLOT_STRATEGY != fixed && $TEST_SLOT_STRATEGY != adaptive && $TEST_SLOT_STRATEGY != ensemble && $TEST_SLOT_STRATEGY != all ]]; then
  printf '%s\n' '--test-slot-strategy must be fixed, adaptive, ensemble, or all' >&2
  exit 2
fi
if ((${#GPUS[@]} == 0)); then
  printf '%s\n' 'GPU_IDS must contain at least one GPU index' >&2
  exit 2
fi
if ! [[ $GPU_MAX_USED_MIB =~ ^[0-9]+$ ]]; then
  printf '%s\n' 'GPU_MAX_USED_MIB must be a non-negative integer' >&2
  exit 2
fi
if ! [[ $GPU_POLL_SECONDS =~ ^[1-9][0-9]*$ ]]; then
  printf '%s\n' 'GPU_POLL_SECONDS must be a positive integer' >&2
  exit 2
fi

cd "$PROJECT_ROOT" || exit 1

WEEK_COUNT=$(
  "$PYTHON_BIN" - "$WEBSITE_COUNT" "$MIN_TRACE_LENGTH" <<'PY'
import contextlib
import sys

from finetune.main import parse_args
from finetune.data import prepare_split

argv = []
website_count = sys.argv[1]
if website_count:
    argv.extend(["--website-count", website_count])
argv.extend(["--min-trace-length", sys.argv[2]])
args = parse_args(argv)
with contextlib.redirect_stdout(sys.stderr):
    _, manifest = prepare_split(args)
print(len(manifest["weeks"]))
PY
)
if ! [[ $WEEK_COUNT =~ ^[1-9][0-9]*$ ]]; then
  printf 'Failed to determine week count: %s\n' "$WEEK_COUNT" >&2
  exit 1
fi

LOG_ROOT="$PROJECT_ROOT/finetune/logs/k${K}"
mkdir -p "$LOG_ROOT"

run_task() {
  local gpu=$1
  local model=$2
  local week=$3
  local use_random_slot=$4
  local use_freeze=$5
  local suffix=''
  local website_suffix=''
  local website_args=()
  local trace_length_args=(--min-trace-length "$MIN_TRACE_LENGTH")
  local random_slot_suffix=''
  local random_slot_args=()
  local test_slot_suffix=''
  local test_slot_args=()
  local freeze_args=()
  local variant=''

  if ((use_freeze)); then
    suffix='_freeze'
    freeze_args=(--freeze)
    variant=' freeze'
  fi
  if [[ -n $WEBSITE_COUNT ]]; then
    website_suffix="_sites${WEBSITE_COUNT}"
    website_args=(--website-count "$WEBSITE_COUNT")
  fi
  if [[ $model == scsm-single ]]; then
    if ((use_random_slot)); then
      random_slot_suffix='_randomslot'
      random_slot_args=(--random-slot)
      variant="${variant} random-slot"
    else
      random_slot_args=(--no-random-slot)
      variant="${variant} no-random-slot"
    fi
  fi
  if [[ $model == scsm-single ]]; then
    if [[ $TEST_SLOT_STRATEGY == all ]]; then
      test_slot_suffix='_alltest'
    elif [[ $TEST_SLOT_STRATEGY == ensemble ]]; then
      test_slot_suffix='_ensemblerange5test'
    else
      test_slot_suffix="_${TEST_SLOT_STRATEGY}test"
    fi
    test_slot_args=(--test-slot-strategy "$TEST_SLOT_STRATEGY")
  fi

  local output="$PROJECT_ROOT/finetune/results/$model/k${K}_seed0${website_suffix}_minlen${MIN_TRACE_LENGTH}${random_slot_suffix}${test_slot_suffix}${suffix}/week$(printf '%02d' "$week")"
  local log_dir="$LOG_ROOT${website_suffix}_minlen${MIN_TRACE_LENGTH}${random_slot_suffix}${test_slot_suffix}${suffix}/$model"
  local log="$log_dir/week$(printf '%02d' "$week").log"
  mkdir -p "$log_dir"

  local completed=0
  if [[ -f "$output/final.pt" && -f "$output/test_metrics.json" ]]; then
    completed=1
    if [[ $model == scsm-single && $TEST_SLOT_STRATEGY == all ]]; then
      for strategy in fixed adaptive ensemble; do
        [[ -f "$output/test_metrics_${strategy}.json" && -f "$output/predictions_${strategy}.npz" ]] || completed=0
      done
    fi
  fi
  if ((completed)); then
    printf '[cuda:%s] skip completed: %s week %s%s\n' "$gpu" "$model" "$week" "$variant"
    return 0
  fi
  if [[ -e "$output" ]]; then
    printf '[cuda:%s] incomplete output exists, refusing to overwrite: %s\n' "$gpu" "$output" >&2
    return 1
  fi

  printf '[cuda:%s] start: %s week %s%s\n' "$gpu" "$model" "$week" "$variant"
  if "$PYTHON_BIN" finetune/main.py \
      --model "$model" \
      --k "$K" \
      "${website_args[@]}" \
      "${trace_length_args[@]}" \
      "${random_slot_args[@]}" \
      "${test_slot_args[@]}" \
      --weeks "$week" \
      --device "cuda:$gpu" \
      "${freeze_args[@]}" >"$log" 2>&1; then
    printf '[cuda:%s] done: %s week %s%s\n' "$gpu" "$model" "$week" "$variant"
    return 0
  fi

  printf '[cuda:%s] failed: %s week %s%s (log: %s)\n' "$gpu" "$model" "$week" "$variant" "$log" >&2
  return 1
}

task_output_path() {
  local model=$1 week=$2 use_random_slot=$3 use_freeze=$4
  local website_suffix='' random_slot_suffix='' test_slot_suffix='' suffix=''
  [[ -n $WEBSITE_COUNT ]] && website_suffix="_sites${WEBSITE_COUNT}"
  ((use_random_slot)) && random_slot_suffix='_randomslot'
  ((use_freeze)) && suffix='_freeze'
  if [[ $model == scsm-single ]]; then
    if [[ $TEST_SLOT_STRATEGY == all ]]; then
      test_slot_suffix='_alltest'
    elif [[ $TEST_SLOT_STRATEGY == ensemble ]]; then
      test_slot_suffix='_ensemblerange5test'
    else
      test_slot_suffix="_${TEST_SLOT_STRATEGY}test"
    fi
  fi
  printf '%s/finetune/results/%s/k%s_seed0%s_minlen%s%s%s%s/week%02d' \
    "$PROJECT_ROOT" "$model" "$K" "$website_suffix" "$MIN_TRACE_LENGTH" \
    "$random_slot_suffix" "$test_slot_suffix" "$suffix" "$week"
}

task_is_completed() {
  local model=$1 week=$2 use_random_slot=$3 use_freeze=$4 output strategy
  output=$(task_output_path "$model" "$week" "$use_random_slot" "$use_freeze")
  [[ -f "$output/final.pt" && -f "$output/test_metrics.json" ]] || return 1
  if [[ $model == scsm-single && $TEST_SLOT_STRATEGY == all ]]; then
    for strategy in fixed adaptive ensemble; do
      [[ -f "$output/test_metrics_${strategy}.json" && -f "$output/predictions_${strategy}.npz" ]] || return 1
    done
  fi
}

PENDING_MODELS=()
PENDING_WEEKS=()
PENDING_RANDOM_SLOTS=()
PENDING_FREEZES=()
TOTAL_TASKS=0
SKIPPED_TASKS=0

queue_task() {
  local model=$1 week=$2 use_random_slot=$3 use_freeze=$4
  TOTAL_TASKS=$((TOTAL_TASKS + 1))
  if task_is_completed "$model" "$week" "$use_random_slot" "$use_freeze"; then
    printf '[scan] skip completed: %s week %s\n' "$model" "$week"
    SKIPPED_TASKS=$((SKIPPED_TASKS + 1))
    return
  fi
  PENDING_MODELS+=("$model")
  PENDING_WEEKS+=("$week")
  PENDING_RANDOM_SLOTS+=("$use_random_slot")
  PENDING_FREEZES+=("$use_freeze")
}

for model in "${MODELS[@]}"; do
  for ((week = 1; week <= WEEK_COUNT; week++)); do
    if [[ $model == scsm-single ]]; then
      for random_slot in 1 0; do queue_task "$model" "$week" "$random_slot" 0; done
    elif [[ $model == swallow-origin || $model == swallow-single ]]; then
      for freeze in 1 0; do queue_task "$model" "$week" 0 "$freeze"; done
    else
      freeze=0
      [[ $model == traverse ]] && freeze=1
      queue_task "$model" "$week" 0 "$freeze"
    fi
  done
done

gpu_is_available() {
  local gpu=$1 used
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null) || return 1
  used=${used%%$'\n'*}
  [[ $used =~ ^[0-9]+$ ]] && ((used <= GPU_MAX_USED_MIB))
}

PENDING_COUNT=${#PENDING_MODELS[@]}
printf 'Found %s tasks across %s models and %s weeks; skipped %s completed, assigning %s pending tasks across %s GPUs (k=%s)\n' \
  "$TOTAL_TASKS" "${#MODELS[@]}" "$WEEK_COUNT" "$SKIPPED_TASKS" "$PENDING_COUNT" "${#GPUS[@]}" "$K"

for gpu in "${GPUS[@]}"; do
  if ! nvidia-smi --id="$gpu" --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1; then
    printf 'Cannot query cuda:%s with nvidia-smi\n' "$gpu" >&2
    exit 1
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
      if kill -0 "$pid" 2>/dev/null; then
        continue
      fi
      task_status=0
      wait "$pid" || task_status=$?
      ((task_status == 0)) || FAILED_TASKS=$((FAILED_TASKS + 1))
      unset 'GPU_PIDS[$gpu]'
      RUNNING_TASKS=$((RUNNING_TASKS - 1))
      made_progress=1
    fi
    if ((NEXT_TASK < PENDING_COUNT)) && [[ -z ${GPU_PIDS[$gpu]:-} ]] && gpu_is_available "$gpu"; then
      run_task "$gpu" "${PENDING_MODELS[$NEXT_TASK]}" "${PENDING_WEEKS[$NEXT_TASK]}" \
        "${PENDING_RANDOM_SLOTS[$NEXT_TASK]}" "${PENDING_FREEZES[$NEXT_TASK]}" &
      GPU_PIDS[$gpu]=$!
      NEXT_TASK=$((NEXT_TASK + 1))
      RUNNING_TASKS=$((RUNNING_TASKS + 1))
      made_progress=1
      WAITING_REPORTED=0
    fi
  done
  if ((NEXT_TASK < PENDING_COUNT && RUNNING_TASKS == 0 && made_progress == 0 && WAITING_REPORTED == 0)); then
    printf 'All configured GPUs are occupied; waiting and polling every %s seconds\n' "$GPU_POLL_SECONDS"
    WAITING_REPORTED=1
  fi
  if ((NEXT_TASK < PENDING_COUNT || RUNNING_TASKS > 0)); then
    sleep "$GPU_POLL_SECONDS"
  fi
done

if ((FAILED_TASKS)); then
  printf 'Experiments finished with %s failed tasks\n' "$FAILED_TASKS" >&2
  exit 1
fi

printf '%s\n' 'All experiments completed successfully.'
