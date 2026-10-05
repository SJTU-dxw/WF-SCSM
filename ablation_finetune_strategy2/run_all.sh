#!/usr/bin/env bash

set -uo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
K=10
WEBSITE_COUNT=200
MIN_TRACE_LENGTH=80
INITIALIZATION=scratch
TEST_SLOT_STRATEGY=all
GPU_IDS=${GPU_IDS:-"0 1 2 3"}
GPU_MAX_USED_MIB=${GPU_MAX_USED_MIB:-512}
GPU_POLL_SECONDS=${GPU_POLL_SECONDS:-10}
read -r -a GPUS <<<"$GPU_IDS"

usage() {
  printf 'Usage: %s [--k K] [--website-count N] [--min-trace-length N] [--initialization scratch|pretrained|all] [--test-slot-strategy fixed|adaptive|ensemble|all]\n' "$0"
}

while (($#)); do
  case "$1" in
    --k|--website-count|--min-trace-length|--initialization|--test-slot-strategy)
      (($# >= 2)) || { usage >&2; exit 2; }
      case "$1" in
        --k) K=$2 ;;
        --website-count) WEBSITE_COUNT=$2 ;;
        --min-trace-length) MIN_TRACE_LENGTH=$2 ;;
        --initialization) INITIALIZATION=$2 ;;
        --test-slot-strategy) TEST_SLOT_STRATEGY=$2 ;;
      esac
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ $K =~ ^[1-9][0-9]*$ ]] || { printf '%s\n' '--k must be a positive integer' >&2; exit 2; }
[[ $WEBSITE_COUNT =~ ^[0-9]+$ ]] && ((WEBSITE_COUNT >= 2)) || {
  printf '%s\n' '--website-count must be an integer of at least 2' >&2; exit 2;
}
[[ $MIN_TRACE_LENGTH =~ ^[1-9][0-9]*$ ]] || {
  printf '%s\n' '--min-trace-length must be a positive integer' >&2; exit 2;
}
[[ $INITIALIZATION == scratch || $INITIALIZATION == pretrained || $INITIALIZATION == all ]] || {
  printf '%s\n' '--initialization must be scratch, pretrained, or all' >&2; exit 2;
}
[[ $TEST_SLOT_STRATEGY == fixed || $TEST_SLOT_STRATEGY == adaptive || $TEST_SLOT_STRATEGY == ensemble || $TEST_SLOT_STRATEGY == all ]] || {
  printf '%s\n' '--test-slot-strategy must be fixed, adaptive, ensemble, or all' >&2; exit 2;
}
((${#GPUS[@]})) || { printf '%s\n' 'GPU_IDS must contain at least one GPU index' >&2; exit 2; }

cd "$PROJECT_ROOT" || exit 1
WEEK_COUNT=$(
  "$PYTHON_BIN" - "$WEBSITE_COUNT" "$MIN_TRACE_LENGTH" <<'PY'
import contextlib
import sys
from finetune.main import parse_args
from finetune.data import prepare_split
args = parse_args(["--website-count", sys.argv[1], "--min-trace-length", sys.argv[2]])
with contextlib.redirect_stdout(sys.stderr):
    _, manifest = prepare_split(args)
print(len(manifest["weeks"]))
PY
)
[[ $WEEK_COUNT =~ ^[1-9][0-9]*$ ]] || { printf 'Failed to determine week count: %s\n' "$WEEK_COUNT" >&2; exit 1; }

if [[ $INITIALIZATION == all ]]; then
  INITIALIZATIONS=(pretrained scratch)
else
  INITIALIZATIONS=("$INITIALIZATION")
fi

TASK_INITIALIZATIONS=()
TASK_WEEKS=()
TASK_SCALINGS=()
for initialization in "${INITIALIZATIONS[@]}"; do
  for ((week = 1; week <= WEEK_COUNT; week++)); do
    for scaling in 1 0; do
      random_suffix=''
      ((scaling)) && random_suffix='_randomslot'
      [[ $TEST_SLOT_STRATEGY == all ]] && test_suffix='_alltest' || test_suffix="_${TEST_SLOT_STRATEGY}test"
      output="$PROJECT_ROOT/ablation_finetune_strategy2/results/$initialization/scsm-single/k${K}_seed0_sites${WEBSITE_COUNT}_minlen${MIN_TRACE_LENGTH}${random_suffix}${test_suffix}/week$(printf '%02d' "$week")"
      completed=0
      if [[ -f $output/final.pt && -f $output/test_metrics.json ]]; then
        completed=1
        if [[ $TEST_SLOT_STRATEGY == all ]]; then
          for strategy in fixed adaptive ensemble; do
            [[ -f $output/test_metrics_${strategy}.json && -f $output/predictions_${strategy}.npz ]] || completed=0
          done
        fi
      fi
      if ((completed)); then
        printf '[scan] skip completed: %s week %s scaling=%s\n' "$initialization" "$week" "$scaling"
      else
        if [[ -e $output ]]; then
          archive="$PROJECT_ROOT/ablation_finetune_strategy2/failed_runs/$(date +%Y%m%d-%H%M%S)/${output#"$PROJECT_ROOT/ablation_finetune_strategy2/results/"}"
          mkdir -p "$(dirname "$archive")"
          mv "$output" "$archive"
          printf '[scan] archived incomplete output: %s -> %s\n' "$output" "$archive"
        fi
        TASK_INITIALIZATIONS+=("$initialization")
        TASK_WEEKS+=("$week")
        TASK_SCALINGS+=("$scaling")
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
  local gpu=$1 initialization=$2 week=$3 scaling=$4
  local scaling_arg=--no-random-slot scaling_name=no-scaling
  if ((scaling)); then scaling_arg=--random-slot; scaling_name=scaling; fi
  local log_dir="$PROJECT_ROOT/ablation_finetune_strategy2/logs/$initialization/k${K}_sites${WEBSITE_COUNT}_minlen${MIN_TRACE_LENGTH}_${scaling_name}"
  local log="$log_dir/week$(printf '%02d' "$week").log"
  mkdir -p "$log_dir"
  printf '[cuda:%s] start: %s week %s %s\n' "$gpu" "$initialization" "$week" "$scaling_name"
  "$PYTHON_BIN" ablation_finetune_strategy2/main.py \
    --initialization "$initialization" --model scsm-single --k "$K" \
    --website-count "$WEBSITE_COUNT" --min-trace-length "$MIN_TRACE_LENGTH" \
    --weeks "$week" "$scaling_arg" --test-slot-strategy "$TEST_SLOT_STRATEGY" \
    --device "cuda:$gpu" >"$log" 2>&1
}

for gpu in "${GPUS[@]}"; do
  nvidia-smi --id="$gpu" --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1 || {
    printf 'Cannot query cuda:%s with nvidia-smi\n' "$gpu" >&2; exit 1;
  }
done

declare -A GPU_PIDS=()
NEXT_TASK=0
RUNNING=0
FAILED=0
TASK_COUNT=${#TASK_WEEKS[@]}
while ((NEXT_TASK < TASK_COUNT || RUNNING > 0)); do
  for gpu in "${GPUS[@]}"; do
    if [[ -n ${GPU_PIDS[$gpu]:-} ]] && ! kill -0 "${GPU_PIDS[$gpu]}" 2>/dev/null; then
      wait "${GPU_PIDS[$gpu]}" || FAILED=$((FAILED + 1))
      unset 'GPU_PIDS[$gpu]'
      RUNNING=$((RUNNING - 1))
    fi
    if ((NEXT_TASK < TASK_COUNT)) && [[ -z ${GPU_PIDS[$gpu]:-} ]] && gpu_is_available "$gpu"; then
      run_task "$gpu" "${TASK_INITIALIZATIONS[$NEXT_TASK]}" "${TASK_WEEKS[$NEXT_TASK]}" "${TASK_SCALINGS[$NEXT_TASK]}" &
      GPU_PIDS[$gpu]=$!
      NEXT_TASK=$((NEXT_TASK + 1))
      RUNNING=$((RUNNING + 1))
    fi
  done
  ((NEXT_TASK < TASK_COUNT || RUNNING > 0)) && sleep "$GPU_POLL_SECONDS"
done

((FAILED == 0)) || { printf '%s tasks failed\n' "$FAILED" >&2; exit 1; }
printf '%s\n' 'All GTT fine-tuning-strategy ablations completed successfully.'
