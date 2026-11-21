#!/bin/bash
# Run dl_test_15 eval for 5 selected models on a CPU box.
#
# Protocol:
#   - 10k steps, eval every 200
#   - 3 train_inner replicas per unique prediction config (not paper's 5)
#   - Merged prediction files (3 generation runs) are deduped by config before eval
#   - 4 parallel tasks (--cpu-profile) on ~96-core hosts
#
# Usage:
#   bash scripts/eval/run_dl_test_15_tier5_cpu.sh
#   SMOKE=1 bash scripts/eval/run_dl_test_15_tier5_cpu.sh   # 1 task smoke per model
#   MODELS="qwen_base_lora_sft_3_dl" bash scripts/eval/run_dl_test_15_tier5_cpu.sh

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$BASE_DIR"

TASKSET_PYTHON="${TASKSET_PYTHON:-/root/miniconda3/envs/taskset_eval/bin/python}"
EVAL_SCRIPT="scripts/eval/run_taskset_dl_eval_optimized.py"
TEST_DATA="data/dl_test_15_modified.jsonl"
BASELINE_CACHE="eval_runs/dl_test_15/task_baselines.json"
EVAL_DIR="eval_runs/dl_test_15"
LOG_DIR="${EVAL_DIR}/tier5_cpu_logs"
NUM_REPLICAS="${NUM_REPLICAS:-3}"
TASK_WORKERS="${TASK_WORKERS:-}"

mkdir -p "$LOG_DIR"

declare -A MODEL_PREDICTIONS=(
  ["qwen_base_lora_sft_3_dl"]="qwen_base_lora_sft_3_dl_test_predictions.jsonl"
  ["qwen_base_lora_sft_4_dl"]="qwen_base_lora_sft_4_dl_test_predictions.jsonl"
  ["qwen_base_lora_sft_5_dl"]="qwen_base_lora_sft_5_dl_test_predictions.jsonl"
  ["qwen_base_lora_sft_7_dl"]="qwen_base_lora_sft_7_dl_test_predictions.jsonl"
  ["gemini_3.1_pro_preview"]="gemini_3.1_pro_preview_test_predictions_run1.jsonl"
)

if [[ -n "${MODELS:-}" ]]; then
  IFS=',' read -ra SELECTED <<< "$MODELS"
else
  SELECTED=(
    qwen_base_lora_sft_3_dl
    qwen_base_lora_sft_4_dl
    qwen_base_lora_sft_5_dl
    qwen_base_lora_sft_7_dl
    gemini_3.1_pro_preview
  )
fi

echo "=== dl_test_15 tier-5 CPU eval ==="
echo "  base_dir=$BASE_DIR"
echo "  python=$TASKSET_PYTHON"
echo "  num_replicas=$NUM_REPLICAS"
echo "  task_workers=${TASK_WORKERS:-cpu-profile default}"
echo "  models=${SELECTED[*]}"
echo "  logs=$LOG_DIR"
echo

for model in "${SELECTED[@]}"; do
  pred_file="${MODEL_PREDICTIONS[$model]:-}"
  if [[ -z "$pred_file" ]]; then
    echo "Unknown model label: $model" >&2
    exit 1
  fi
  predictions="${EVAL_DIR}/${pred_file}"
  if [[ ! -f "$predictions" ]]; then
    echo "Missing predictions: $predictions" >&2
    exit 1
  fi

  work_dir="${EVAL_DIR}/tier5_cpu/${model}"
  output_json="${EVAL_DIR}/${model}_tier5_cpu_eval.json"
  log_file="${LOG_DIR}/${model}.log"

  extra_args=()
  if [[ "${SMOKE:-0}" == "1" ]]; then
    extra_args+=(--smoke-test --task-indices 0)
    output_json="${EVAL_DIR}/${model}_tier5_cpu_smoke.json"
    work_dir="${EVAL_DIR}/tier5_cpu_smoke/${model}"
  fi
  if [[ -n "$TASK_WORKERS" ]]; then
    extra_args+=(--task-workers "$TASK_WORKERS")
  fi

  if [[ "${SKIP_DONE:-1}" == "1" && -f "$output_json" && "${SMOKE:-0}" != "1" ]]; then
    echo "[$(date -Iseconds)] SKIP $model (output exists: $output_json)"
    echo
    continue
  fi

  echo "[$(date -Iseconds)] START $model"
  echo "  predictions=$predictions"
  echo "  output=$output_json"
  echo "  work_dir=$work_dir"

  "$TASKSET_PYTHON" "$EVAL_SCRIPT" \
    --test-data "$TEST_DATA" \
    --predictions "$predictions" \
    --model-label "$model" \
    --baseline-cache "$BASELINE_CACHE" \
    --skip-baseline-build \
    --num-replicas "$NUM_REPLICAS" \
    --cpu-profile \
    --work-dir "$work_dir" \
    --output-json "$output_json" \
    "${extra_args[@]}" \
    2>&1 | tee "$log_file"

  echo "[$(date -Iseconds)] DONE $model -> $output_json"
  echo
done

echo "All selected models finished."
