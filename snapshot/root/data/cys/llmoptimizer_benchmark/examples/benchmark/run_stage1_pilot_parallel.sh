#!/bin/bash
# Stage 1 pilot: 5 parallel workers (one per pilot dataset).
# Each worker sweeps prompts P0 P1 P2 P4 P5 P6 P7 P8 on ptllm only (pure LLM).
# Stage 2 later sweeps hybrids on ptllm_tpe with the winning prompt.
#
# History JSON is written per variant:
#   history_data/{cls|rgs}/{metric}/{dataset}/{variant}/random_forest.json
# so workers do not need to wait for baseline on the same dataset.
#
# Usage:
#   bash examples/benchmark/run_stage1_pilot_parallel.sh
#   WORKERS="cls covertype" bash examples/benchmark/run_stage1_pilot_parallel.sh
#   WORKERS="cls covertype;rgs boston" bash ...   # multiple workers
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

PYTHON="${PYTHON:-/root/miniconda3/envs/openbox_eval/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark_data/real_hpo_v2}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
LOG_DIR="${OUTPUT_DIR}/logs/stage1_pilot_$(date +%Y%m%d_%H%M%S)"
PROMPTS=(P0 P1 P2 P4 P5 P6 P7 P8)

mkdir -p "$LOG_DIR"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

run_dataset_worker() {
  local task_type="$1"
  local dataset="$2"
  local log="$LOG_DIR/${task_type}_${dataset}.log"

  {
    echo "[$(date -Is)] worker start ${task_type}/${dataset}"

    for prompt in "${PROMPTS[@]}"; do
      echo ""
      echo "[$(date -Is)] ${task_type}/${dataset} prompt=${prompt}"
      "$PYTHON" examples/benchmark/run_toy_categorical_optimizers.py \
        --algorithm random_forest \
        --n-trials "$N_TRIALS" \
        --seed "$SEED" \
        --skip-test-eval \
        --llm-backends gpt4o-mini \
        --qwen-backend auto \
        --resume \
        --output-dir "$OUTPUT_DIR" \
        --task-type "$task_type" \
        --dataset "$dataset" \
        --optimizer ptllm \
        --prompt-variant "$prompt" \
        --hybrid-variant H0
    done

    echo "[$(date -Is)] worker done ${task_type}/${dataset}"
  } >>"$log" 2>&1
}

echo "[Stage1 pilot parallel] logs=$LOG_DIR"
echo "Optimizer: ptllm only (prompt sweep; ptllm_tpe hybrids are Stage 2)"
echo "Prompts: ${PROMPTS[*]}"
echo "History: per-variant under history_data/.../{dataset}/{variant}/"

ALL_SPECS=("cls yeast" "cls adult" "cls covertype" "rgs boston" "rgs elevators")
SPECS=()
if [[ -n "${WORKERS:-}" ]]; then
  if [[ "$WORKERS" == *";"* ]]; then
    IFS=';' read -ra SPECS <<< "$WORKERS"
  else
    SPECS=("$WORKERS")
  fi
else
  SPECS=("${ALL_SPECS[@]}")
fi

PIDS=()
for spec in "${SPECS[@]}"; do
  set -- $spec
  run_dataset_worker "$1" "$2" &
  PIDS+=($!)
  echo "  launched worker ${1}/${2} pid=$!"
done

echo ""
echo "Workers: ${PIDS[*]}"
echo "Monitor: tail -f $LOG_DIR/*.log"

wait "${PIDS[@]}" || true
echo "[Done] stage1 pilot parallel — see $LOG_DIR"
