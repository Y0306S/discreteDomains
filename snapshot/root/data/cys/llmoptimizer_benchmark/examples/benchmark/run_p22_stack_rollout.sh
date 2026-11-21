#!/bin/bash
# P22 stacked ablations + ptllm_smac P22/H2 on eval8 (80 trials).
#
# Variants:
#   P27  P22 + coverage_map
#   P28  P22 + best_so_far_trajectory
#   P29  P22 + playbook_run_context + coverage_map
#   ptllm_smac P22/H2
#
# Usage:
#   STAGE=all bash examples/benchmark/run_p22_stack_rollout.sh
#   STAGE=ptllm bash examples/benchmark/run_p22_stack_rollout.sh
#   STAGE=smac bash examples/benchmark/run_p22_stack_rollout.sh
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

PYTHON="${PYTHON:-/root/miniconda3/envs/openbox_eval/bin/python}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
STAGE="${STAGE:-all}"
TIME_LIMIT="${TIME_LIMIT:-1800}"
DATASET_RETRIES="${DATASET_RETRIES:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark_data/real_hpo_v2}"
EVAL_DATASETS="${EVAL_DATASETS:-yeast,car,PhishingWebsites,segment,boston,tecator,socmob,pol}"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

BASE_RUNNER=( "$PYTHON" examples/benchmark/run_toy_categorical_optimizers.py
  --algorithm random_forest
  --n-trials "$N_TRIALS"
  --seed "$SEED"
  --skip-test-eval
  --llm-backends gpt4o-mini
  --qwen-backend auto
  --resume
  --output-dir "$OUTPUT_DIR"
  --all-datasets
  --include-datasets "$EVAL_DATASETS"
  --time-limit "$TIME_LIMIT"
  --dataset-retries "$DATASET_RETRIES"
)

run_ptllm_stack() {
  for P in P27 P28 P29; do
    echo "=== ptllm stack: $P ==="
    "${BASE_RUNNER[@]}" --optimizer ptllm --prompt-variant "$P" --hybrid-variant H0
  done
}

run_smac_hybrid() {
  echo "=== ptllm_smac: P22 / H2 ==="
  "${BASE_RUNNER[@]}" --optimizer ptllm_smac --prompt-variant P22 --hybrid-variant H2
}

case "$STAGE" in
  ptllm) run_ptllm_stack ;;
  smac)  run_smac_hybrid ;;
  all)
    run_ptllm_stack
    run_smac_hybrid
    ;;
  *)
    echo "Unknown STAGE=$STAGE (ptllm | smac | all)" >&2
    exit 1
    ;;
esac

echo ""
echo "[Done] P22 stack rollout STAGE=$STAGE on: $EVAL_DATASETS"
echo "Output: $OUTPUT_DIR"
