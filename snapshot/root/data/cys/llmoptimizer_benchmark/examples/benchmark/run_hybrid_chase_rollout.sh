#!/bin/bash
# Hybrid chase rollout: ptllm_tpe / ptllm_smac / ptllm_cmaes on P7, P22, P27 (H2, eval8).
#
# Skips variants already complete (--resume). P22 ptllm_tpe and ptllm_smac already ran.
#
# Usage:
#   STAGE=all bash examples/benchmark/run_hybrid_chase_rollout.sh
#   STAGE=tpe bash examples/benchmark/run_hybrid_chase_rollout.sh
#   STAGE=smac bash examples/benchmark/run_hybrid_chase_rollout.sh
#   STAGE=cmaes bash examples/benchmark/run_hybrid_chase_rollout.sh
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
  --hybrid-variant H2
)

run_hybrid() {
  local optimizer="$1"
  local prompt="$2"
  echo "=== ${optimizer}: ${prompt} / H2 ==="
  "${BASE_RUNNER[@]}" --optimizer "$optimizer" --prompt-variant "$prompt"
}

run_tpe() {
  for P in P7 P22 P27; do
    run_hybrid ptllm_tpe "$P"
  done
}

run_smac() {
  for P in P7 P22 P27; do
    run_hybrid ptllm_smac "$P"
  done
}

run_cmaes() {
  for P in P7 P22 P27; do
    run_hybrid ptllm_cmaes "$P"
  done
}

case "$STAGE" in
  tpe)   run_tpe ;;
  smac)  run_smac ;;
  cmaes) run_cmaes ;;
  all)
    run_tpe
    run_smac
    run_cmaes
    ;;
  *)
    echo "Unknown STAGE=$STAGE (tpe | smac | cmaes | all)" >&2
    exit 1
    ;;
esac

echo ""
echo "[Done] hybrid chase STAGE=$STAGE on: $EVAL_DATASETS"
echo "Output: $OUTPUT_DIR"
