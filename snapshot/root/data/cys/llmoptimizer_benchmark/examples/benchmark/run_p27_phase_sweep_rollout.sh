#!/bin/bash
# P27 phase sweep: LLM prefix 5/10/15/20/25 trials then 100% TPE (80 total).
#
# Hybrids: H9(5), H5(10), H11(15), H3(20), H12(25)
#
# Usage:
#   bash examples/benchmark/run_p27_phase_sweep_rollout.sh
#   PHASES=5 bash examples/benchmark/run_p27_phase_sweep_rollout.sh   # single phase only
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

PYTHON="${PYTHON:-/root/miniconda3/envs/openbox_eval/bin/python}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-1800}"
DATASET_RETRIES="${DATASET_RETRIES:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark_data/real_hpo_v2}"
EVAL_DATASETS="${EVAL_DATASETS:-yeast,car,PhishingWebsites,segment,boston,tecator,socmob,pol}"

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
  --optimizer ptllm_tpe
  --prompt-variant P27
)

run_phase() {
  local llm_trials="$1"
  local hybrid_id="$2"
  echo ""
  echo "=== ptllm_tpe P27: ${llm_trials} LLM trials + $((N_TRIALS - llm_trials)) TPE (${hybrid_id}) ==="
  "${BASE_RUNNER[@]}" --hybrid-variant "$hybrid_id"
}

if [[ -n "${PHASES:-}" ]]; then
  case "$PHASES" in
    5)  run_phase 5  H9 ;;
    10) run_phase 10 H5 ;;
    15) run_phase 15 H11 ;;
    20) run_phase 20 H3 ;;
    25) run_phase 25 H12 ;;
    *) echo "Unknown PHASES=$PHASES (5|10|15|20|25)" >&2; exit 1 ;;
  esac
else
  run_phase 5  H9
  run_phase 10 H5
  run_phase 15 H11
  run_phase 20 H3
  run_phase 25 H12
fi

echo ""
echo "[Done] P27 phase sweep on: $EVAL_DATASETS"
echo "Variants: ptllm_tpe__P27__H9 H5 H11 H3 H12"
