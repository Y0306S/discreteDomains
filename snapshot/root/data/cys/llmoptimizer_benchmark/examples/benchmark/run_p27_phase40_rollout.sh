#!/bin/bash
# P27 phase hybrid: 40 LLM trials then 40 TPE trials (H8, eval8, 80 total).
#
# Schedule (iter_idx):
#   0-3   llm_warm_start
#   4-39  llm with history
#   40-79 100% TPE
#
# Usage:
#   bash examples/benchmark/run_p27_phase40_rollout.sh
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

echo "=== ptllm_tpe P27 / H8 (40 LLM + 40 TPE) ==="
"$PYTHON" examples/benchmark/run_toy_categorical_optimizers.py \
  --algorithm random_forest \
  --n-trials "$N_TRIALS" \
  --seed "$SEED" \
  --skip-test-eval \
  --llm-backends gpt4o-mini \
  --qwen-backend auto \
  --resume \
  --output-dir "$OUTPUT_DIR" \
  --all-datasets \
  --include-datasets "$EVAL_DATASETS" \
  --time-limit "$TIME_LIMIT" \
  --dataset-retries "$DATASET_RETRIES" \
  --optimizer ptllm_tpe \
  --prompt-variant P27 \
  --hybrid-variant H8

echo ""
echo "[Done] ptllm_tpe P27/H8 on: $EVAL_DATASETS"
echo "Variant dir: ptllm_tpe__P27__H8"
