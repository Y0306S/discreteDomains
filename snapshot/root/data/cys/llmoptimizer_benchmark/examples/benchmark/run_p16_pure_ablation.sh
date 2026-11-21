#!/bin/bash
# Pure-LLM P16 ablation: ptllm + P16 (TCS + Analysis + Optimizer), no TPE hybrid.
#
# Compare against ptllm_tpe P16/H2 (70% LLM + 30% TPE) and ptllm P5 (pure LLM baseline).
# Variant dir: ptllm__P16
#
# Usage:
#   DRY_RUN=1 bash examples/benchmark/run_p16_pure_ablation.sh
#   bash examples/benchmark/run_p16_pure_ablation.sh
#   EVAL_DATASETS=yeast,boston bash examples/benchmark/run_p16_pure_ablation.sh
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

PYTHON="${PYTHON:-/root/miniconda3/envs/openbox_eval/bin/python}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
DRY_RUN="${DRY_RUN:-0}"
TIME_LIMIT="${TIME_LIMIT:-1800}"
DATASET_RETRIES="${DATASET_RETRIES:-3}"

if [[ "$DRY_RUN" == "1" ]]; then
  OUTPUT_DIR="benchmark_data/p16_pure_dryrun"
  N_TRIALS=2
  TIME_LIMIT=300
  EVAL_DATASETS="yeast,boston"
  echo "[DRY_RUN] ptllm P16 on $EVAL_DATASETS (mock LLM)"
else
  OUTPUT_DIR="${OUTPUT_DIR:-benchmark_data/real_hpo_v2}"
  EVAL_DATASETS="${EVAL_DATASETS:-yeast,car,PhishingWebsites,segment,boston,tecator,socmob,pol}"
fi

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
  --time-limit "$TIME_LIMIT"
  --dataset-retries "$DATASET_RETRIES"
  --optimizer ptllm
  --prompt-variant P16
  --hybrid-variant H0
)

if [[ "$DRY_RUN" == "1" ]]; then
  BASE_RUNNER+=( --llm-mock )
fi

echo "=== Pure P16 ablation: ptllm P16 (TCS two-call, 100% LLM, no TPE) ==="
echo "time_limit=${TIME_LIMIT}s dataset_retries=$DATASET_RETRIES datasets=$EVAL_DATASETS"
echo "Compare: ptllm__P16 vs ptllm_tpe__P16__H2 vs ptllm__P5"
echo "output=$OUTPUT_DIR"

IFS=',' read -ra DATASETS <<< "$EVAL_DATASETS"
for DS in "${DATASETS[@]}"; do
  DS="${DS// /}"
  [[ -z "$DS" ]] && continue
  echo ""
  echo "--- dataset: $DS ---"
  "${BASE_RUNNER[@]}" --include-datasets "$DS"
done

echo ""
echo "[Done] Pure P16 ablation on: $EVAL_DATASETS"
echo "Output: $OUTPUT_DIR"
echo "Regenerate: bash examples/benchmark/plot_all_valid_best.sh"
