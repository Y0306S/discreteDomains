#!/bin/bash
# Paper-faithful TCS + Analysis Agent eval on the 8-dataset eval set (gpt-4o-mini).
#
# Variant: ptllm_tpe P16 (TCS) + H2 (70% LLM / 30% TPE) — our production hybrid.
# Compare against existing ptllm_tpe P5/H2 results in real_hpo_v2.
#
# TCS uses 2 extra LLM calls per history turn, so we use a longer CASH time limit (1800s)
# and run each dataset in a fresh Python subprocess to avoid OpenBox thread exhaustion.
#
# Usage:
#   DRY_RUN=1 bash examples/benchmark/run_tcs_eval_rollout.sh
#   bash examples/benchmark/run_tcs_eval_rollout.sh
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
  OUTPUT_DIR="benchmark_data/tcs_eval_dryrun"
  N_TRIALS=2
  TIME_LIMIT=300
  EVAL_DATASETS="yeast,segment,boston"
  echo "[DRY_RUN] output=$OUTPUT_DIR n_trials=$N_TRIALS time_limit=${TIME_LIMIT}s datasets=$EVAL_DATASETS (mock LLM)"
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
  --optimizer ptllm_tpe
  --prompt-variant P16
  --hybrid-variant H2
)

if [[ "$DRY_RUN" == "1" ]]; then
  BASE_RUNNER+=( --llm-mock )
fi

echo "=== TCS eval: ptllm_tpe P16/H2 (TCS + Analysis Agent, gpt-4o-mini) ==="
echo "time_limit=${TIME_LIMIT}s dataset_retries=$DATASET_RETRIES (one subprocess per dataset)"

IFS=',' read -ra DATASETS <<< "$EVAL_DATASETS"
for DS in "${DATASETS[@]}"; do
  DS="${DS// /}"
  [[ -z "$DS" ]] && continue
  echo ""
  echo "=== TCS dataset: $DS ==="
  "${BASE_RUNNER[@]}" --include-datasets "$DS"
done

echo ""
echo "[Done] TCS eval rollout on: $EVAL_DATASETS"
echo "Output: $OUTPUT_DIR"
