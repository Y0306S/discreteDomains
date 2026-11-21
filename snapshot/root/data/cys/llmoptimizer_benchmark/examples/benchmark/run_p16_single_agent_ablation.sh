#!/bin/bash
# Ablation: P16 (two-call TCS) vs P30 (single-call TCS + analysis + JSON).
#
# Both use the same hybrid (default H2 = 70% LLM). P16/H2 results already exist;
# this script runs only P30/H2 on eval8 unless PROMPTS includes P16.
#
# Usage:
#   DRY_RUN=1 bash examples/benchmark/run_p16_single_agent_ablation.sh
#   bash examples/benchmark/run_p16_single_agent_ablation.sh
#   HYBRID=H1 PROMPTS=P30 bash examples/benchmark/run_p16_single_agent_ablation.sh
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
HYBRID="${HYBRID:-H2}"
PROMPTS="${PROMPTS:-P30}"

if [[ "$DRY_RUN" == "1" ]]; then
  OUTPUT_DIR="benchmark_data/p16_single_agent_dryrun"
  N_TRIALS=2
  TIME_LIMIT=300
  EVAL_DATASETS="yeast,boston"
  echo "[DRY_RUN] prompts=$PROMPTS hybrid=$HYBRID datasets=$EVAL_DATASETS (mock LLM)"
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
  --hybrid-variant "$HYBRID"
)

if [[ "$DRY_RUN" == "1" ]]; then
  BASE_RUNNER+=( --llm-mock )
fi

declare -A PROMPT_LABEL=(
  [P16]="two-call (Analysis + Optimizer)"
  [P30]="single-call (analysis + JSON together)"
)

echo "=== TCS agent ablation: prompts=$PROMPTS hybrid=$HYBRID ==="
echo "time_limit=${TIME_LIMIT}s datasets=$EVAL_DATASETS output=$OUTPUT_DIR"

IFS=',' read -ra PROMPT_LIST <<< "$PROMPTS"
IFS=',' read -ra DATASETS <<< "$EVAL_DATASETS"

for PROMPT in "${PROMPT_LIST[@]}"; do
  PROMPT="${PROMPT// /}"
  [[ -z "$PROMPT" ]] && continue
  LABEL="${PROMPT_LABEL[$PROMPT]:-$PROMPT}"
  echo ""
  echo "=== $PROMPT/$HYBRID ($LABEL) ==="
  for DS in "${DATASETS[@]}"; do
    DS="${DS// /}"
    [[ -z "$DS" ]] && continue
    echo "--- dataset: $DS ---"
    "${BASE_RUNNER[@]}" --prompt-variant "$PROMPT" --include-datasets "$DS"
  done
done

echo ""
echo "[Done] TCS agent ablation prompts=$PROMPTS hybrid=$HYBRID"
echo "Compare: ptllm_tpe__P16__${HYBRID} vs ptllm_tpe__P30__${HYBRID}"
echo "Regenerate: bash examples/benchmark/plot_all_valid_best.sh"
