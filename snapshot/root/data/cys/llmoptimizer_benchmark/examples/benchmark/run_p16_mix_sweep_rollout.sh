#!/bin/bash
# P16 (TCS + Analysis Agent) LLM-mix ablation: 20% / 30% / 40% LLM after warm-start.
#
# Compares against existing P16/H2 (70% LLM) in real_hpo_v2.
# H20=20%, H1=30%, H40=40% (probability mix, not phase prefix).
#
# Usage:
#   DRY_RUN=1 bash examples/benchmark/run_p16_mix_sweep_rollout.sh
#   bash examples/benchmark/run_p16_mix_sweep_rollout.sh
#   HYBRIDS=H20,H1 bash examples/benchmark/run_p16_mix_sweep_rollout.sh   # subset
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
HYBRIDS="${HYBRIDS:-H20,H1,H40}"

if [[ "$DRY_RUN" == "1" ]]; then
  OUTPUT_DIR="benchmark_data/p16_mix_sweep_dryrun"
  N_TRIALS=2
  TIME_LIMIT=300
  EVAL_DATASETS="yeast,boston"
  echo "[DRY_RUN] output=$OUTPUT_DIR n_trials=$N_TRIALS hybrids=$HYBRIDS datasets=$EVAL_DATASETS (mock LLM)"
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
)

if [[ "$DRY_RUN" == "1" ]]; then
  BASE_RUNNER+=( --llm-mock )
fi

declare -A HYBRID_LABEL=(
  [H20]="20% LLM"
  [H1]="30% LLM"
  [H40]="40% LLM"
  [H2]="70% LLM (baseline)"
)

echo "=== P16 mix sweep: TCS + Analysis Agent (gpt-4o-mini) ==="
echo "time_limit=${TIME_LIMIT}s dataset_retries=$DATASET_RETRIES hybrids=$HYBRIDS"
echo "datasets=$EVAL_DATASETS output=$OUTPUT_DIR"

IFS=',' read -ra HYBRID_LIST <<< "$HYBRIDS"
IFS=',' read -ra DATASETS <<< "$EVAL_DATASETS"

for HYBRID in "${HYBRID_LIST[@]}"; do
  HYBRID="${HYBRID// /}"
  [[ -z "$HYBRID" ]] && continue
  LABEL="${HYBRID_LABEL[$HYBRID]:-$HYBRID}"
  echo ""
  echo "=== P16/$HYBRID ($LABEL) ==="
  for DS in "${DATASETS[@]}"; do
    DS="${DS// /}"
    [[ -z "$DS" ]] && continue
    echo "--- dataset: $DS ---"
    "${BASE_RUNNER[@]}" --hybrid-variant "$HYBRID" --include-datasets "$DS"
  done
done

echo ""
echo "[Done] P16 mix sweep on: $EVAL_DATASETS hybrids=$HYBRIDS"
echo "Output: $OUTPUT_DIR"
echo "Regenerate leaderboard: bash examples/benchmark/plot_all_valid_best.sh"
