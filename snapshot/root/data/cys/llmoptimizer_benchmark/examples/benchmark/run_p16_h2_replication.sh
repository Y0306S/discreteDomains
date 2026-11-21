#!/bin/bash
# Re-run ptllm_tpe P16/H2 on eval8 without touching the original real_hpo_v2 runs.
#
# Original (seed=1): benchmark_data/real_hpo_v2/.../ptllm_tpe__P16__H2
# Replication:      benchmark_data/real_hpo_v2_p16_rep2/.../ptllm_tpe__P16__H2
# Uses isolated history under history_data/p16_h2_rep2/ so BO state does not leak.
#
# Usage:
#   DRY_RUN=1 bash examples/benchmark/run_p16_h2_replication.sh
#   bash examples/benchmark/run_p16_h2_replication.sh
#   SEED=1 bash examples/benchmark/run_p16_h2_replication.sh   # same seed, LLM stochasticity only
#   SEED=2 bash examples/benchmark/run_p16_h2_replication.sh   # default: different BO seed too
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

PYTHON="${PYTHON:-/root/miniconda3/envs/openbox_eval/bin/python}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-2}"
DRY_RUN="${DRY_RUN:-0}"
TIME_LIMIT="${TIME_LIMIT:-1800}"
DATASET_RETRIES="${DATASET_RETRIES:-3}"
REP_TAG="${REP_TAG:-rep2}"

if [[ "$DRY_RUN" == "1" ]]; then
  OUTPUT_DIR="benchmark_data/p16_h2_${REP_TAG}_dryrun"
  HISTORY_DIR="history_data/p16_h2_${REP_TAG}_dryrun"
  N_TRIALS=2
  TIME_LIMIT=300
  EVAL_DATASETS="yeast,boston"
  echo "[DRY_RUN] seed=$SEED output=$OUTPUT_DIR"
else
  OUTPUT_DIR="${OUTPUT_DIR:-benchmark_data/real_hpo_v2_p16_${REP_TAG}}"
  HISTORY_DIR="${HISTORY_DIR:-history_data/p16_h2_${REP_TAG}}"
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
  --output-dir "$OUTPUT_DIR"
  --history-dir "$HISTORY_DIR"
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

echo "=== P16/H2 replication ($REP_TAG) — original runs preserved in real_hpo_v2 ==="
echo "seed=$SEED n_trials=$N_TRIALS time_limit=${TIME_LIMIT}s"
echo "output=$OUTPUT_DIR"
echo "history=$HISTORY_DIR"
echo "datasets=$EVAL_DATASETS"
echo "Compare when done: python3 examples/benchmark/plot_p16_h2_replication.py --rep-tag $REP_TAG"

IFS=',' read -ra DATASETS <<< "$EVAL_DATASETS"
for DS in "${DATASETS[@]}"; do
  DS="${DS// /}"
  [[ -z "$DS" ]] && continue
  echo ""
  echo "--- dataset: $DS ---"
  "${BASE_RUNNER[@]}" --include-datasets "$DS"
done

echo ""
echo "[Done] P16/H2 replication ($REP_TAG) seed=$SEED on: $EVAL_DATASETS"
echo "Original untouched: benchmark_data/real_hpo_v2"
echo "Replication output: $OUTPUT_DIR"
