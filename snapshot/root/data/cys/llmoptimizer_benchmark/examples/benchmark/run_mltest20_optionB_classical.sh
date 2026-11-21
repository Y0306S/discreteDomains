#!/bin/bash
# Option B held-out eval: same 8 eval8 dataset names, ml_test_20 algorithms.
# Classical baselines (smac/tpe/cmaes) — CPU, resumable.
#
# Usage:
#   bash examples/benchmark/run_mltest20_optionB_classical.sh
#   DRY_RUN=1 bash examples/benchmark/run_mltest20_optionB_classical.sh
#
set -euo pipefail
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

PYTHON="${PYTHON:-/root/miniconda3/envs/openbox_eval/bin/python}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
DRY_RUN="${DRY_RUN:-0}"
TIME_LIMIT="${TIME_LIMIT:-7200}"
DATASET_RETRIES="${DATASET_RETRIES:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark_data/real_hpo_v2_mltest20_algoB}"
LOGDIR="${LOGDIR:-benchmark_data/real_hpo_v2/logs}"
PARALLEL="${PARALLEL:-3}"  # concurrent classical jobs

mkdir -p "$LOGDIR" "$OUTPUT_DIR"

# dataset:algorithm pairs from ml_test_20 (Option B)
TASKS=(
  "cls:yeast:lightgbm"
  "cls:car:extra_trees"
  "cls:PhishingWebsites:xgboost"
  "cls:segment:catboost"
  "rgs:boston:extra_trees"
  "rgs:tecator:extra_trees"
  "rgs:socmob:extra_trees"
  "rgs:pol:xgboost"
)

if [[ "$DRY_RUN" == "1" ]]; then
  N_TRIALS=4
  TIME_LIMIT=300
  TASKS=("cls:yeast:lightgbm" "rgs:boston:extra_trees")
fi

export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOGDIR/mltest20_algoB_classical_${TS}.log"
echo "[classical] output=$OUTPUT_DIR n_trials=$N_TRIALS parallel=$PARALLEL" | tee "$LOG"

run_one() {
  local task_type="$1" ds="$2" algo="$3" opt="$4"
  echo "[$(date -Is)] START $opt $task_type/$ds algo=$algo" | tee -a "$LOG"
  "$PYTHON" examples/benchmark/run_toy_categorical_optimizers.py \
    --algorithm "$algo" \
    --n-trials "$N_TRIALS" \
    --seed "$SEED" \
    --skip-test-eval \
    --optimizer "$opt" \
    --resume \
    --output-dir "$OUTPUT_DIR" \
    --all-datasets \
    --include-datasets "$ds" \
    --time-limit "$TIME_LIMIT" \
    --dataset-retries "$DATASET_RETRIES" \
    >>"$LOG" 2>&1
  echo "[$(date -Is)] DONE $opt $task_type/$ds" | tee -a "$LOG"
}

# Job queue: all classical × all tasks
JOBS=()
for T in "${TASKS[@]}"; do
  IFS=':' read -r TT DS ALGO <<<"$T"
  for OPT in smac tpe cmaes; do
    JOBS+=("$TT|$DS|$ALGO|$OPT")
  done
done

echo "[classical] ${#JOBS[@]} jobs" | tee -a "$LOG"

# Simple parallel pool
running=0
pids=()
for job in "${JOBS[@]}"; do
  IFS='|' read -r TT DS ALGO OPT <<<"$job"
  while [[ $(jobs -rp | wc -l) -ge $PARALLEL ]]; do
    sleep 5
  done
  run_one "$TT" "$DS" "$ALGO" "$OPT" &
  pids+=($!)
done
wait
echo "[classical] all done $(date -Is)" | tee -a "$LOG"
