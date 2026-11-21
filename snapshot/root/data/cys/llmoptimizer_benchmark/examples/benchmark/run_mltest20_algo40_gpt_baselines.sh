#!/usr/bin/env bash
# gpt-4o-mini baselines on approved algo40 held-out tasks (40 dataset×algorithm pairs).
#
# Variants:
#   ptllm_smac  P5/H0
#   ptllm_tpe   P5/H0
#   ptllm_tpe   P16/H2  (TCS + Analysis Agent; isolated history dir)
#
# Usage:
#   bash examples/benchmark/run_mltest20_algo40_gpt_baselines.sh
#   STAGE=p5_smac|p5_tpe|p16_h2|all PARALLEL=4 bash ...
#   DRY_RUN=1 bash ...   # 2 trials, mock LLM, yeast only
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

TASKS_JSON="${TASKS_JSON:-benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_data/real_hpo_v2_mltest20_algo40}"
LOGDIR="${LOGDIR:-benchmark_data/real_hpo_v2_mltest20_algo40/logs}"
PYTHON="${PYTHON:-/root/miniconda3/envs/openbox_eval/bin/python}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-7200}"
PER_RUN_TIME_LIMIT="${PER_RUN_TIME_LIMIT:-300}"
DATASET_RETRIES="${DATASET_RETRIES:-3}"
PARALLEL="${PARALLEL:-4}"
STAGE="${STAGE:-all}"
EVAL_RESUME="${EVAL_RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="$LOGDIR/algo40_gpt_baselines_${TS}.log"
P16_HISTORY="${P16_HISTORY:-history_data/mltest20_algo40_p16_h2}"

mkdir -p "$LOGDIR" "$OUTPUT_ROOT" "$P16_HISTORY"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

log() { echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"; }

mapfile -t TASK_LINES < <("$PYTHON" - <<PY
import json
from pathlib import Path
tasks = json.loads(Path("$TASKS_JSON").read_text())
for t in tasks:
    print(f"{t['task_type']}|{t['dataset']}|{t['algorithm']}")
print(f"# n={len(tasks)}", flush=True)
PY
)
TASKS=()
for line in "${TASK_LINES[@]}"; do
  [[ "$line" == \#* ]] && continue
  TASKS+=("$line")
done

if [[ "$DRY_RUN" == "1" ]]; then
  OUTPUT_ROOT="benchmark_data/real_hpo_v2_mltest20_algo40_gpt_dryrun"
  N_TRIALS=2
  TIME_LIMIT=300
  TASKS=("cls|yeast|lightgbm")
  log "DRY_RUN output=$OUTPUT_ROOT n_trials=$N_TRIALS tasks=1"
fi

RESUME_FLAG=()
[[ "$EVAL_RESUME" == "1" ]] && RESUME_FLAG=(--resume)

run_gpt_job() {
  local opt="$1" prompt="$2" hybrid="$3" tt="$4" ds="$5" algo="$6"
  local hist_args=()
  [[ "$prompt" == "P16" ]] && hist_args=(--history-dir "$P16_HISTORY")
  local tag="${opt}__${prompt}__${hybrid}"
  local logf="$LOGDIR/gpt_${tag}_${ds}_${TS}.log"
  log "START $tag $tt/$ds/$algo"
  local mock_flag=()
  [[ "$DRY_RUN" == "1" ]] && mock_flag=(--llm-mock)
  "$PYTHON" examples/benchmark/run_toy_categorical_optimizers.py \
    --algorithm "$algo" \
    --n-trials "$N_TRIALS" \
    --seed "$SEED" \
    --skip-test-eval \
    --llm-backends gpt4o-mini \
    --output-dir "$OUTPUT_ROOT" \
    "${hist_args[@]}" \
    --all-datasets \
    --include-datasets "$ds" \
    --time-limit "$TIME_LIMIT" \
    --per-run-time-limit "$PER_RUN_TIME_LIMIT" \
    --dataset-retries "$DATASET_RETRIES" \
    --optimizer "$opt" \
    --prompt-variant "$prompt" \
    --hybrid-variant "$hybrid" \
    "${RESUME_FLAG[@]}" \
    "${mock_flag[@]}" \
    >>"$logf" 2>&1
  log "DONE $tag $tt/$ds/$algo"
}

run_stage() {
  local opt="$1" prompt="$2" hybrid="$3" label="$4"
  local jobs=()
  local t tt ds algo
  for t in "${TASKS[@]}"; do
    IFS='|' read -r tt ds algo <<<"$t"
    jobs+=("$tt|$ds|$algo")
  done
  log "$label: ${#jobs[@]} jobs (parallel=$PARALLEL)"
  local job
  for job in "${jobs[@]}"; do
    IFS='|' read -r tt ds algo <<<"$job"
    while [[ $(jobs -rp | wc -l) -ge $PARALLEL ]]; do sleep 10; done
    ( run_gpt_job "$opt" "$prompt" "$hybrid" "$tt" "$ds" "$algo" ) &
  done
  wait
  log "$label done"
}

log "=== algo40 gpt-4o-mini baselines STAGE=$STAGE n_trials=$N_TRIALS parallel=$PARALLEL ==="
log "output=$OUTPUT_ROOT tasks=${#TASKS[@]}"

case "$STAGE" in
  p5_smac) run_stage ptllm_smac P5 H0 "P5/H0 smac" ;;
  p5_tpe)  run_stage ptllm_tpe  P5 H0 "P5/H0 tpe" ;;
  p16_h2)  run_stage ptllm_tpe  P16 H2 "P16/H2 tpe" ;;
  all)
    run_stage ptllm_smac P5 H0 "P5/H0 smac"
    run_stage ptllm_tpe  P5 H0 "P5/H0 tpe"
    run_stage ptllm_tpe  P16 H2 "P16/H2 tpe"
    ;;
  *) log "unknown STAGE=$STAGE"; exit 1 ;;
esac

log "=== algo40 gpt baselines complete ==="
