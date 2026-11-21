#!/usr/bin/env bash
# A5 T50 X50 ablation: log bins only (L) + cat-neither history gate (C).
#
#   bash examples/benchmark/run_p5plus_a5_t50_x50lc_h2_eval.sh
#   PARALLEL=4 bash ...
#   DRY_RUN=1 bash ...
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
EVAL_RESUME="${EVAL_RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="$LOGDIR/algo40_p5plus_a5_t50_x50lc_h2_${TS}.log"

mkdir -p "$LOGDIR" "$OUTPUT_ROOT"
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
  OUTPUT_ROOT="benchmark_data/real_hpo_v2_mltest20_algo40_p5plus_a5_t50_x50lc_h2_dryrun"
  N_TRIALS=55
  TIME_LIMIT=300
  TASKS=("cls|yeast|lightgbm")
  log "DRY_RUN output=$OUTPUT_ROOT n_trials=$N_TRIALS"
fi

RESUME_FLAG=()
[[ "$EVAL_RESUME" == "1" ]] && RESUME_FLAG=(--resume)

PROMPTS=(P5plusA5T50X50L P5plusA5T50X50C)

run_job() {
  local prompt="$1" tt="$2" ds="$3" algo="$4"
  local tag="ptllm_tpe__${prompt}__H2"
  local logf="$LOGDIR/t50x50lc_h2_${tag}_${ds}_${algo}_${TS}.log"
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
    --all-datasets \
    --include-datasets "$ds" \
    --time-limit "$TIME_LIMIT" \
    --per-run-time-limit "$PER_RUN_TIME_LIMIT" \
    --dataset-retries "$DATASET_RETRIES" \
    --optimizer ptllm_tpe \
    --prompt-variant "$prompt" \
    --hybrid-variant H2 \
    "${RESUME_FLAG[@]}" \
    "${mock_flag[@]}" \
    >>"$logf" 2>&1
  log "DONE $tag $tt/$ds/$algo"
}

log "=== P5+ A5 T50 X50L + X50C ablation on ptllm_tpe H2 ==="
log "output=$OUTPUT_ROOT tasks=${#TASKS[@]} prompts=${PROMPTS[*]} seed=$SEED parallel=$PARALLEL"
log "compare vs: ptllm_tpe__P5plusA5T50X50__H2"

queue=()
for prompt in "${PROMPTS[@]}"; do
  for task in "${TASKS[@]}"; do
    IFS='|' read -r tt ds algo <<<"$task"
    queue+=("$prompt|$tt|$ds|$algo")
  done
done

active=0
for spec in "${queue[@]}"; do
  IFS='|' read -r prompt tt ds algo <<<"$spec"
  while (( active >= PARALLEL )); do
    wait -n 2>/dev/null || wait
    active=$((active - 1))
  done
  run_job "$prompt" "$tt" "$ds" "$algo" &
  active=$((active + 1))
done
wait

log "=== P5+ A5 T50 X50L/X50C H2 eval complete (${#queue[@]} jobs) ==="
