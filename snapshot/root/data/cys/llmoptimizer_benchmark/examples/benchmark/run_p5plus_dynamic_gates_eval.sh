#!/usr/bin/env bash
# Live gpt-4o-mini sweep vs locked baseline ptllm_tpe__P5plusA5T50X50__H2.
# Part A: mix + UCB/PUCB + HDTS/HAP/D-UCB/DMAB/transient (prompt stays X50).
# Part B: region-insertion gates (hybrid stays H2).
# Part C: explore/exploit XOR (hybrid stays H2).
# Does not overwrite X50/H2. Does not stop the RC/RS retrieve bake-off.
#
#   bash examples/benchmark/run_p5plus_dynamic_gates_eval.sh
#   PARALLEL=16 bash ...
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
TIME_LIMIT="${TIME_LIMIT:-14400}"
PER_RUN_TIME_LIMIT="${PER_RUN_TIME_LIMIT:-300}"
DATASET_RETRIES="${DATASET_RETRIES:-3}"
PARALLEL="${PARALLEL:-16}"
EVAL_RESUME="${EVAL_RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="$LOGDIR/algo40_p5plus_dynamic_gates_${TS}.log"

# Capture hardware cores BEFORE exporting OMP_NUM_THREADS — nproc honors that
# env var and would otherwise report THREADS_PER_JOB (the 5-5 pinning bug).
NCORES="${NCORES:-$(nproc --all 2>/dev/null || nproc)}"
CORE_BASE="${CORE_BASE:-0}"
THREADS_PER_JOB="${THREADS_PER_JOB:-6}"
export OMP_NUM_THREADS="$THREADS_PER_JOB"
export OPENBLAS_NUM_THREADS="$THREADS_PER_JOB"
export MKL_NUM_THREADS="$THREADS_PER_JOB"
export NUMEXPR_NUM_THREADS="$THREADS_PER_JOB"
export VECLIB_MAXIMUM_THREADS="$THREADS_PER_JOB"
export LOKY_MAX_CPU_COUNT="$THREADS_PER_JOB"

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
  OUTPUT_ROOT="benchmark_data/real_hpo_v2_mltest20_algo40_dynamic_gates_dryrun"
  N_TRIALS=6
  TIME_LIMIT=300
  TASKS=("cls|yeast|lightgbm")
  log "DRY_RUN output=$OUTPUT_ROOT n_trials=$N_TRIALS"
fi

RESUME_FLAG=()
[[ "$EVAL_RESUME" == "1" ]] && RESUME_FLAG=(--resume)

# prompt|hybrid  — never X50+H2
SPECS=()
# Part A: mix + selectors, prompt fixed
for h in H60 H0 H40 HUCB HPUCB HPUCB_breakthrough HPUCB_arm HDTS HAP HDUCB HDMAB HTRAN; do
  SPECS+=("P5plusA5T50X50|$h")
done
# Part B: insertion gates, hybrid fixed
for p in P5plusA5TRdyX50 P5plusA5TRaceX50 P5plusA5TRdyFanovaX50 P5plusA5TRdyTurboX50 P5plusA5TRdyStrengthX50 P5plusA5TContrastX50 P5plusA5TStagX50; do
  SPECS+=("$p|H2")
done
# Part C: explore/exploit XOR, hybrid fixed
for p in P5plusA5T50XStag P5plusA5T50XCov P5plusA5T50XInt P5plusA5T50XRefine P5plusA5T50XHAP; do
  SPECS+=("$p|H2")
done

# CPU pinning: catboost/loky ignore OMP_NUM_THREADS and size their pools to the
# number of *available* cores. taskset shrinks that set so each job gets its own
# THREADS_PER_JOB cores instead of all 128 -> no cross-job pool oversubscription.
PIN_CORES="${PIN_CORES:-1}"
HAVE_TASKSET=0
command -v taskset >/dev/null 2>&1 && HAVE_TASKSET=1

run_job() {
  local prompt="$1" hybrid="$2" tt="$3" ds="$4" algo="$5" cores="${6:-}"
  local tag="ptllm_tpe__${prompt}__${hybrid}"
  local logf="$LOGDIR/dyn_${tag}_${ds}_${algo}_${TS}.log"
  log "START $tag $tt/$ds/$algo cores=${cores:-all}"
  local mock_flag=()
  [[ "$DRY_RUN" == "1" ]] && mock_flag=(--llm-mock)
  local pin=()
  if [[ "$PIN_CORES" == "1" && "$HAVE_TASKSET" == "1" && -n "$cores" ]]; then
    pin=(taskset -c "$cores")
  fi
  "${pin[@]}" "$PYTHON" examples/benchmark/run_toy_categorical_optimizers.py \
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
    --hybrid-variant "$hybrid" \
    "${RESUME_FLAG[@]}" \
    "${mock_flag[@]}" \
    >>"$logf" 2>&1
  log "DONE $tag $tt/$ds/$algo"
}

log "=== P5+ dynamic gates A+B+C vs X50/H2 gpt-4o-mini ==="
log "output=$OUTPUT_ROOT tasks=${#TASKS[@]} arms=${#SPECS[@]} seed=$SEED parallel=$PARALLEL threads_per_job=$THREADS_PER_JOB ncores=$NCORES pin=$PIN_CORES"
log "baseline (do not overwrite): ptllm_tpe__P5plusA5T50X50__H2"
log "retrieve RC/RS bake-off is independent and must keep running"

queue=()
for task in "${TASKS[@]}"; do
  IFS='|' read -r tt ds algo <<<"$task"
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r prompt hybrid <<<"$spec"
    queue+=("$prompt|$hybrid|$tt|$ds|$algo")
  done
done

log "jobs=${#queue[@]} (arms ${#SPECS[@]} x tasks ${#TASKS[@]})"

# Round-robin core blocks packed from CORE_BASE. nproc --all was captured
# before OMP_NUM_THREADS so this is the real 128, not 6.
block_for_slot() {
  local slot="$1"
  local lo=$(( CORE_BASE + slot * THREADS_PER_JOB ))
  local hi=$(( lo + THREADS_PER_JOB - 1 ))
  if (( lo > NCORES - 1 )); then lo=$(( NCORES - 1 )); fi
  if (( hi > NCORES - 1 )); then hi=$(( NCORES - 1 )); fi
  printf '%s-%s\n' "$lo" "$hi"
}

active=0
dispatch=0
for spec in "${queue[@]}"; do
  IFS='|' read -r prompt hybrid tt ds algo <<<"$spec"
  while (( active >= PARALLEL )); do
    wait -n 2>/dev/null || wait
    active=$((active - 1))
  done
  cores="$(block_for_slot $(( dispatch % PARALLEL )))"
  dispatch=$((dispatch + 1))
  run_job "$prompt" "$hybrid" "$tt" "$ds" "$algo" "$cores" &
  active=$((active + 1))
done
wait

log "=== dynamic gates A+B+C complete (${#queue[@]} jobs) ==="
