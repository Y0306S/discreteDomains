#!/usr/bin/env bash
# PT4Opt external baselines on the ml_ds_test_19 held-out split.
#
#   task list : benchmark_data/ml_test_20_fixed/tasks_132.json
#               (19 datasets x 7 algorithms = 132 tasks, from
#                LLMOptimizer/bolt_qwen9b_finetune/data/ml_ds_test_19.jsonl)
#   protocol  : the study protocol -- 80 trials, seed 1, time_limit 10800,
#               per_run_time_limit 300, holdout, matching
#               real_hpo_v2_mltest20_algo40_p5plus_grpo_rsmix_v12_..._ckpt32
#
# This script does NOT auto-run at high concurrency. PARALLEL defaults to 4 and
# MAX_LOAD guards against piling onto a machine that is already busy with other
# experiments. Raise PARALLEL deliberately.
#
# Examples
#   # what would run, no execution
#   DRY_RUN=1 bash examples/benchmark/run_ml_ds_test_19_baselines.sh
#   # cheap datasets only, one optimizer, small budget -- pipeline validation
#   OPTIMIZERS=tpe N_TRIALS=8 ONLY_DATASETS="socmob credit-g" \
#     bash examples/benchmark/run_ml_ds_test_19_baselines.sh
#   # the real thing
#   PARALLEL=8 bash examples/benchmark/run_ml_ds_test_19_baselines.sh
set -uo pipefail

if [[ -d /root/data/cys/llmoptimizer_benchmark ]]; then
  REPO="${REPO:-/root/data/cys/llmoptimizer_benchmark}"
else
  REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
fi
cd "$REPO"

TASKS_JSON="${TASKS_JSON:-benchmark_data/ml_test_20_fixed/tasks_132.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_data/ml_test_20_fixed_baselines}"
HISTORY_DIR="${HISTORY_DIR:-$REPO/history_data}"
LOGDIR="${LOGDIR:-/root/data/cys/baseline/logs/ml_test_20_fixed}"

# Baselines that need no extra training or prerequisites.
OPTIMIZERS="${OPTIMIZERS:-random_search tpe lolbo}"

# Study protocol.
N_TRIALS="${N_TRIALS:-100}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-10800}"
PER_RUN_TIME_LIMIT="${PER_RUN_TIME_LIMIT:-300}"
DATASET_RETRIES="${DATASET_RETRIES:-1}"

# Scheduling guards.
PARALLEL="${PARALLEL:-4}"
MAX_LOAD="${MAX_LOAD:-0}"        # 0 disables the check; e.g. 100 waits below load 100
POLL_SEC="${POLL_SEC:-60}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
ONLY_DATASETS="${ONLY_DATASETS:-}"   # space-separated allow-list
# 1 = taskset each job to a single logical CPU so sklearn n_jobs / CatBoost
# cannot fan out to min(16, nproc) and fight TaskSet BO on a shared box.
PIN_CPUS="${PIN_CPUS:-0}"

# LOL-BO needs gpytorch/botorch, which openbox_eval lacks; bl_lolbo_harness is a
# clone of openbox_eval plus those two, so the oracle stack is identical.
if [[ -x /root/data/cys/envs/openbox_qwen_eval/bin/python ]]; then
  PYTHON_CLASS="${PYTHON_CLASS:-/root/data/cys/envs/openbox_qwen_eval/bin/python}"
else
  PYTHON_CLASS="${PYTHON_CLASS:-/root/data/cys/blockdata_archive/miniconda3/envs/openbox_eval/bin/python}"
fi
PYTHON_LOLBO="${PYTHON_LOLBO:-/root/data/cys/baseline/envs/bl_lolbo_harness/bin/python}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOGDIR" "$OUTPUT_ROOT"
MASTER_LOG="$LOGDIR/baselines_${TS}.log"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

log() { echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"; }

mapfile -t TASK_LINES < <("$PYTHON_CLASS" - <<PY
import json
from pathlib import Path
for t in json.loads(Path("$TASKS_JSON").read_text()):
    print(f"{t['task_type']}|{t['dataset']}|{t['algorithm']}|{t['metric']}")
PY
)

TASKS=()
for line in "${TASK_LINES[@]}"; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  if [[ -n "$ONLY_DATASETS" ]]; then
    ds="$(cut -d'|' -f2 <<<"$line")"
    [[ " $ONLY_DATASETS " == *" $ds "* ]] || continue
  fi
  TASKS+=("$line")
done

log "=== ml_ds_test_19 baselines START ==="
log "tasks=${#TASKS[@]} optimizers=[$OPTIMIZERS] n_trials=$N_TRIALS seed=$SEED parallel=$PARALLEL pin_cpus=$PIN_CPUS"
log "output_root=$OUTPUT_ROOT logdir=$LOGDIR python=$PYTHON_CLASS"
[[ "$MAX_LOAD" != "0" ]] && log "will hold while 1-min load average exceeds $MAX_LOAD"

wait_for_load() {
  [[ "$MAX_LOAD" == "0" ]] && return 0
  while :; do
    local l; l=$(awk '{printf "%d", $1}' /proc/loadavg)
    (( l < MAX_LOAD )) && return 0
    log "load $l >= $MAX_LOAD, holding ${POLL_SEC}s"
    sleep "$POLL_SEC"
  done
}

RESUME_FLAG=()
[[ "$RESUME" == "1" ]] && RESUME_FLAG=(--resume)
NPROC="$(nproc)"

# Launch via an explicit subshell (not a backgrounded function). Bash function
# backgrounds on this host were observed pinning every job to CPU 0 even though
# n_launched advanced in the parent — all 16 RS workers landed on one core.
run_one() {
  local cpu="$1"
  local py="$2"
  local opt="$3"
  local tt="$4"
  local ds="$5"
  local algo="$6"
  local metric="$7"
  local jlog="$8"
  echo "[$(date -Is)] START $opt $tt/$ds/$algo pin=taskset -c $cpu" >>"$MASTER_LOG"
  if [[ "$PIN_CPUS" == "1" ]]; then
    taskset -c "$cpu" "$py" examples/benchmark/run_toy_categorical_optimizers.py \
      --optimizer "$opt" \
      --task-type "$tt" \
      --algorithm "$algo" \
      --metric "$metric" \
      --n-trials "$N_TRIALS" \
      --seed "$SEED" \
      --time-limit "$TIME_LIMIT" \
      --per-run-time-limit "$PER_RUN_TIME_LIMIT" \
      --dataset-retries "$DATASET_RETRIES" \
      --skip-test-eval \
      --all-datasets --include-datasets "$ds" \
      --output-dir "$OUTPUT_ROOT" \
      --history-dir "$HISTORY_DIR" \
      "${RESUME_FLAG[@]}" >>"$jlog" 2>&1
  else
    "$py" examples/benchmark/run_toy_categorical_optimizers.py \
      --optimizer "$opt" \
      --task-type "$tt" \
      --algorithm "$algo" \
      --metric "$metric" \
      --n-trials "$N_TRIALS" \
      --seed "$SEED" \
      --time-limit "$TIME_LIMIT" \
      --per-run-time-limit "$PER_RUN_TIME_LIMIT" \
      --dataset-retries "$DATASET_RETRIES" \
      --skip-test-eval \
      --all-datasets --include-datasets "$ds" \
      --output-dir "$OUTPUT_ROOT" \
      --history-dir "$HISTORY_DIR" \
      "${RESUME_FLAG[@]}" >>"$jlog" 2>&1
  fi
  local rc=$?
  echo "[$(date -Is)] DONE($rc) $opt $tt/$ds/$algo" >>"$MASTER_LOG"
  return "$rc"
}

n_launched=0
for opt in $OPTIMIZERS; do
  py="$PYTHON_CLASS"
  [[ "$opt" == "lolbo" ]] && py="$PYTHON_LOLBO"
  for t in "${TASKS[@]}"; do
    IFS='|' read -r tt ds algo metric <<<"$t"
    jlog="$LOGDIR/${opt}_${tt}_${ds}_${algo}_${TS}.log"
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "DRY $opt $tt/$ds/$algo metric=$metric python=$(basename $(dirname $(dirname $py)))"
      continue
    fi
    while [[ $(jobs -rp | wc -l) -ge $PARALLEL ]]; do sleep 5; done
    wait_for_load
    # Spread across the box (stride), not 0..PARALLEL-1 clustered on one socket.
    if (( PARALLEL > 0 )); then
      cpu=$(( (n_launched % PARALLEL) * (NPROC / PARALLEL) ))
    else
      cpu=$(( n_launched % NPROC ))
    fi
    # Expand cpu into the subshell at fork time; do not background the function.
    (
      run_one "$cpu" "$py" "$opt" "$tt" "$ds" "$algo" "$metric" "$jlog"
    ) &
    n_launched=$((n_launched+1))
  done
done
wait
log "=== ml_ds_test_19 baselines DONE (launched $n_launched jobs) ==="
log "master log: $MASTER_LOG"
