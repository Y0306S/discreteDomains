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
# Round-robin workers across NGPU cards. lolbo_sampler defaults to bare "cuda",
# so without this every worker lands on cuda:0 and cards 1..N-1 sit idle.
# NGPU=0 (default) leaves behaviour identical to the CPU driver.
NGPU="${NGPU:-0}"

# Cores per task. taskset -c N pins to ONE core and forked children inherit that
# mask, so sklearn/CatBoost/OpenMP cannot spill to N+1..N+3. Measured effect on
# the controller: 16 pinned slots yielded only 5.8 cores of real work (12-39% of
# each pinned core) because unpinned co-tenants (ckpt32, TPE) roam [0-127] and
# contend for those same cores. A bounded range lets a worker escape a contended
# core while still capping fan-out. Must stay <= the stride (NPROC/PARALLEL) so
# slots never overlap. 1 = original single-core behaviour.
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"

# Runtime-adjustable width. run_one re-reads this file before EVERY task launch,
# so you can widen/narrow the pin without restarting the run - newly launched
# tasks pick up the new value, in-flight ones keep theirs. When the co-tenants
# finish (ckpt32 LLM-TPE / TPE here, TaskSet BO on LLMOptimizer4) there is no
# longer anything to protect cores from, so:
#     echo 8 > <OUTPUT_ROOT>/.cpus_per_task
# Clamped to the stride (NPROC/PARALLEL) so slots can never overlap each other.
CPUS_CTL_FILE="${CPUS_CTL_FILE:-$OUTPUT_ROOT/.cpus_per_task}"
cpus_for_task() {
  local v=""
  [[ -r "$CPUS_CTL_FILE" ]] && v="$(tr -dc '0-9' < "$CPUS_CTL_FILE" 2>/dev/null | head -c 3)"
  [[ -z "$v" ]] && v="$CPUS_PER_TASK"
  (( v < 1 )) && v=1
  local stride=1
  (( PARALLEL > 0 )) && stride=$(( NPROC / PARALLEL ))
  (( stride < 1 )) && stride=1
  (( v > stride )) && v=$stride
  echo "$v"
}

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
  local cpuset="$cpu" cpt
  cpt="$(cpus_for_task)"
  if (( cpt > 1 )); then
    local hi=$(( cpu + cpt - 1 ))
    (( hi > NPROC - 1 )) && hi=$(( NPROC - 1 ))
    cpuset="${cpu}-${hi}"
  fi
  echo "[$(date -Is)] START $opt $tt/$ds/$algo pin=taskset -c $cpuset gpu=${CUDA_VISIBLE_DEVICES:-none}" >>"$MASTER_LOG"
  if [[ "$PIN_CPUS" == "1" ]]; then
    taskset -c "$cpuset" "$py" examples/benchmark/run_toy_categorical_optimizers.py \
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
    if (( NGPU > 0 )); then gpu=$(( n_launched % NGPU )); else gpu=""; fi
    (
      [[ -n "$gpu" ]] && export CUDA_VISIBLE_DEVICES="$gpu"
      run_one "$cpu" "$py" "$opt" "$tt" "$ds" "$algo" "$metric" "$jlog"
    ) &
    n_launched=$((n_launched+1))
  done
done
wait
log "=== ml_ds_test_19 baselines DONE (launched $n_launched jobs) ==="
log "master log: $MASTER_LOG"
