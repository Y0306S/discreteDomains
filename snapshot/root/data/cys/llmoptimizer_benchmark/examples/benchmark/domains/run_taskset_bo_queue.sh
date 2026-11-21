#!/usr/bin/env bash
# Work-queue TaskSet BO: N workers pull tasks from a shared queue until it drains.
#
# Why a queue and not static shards: measured per-eval cost varies 5.7x by family
# (quadratic 8.8s -> maf 49.8s at 1000 steps). With round-robin sharding, run A
# ended with 3 workers grinding for 90 minutes while 9 sat idle. Over a 10h job
# that tail is hours of wasted machine.
#
# Claiming is atomic via `mkdir` (succeeds for exactly one worker), so no locks
# and no coordinator process.
#
# TaskSet's train_inner is TF 1.15 CPU-only, so this is pure CPU; GPUs are idle
# by design. TRAIN_INNER_SUBPROCESS_THREADS=4 is hardcoded upstream, and a worker
# measures at ~1.8 busy cores, so WORKERS is the throughput lever.
#
# Usage (64-core / 256GB box):
#   WORKERS=16 CPUS_PER_WORKER=4 STEPS=1000 N_TRIALS=100 ./run_taskset_bo_queue.sh
#   DRY_RUN=1 ./run_taskset_bo_queue.sh
#
# train_inner uses 4 TF threads. Pin 4 CPUs per worker on a 64-core machine
# (16 workers * 4 = 64). Fewer CPUs than 4 caused a ~12x slowdown.

set -euo pipefail

REPO="${REPO:-/root/data/cys/llmoptimizer_benchmark}"
PY="${PY:-/root/data/cys/conda_envs/hpo_benchmark/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="/root/data/cys/conda_envs/hpo_benchmark/bin/python"
fi

CORES="$(nproc)"
if (( CORES <= 64 )); then
  WORKERS="${WORKERS:-16}"
  CPUS_PER_WORKER="${CPUS_PER_WORKER:-4}"
elif (( CORES <= 96 )); then
  # 2x24c SMT box (e.g. 2x EPYC 7413): 96 logical, real NUMA 0-23,48-71 / 24-47,72-95
  WORKERS="${WORKERS:-24}"
  CPUS_PER_WORKER="${CPUS_PER_WORKER:-4}"
else
  WORKERS="${WORKERS:-48}"
  CPUS_PER_WORKER="${CPUS_PER_WORKER:-2}"
fi
STEPS="${STEPS:-1000}"
EVAL_EVERY="${EVAL_EVERY:-$STEPS}"
REPLICAS="${REPLICAS:-1}"
N_TRIALS="${N_TRIALS:-100}"
N_STARTUP="${N_STARTUP:-1}"
OPTIMIZER="${OPTIMIZER:-tpe}"
SEED="${SEED:-1}"
METRIC="${METRIC:-min_normalized_valid_outer}"
SPLIT_IDS="${SPLIT_IDS:-/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/dl_split_80_20_task_ids.json}"
SIDE="${SIDE:-train}"
RUN="${RUN:-taskset_corpus_${SIDE}_${STEPS}s_${N_TRIALS}t}"
OUT="${OUT:-/root/data/logs/$RUN}"
DRY_RUN="${DRY_RUN:-0}"

QUEUE="$OUT/_queue"
CLAIMS="$QUEUE/claims"
mkdir -p "$CLAIMS" "$OUT"
cd "$REPO"

# --- build the queue: every task on the requested side of the 80/20 split ---
$PY - "$SPLIT_IDS" "$SIDE" > "$QUEUE/tasks.txt" <<'PYEOF'
import json, sys
ids = json.load(open(sys.argv[1]))
for t in sorted(ids[sys.argv[2]]):
    print(t)
PYEOF
N_TASKS=$(wc -l < "$QUEUE/tasks.txt")

# Interleave families so early claims are cost-mixed rather than all-maf.
$PY - "$QUEUE/tasks.txt" > "$QUEUE/tasks.ordered" <<'PYEOF'
import re, sys
from collections import defaultdict, OrderedDict
fam = lambda t: (re.match(r'^(.*?_family)_seed\d+$', t) or [None, 'other'])[1]
by = defaultdict(list)
for line in open(sys.argv[1]):
    t = line.strip()
    if t: by[fam(t)].append(t)
order = OrderedDict(sorted(by.items()))
while any(order.values()):
    for k in list(order):
        if order[k]:
            print(order[k].pop(0))
PYEOF
mv "$QUEUE/tasks.ordered" "$QUEUE/tasks.txt"

echo "=== TaskSet BO corpus run (work queue) ==="
echo "  split side       : $SIDE  ($SPLIT_IDS)"
echo "  tasks in queue   : $N_TASKS"
echo "  trials per task  : $N_TRIALS   (n_startup_trials=$N_STARTUP)"
echo "  inner protocol   : $STEPS steps, eval_every=$EVAL_EVERY, replicas=$REPLICAS"
echo "  objective        : $METRIC"
echo "  optimizer / seed : $OPTIMIZER / $SEED"
echo "  workers          : $WORKERS x ${CPUS_PER_WORKER} pinned CPUs (nproc=$CORES)"
echo "  python           : $PY"
echo "  repo             : $REPO"
echo "  total evals      : $(( N_TASKS * N_TRIALS ))"
echo "  output           : $OUT"
[ "$DRY_RUN" = "1" ] && echo "  MODE             : DRY RUN"

if (( CORES <= 64 )); then
  NODE0=(); for c in $(seq 0 31); do NODE0+=("$c"); done
  NODE1=(); for c in $(seq 32 63); do NODE1+=("$c"); done
  POOL_HINT=32
elif (( CORES <= 96 )); then
  NODE0=(); for c in $(seq 0 23) $(seq 48 71); do NODE0+=("$c"); done
  NODE1=(); for c in $(seq 24 47) $(seq 72 95); do NODE1+=("$c"); done
  POOL_HINT=48
else
  NODE0=(); for c in $(seq 0 31)  $(seq 64 95);  do NODE0+=("$c"); done
  NODE1=(); for c in $(seq 32 63) $(seq 96 127); do NODE1+=("$c"); done
  POOL_HINT=64
fi
per_node=$(( (WORKERS + 1) / 2 ))
demand=$(( per_node * CPUS_PER_WORKER ))
if (( demand > POOL_HINT )); then
  echo "  cpuset overlap    : ${demand} CPU-slots over ${POOL_HINT} per NUMA node (pins wrap)"
  echo "                      keep CPUS_PER_WORKER>=4 (TF train_inner uses 4 threads)"
elif (( CPUS_PER_WORKER < 4 )); then
  echo "  ! WARNING: CPUS_PER_WORKER=$CPUS_PER_WORKER < TRAIN_INNER_SUBPROCESS_THREADS=4."
  echo "  !          Measured 12x slowdown on compute-heavy tasks when threads exceed the pin."
fi
echo

cpuset_for() {
  local w=$1 node=$(( $1 % 2 )) slot=$(( $1 / 2 )) start
  start=$(( slot * CPUS_PER_WORKER ))
  local -n pool=$([ "$node" -eq 0 ] && echo NODE0 || echo NODE1)
  start=$(( start % ${#pool[@]} ))
  local ids=("${pool[@]:start:CPUS_PER_WORKER}")
  local IFS=','; echo "${ids[*]}"
}

if [ "$DRY_RUN" = "1" ]; then
  echo "  first 12 queue entries (family-interleaved):"
  head -12 "$QUEUE/tasks.txt" | sed 's/^/    /'
  for w in 0 1 2 3; do echo "    worker $w -> node$(( w % 2 )) cpus=$(cpuset_for $w)"; done
  echo "  worker $((WORKERS-1)) -> cpus=$(cpuset_for $((WORKERS-1)))"
  echo; echo "DRY RUN complete - queue built at $QUEUE/tasks.txt, nothing launched"
  exit 0
fi

# --- worker loop: skip complete histories; flock so a dead claim can resume ---
cat > "$QUEUE/worker.sh" <<'WEOF'
#!/usr/bin/env bash
set -uo pipefail
w=$1
source "$2"            # exported settings
hist_nobs() {
  "$PY" - "$1" "$REPO" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from mindware.domains.history import history_exists, load_openbox_history
p = Path(sys.argv[1])
print(len(load_openbox_history(p).get("observations") or []) if history_exists(p) else 0)
PY
}
while read -r task; do
  [ -z "$task" ] && continue
  task_out="$OUT/$task/$OPTIMIZER/seed$SEED"
  nobs=$(hist_nobs "$task_out/history.json") || exit 1
  if [ "${nobs:-0}" -ge "$N_TRIALS" ]; then
    echo "[w$w] $(date -Is) SKIP  $task nobs=$nobs" >> "$OUT/queue.log"
    continue
  fi
  mkdir -p "$CLAIMS"
  exec 9>"$CLAIMS/$task.lock"
  if ! flock -n 9; then
    echo "[w$w] $(date -Is) BUSY  $task" >> "$OUT/queue.log"
    continue
  fi
  echo "[w$w] $(date -Is) START $task nobs=$nobs" >> "$OUT/queue.log"
  "$PY" examples/benchmark/domains/run_taskset_bo.py \
      --optimizer "$OPTIMIZER" --n-trials "$N_TRIALS" --seed "$SEED" --no-dummy \
      --training-steps "$STEPS" --eval-every-n "$EVAL_EVERY" --num-replicas "$REPLICAS" \
      --objective-metric "$METRIC" --n-startup-trials "$N_STARTUP" \
      --task-ids "$task" --output-dir "$OUT" \
      >> "$OUT/worker${w}.log" 2>&1
  rc=$?
  echo "[w$w] $(date -Is) DONE  $task rc=$rc" >> "$OUT/queue.log"
done < "$QUEUE/tasks.txt"
echo "[w$w] $(date -Is) EXIT queue drained" >> "$OUT/queue.log"
WEOF
chmod +x "$QUEUE/worker.sh"

cat > "$QUEUE/env.sh" <<EOF
export PY="$PY" OUT="$OUT" CLAIMS="$CLAIMS" QUEUE="$QUEUE"
export OPTIMIZER="$OPTIMIZER" N_TRIALS="$N_TRIALS" SEED="$SEED"
export STEPS="$STEPS" EVAL_EVERY="$EVAL_EVERY" REPLICAS="$REPLICAS"
export METRIC="$METRIC" N_STARTUP="$N_STARTUP"
export PYTHONPATH="$REPO\${PYTHONPATH:+:\$PYTHONPATH}"
export TASKSET_EVAL_PYTHON="/root/data/cys/conda_envs/taskset_eval/bin/python"
EOF

: > "$OUT/worker.pids"
for (( w=0; w<WORKERS; w++ )); do
  cpus=$(cpuset_for "$w")
  if [ -n "$cpus" ]; then pin=(taskset -c "$cpus"); else pin=(); fi
  nohup "${pin[@]}" bash "$QUEUE/worker.sh" "$w" "$QUEUE/env.sh" >> "$OUT/worker${w}.boot.log" 2>&1 &
  echo "$!" >> "$OUT/worker.pids"
done

echo "launched $WORKERS workers; pids in $OUT/worker.pids"
echo "queue log : $OUT/queue.log"
echo "progress  : find $OUT -name history.json | wc -l   (tasks with a checkpoint)"
echo "resume    : copy $OUT to the new host and rerun this script with the same OUT/N_TRIALS"
