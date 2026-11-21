#!/usr/bin/env bash
# Parallel TaskSet BO over a family-balanced task split, with NUMA-aware CPU pinning.
#
# TaskSet's train_inner is TF 1.15 CPU-only (tf.test.is_built_with_cuda() == False),
# so this is a pure CPU workload -- the GPUs are irrelevant here.
#
# This box: 128 logical CPUs = 2 sockets x 32 cores x 2 threads, 2 NUMA nodes
#   node0 = 0-31,64-95     node1 = 32-63,96-127
#
# Workers are assigned to sockets ALTERNATELY (worker0->node0, worker1->node1,
# worker2->node0, ...) so any worker count spreads evenly across both sockets and
# across both hyperthread halves, instead of piling onto one socket's memory
# controller.
#
# NOTE on threads: run_taskset_dl_eval_optimized.py hardcodes
# TRAIN_INNER_SUBPROCESS_THREADS = 4 and applies it to OMP/MKL/TF on every
# train_inner subprocess, so each worker consumes roughly 4 busy CPUs no matter how
# wide its affinity mask is. Raising WORKERS is therefore the lever for using more
# of the machine; widening CPUS_PER_WORKER only gives headroom.
#
# GPUs are NOT usable for this workload: the taskset_eval TF build reports
# tf.test.is_built_with_cuda() == False and lists zero GPU devices, and these are
# A800s (compute capability 8.0) which stock TF 1.15 GPU wheels (CUDA 10.0,
# sm_<=75) do not support anyway. Scaling here means more CPU workers.
#
# Usage:
#   WORKERS=12 ./run_taskset_bo_parallel.sh
#   DRY_RUN=1 WORKERS=12 ./run_taskset_bo_parallel.sh   # print the plan, launch nothing

set -euo pipefail

REPO="/root/data/cys/llmoptimizer_benchmark"
PY="/root/data/cys/conda_envs/hpo_benchmark/bin/python"

WORKERS="${WORKERS:-4}"
CPUS_PER_WORKER="${CPUS_PER_WORKER:-8}"
PER_FAMILY="${PER_FAMILY:-5}"
STEPS="${STEPS:-1000}"
EVAL_EVERY="${EVAL_EVERY:-$STEPS}"     # one eval at the end (plus the unavoidable step-0)
REPLICAS="${REPLICAS:-1}"
N_TRIALS="${N_TRIALS:-20}"
# 1 = minimum reachable. run_domain_search would otherwise derive
# max(1, min(8, n_trials//10)) = 2 at n_trials=20, and clamps with max(1, ...),
# so zero random startup trials is not reachable.
N_STARTUP="${N_STARTUP:-1}"
OPTIMIZER="${OPTIMIZER:-tpe}"
SEED="${SEED:-1}"
SOURCE="${SOURCE:-train}"              # dl_train_85 side of the LLM split
METRIC="${METRIC:-min_normalized_valid_outer}"
RUN="${RUN:-taskset_bo_${SOURCE}_${STEPS}steps}"
OUT="${OUT:-/root/data/logs/$RUN}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$OUT"
cd "$REPO"

# --- task list: equal number per family, from the requested side of the split ---
TASKS="$($PY examples/benchmark/domains/make_taskset_bo_split.py \
          --source "$SOURCE" --per-family "$PER_FAMILY" --print-ids)"
IFS=',' read -r -a TASK_ARR <<< "$TASKS"
N_TASKS=${#TASK_ARR[@]}

# --- NUMA-aware CPU sets, alternating sockets ---
NODE0=(); for c in $(seq 0 31)  $(seq 64 95);  do NODE0+=("$c"); done
NODE1=(); for c in $(seq 32 63) $(seq 96 127); do NODE1+=("$c"); done

cpuset_for() {                      # $1 = worker index
  local w=$1 node=$(( $1 % 2 )) slot=$(( $1 / 2 )) start
  start=$(( slot * CPUS_PER_WORKER ))
  local -n pool=$([ "$node" -eq 0 ] && echo NODE0 || echo NODE1)
  if (( start + CPUS_PER_WORKER > ${#pool[@]} )); then
    echo ""                         # pool exhausted -> caller leaves it unpinned
    return
  fi
  local ids=("${pool[@]:start:CPUS_PER_WORKER}")
  local IFS=','; echo "${ids[*]}"
}

echo "=== TaskSet BO (parallel, CPU) ==="
echo "  split source     : $SOURCE (dl_train_85 -> recovered task ids)"
echo "  tasks            : $N_TASKS  (${PER_FAMILY} per family)"
echo "  inner protocol   : ${STEPS} steps, eval_every=${EVAL_EVERY}, replicas=${REPLICAS}"
echo "  objective        : $METRIC"
echo "  optimizer/trials : $OPTIMIZER / $N_TRIALS  (n_startup_trials=$N_STARTUP)"
echo "  workers          : $WORKERS  ($CPUS_PER_WORKER logical CPUs each, alternating sockets)"
echo "  output           : $OUT"
echo "  total evals      : $(( N_TASKS * N_TRIALS ))"
echo "  device           : CPU only (taskset_eval TF 1.15 has no CUDA support)"
[ "$DRY_RUN" = "1" ] && echo "  MODE             : DRY RUN - nothing will be launched"
echo

# Warn if the requested workers cannot all be pinned: each NUMA pool holds 64
# logical CPUs, so ceil(WORKERS/2) * CPUS_PER_WORKER must stay <= 64.
per_node=$(( (WORKERS + 1) / 2 ))
if (( per_node * CPUS_PER_WORKER > 64 )); then
  echo "  ! WARNING: $per_node workers/node x $CPUS_PER_WORKER CPUs = $(( per_node * CPUS_PER_WORKER )) > 64 available."
  echo "  !          Later workers will run UNPINNED. Reduce CPUS_PER_WORKER to $(( 64 / per_node )) to pin them all."
  echo
fi

# --- shard tasks round-robin so each worker gets a family-mixed load ---
for (( w=0; w<WORKERS; w++ )); do
  shard=()
  for (( i=w; i<N_TASKS; i+=WORKERS )); do shard+=("${TASK_ARR[$i]}"); done
  if (( ${#shard[@]} == 0 )); then continue; fi
  ids=$(IFS=','; echo "${shard[*]}")
  cpus=$(cpuset_for "$w")
  log="$OUT/worker${w}.log"

  if [ -n "$cpus" ]; then pin=(taskset -c "$cpus"); else pin=(); fi
  echo "  worker $w: node$(( w % 2 )) cpus=${cpus:-<UNPINNED>} tasks=${#shard[@]}  [${shard[0]} ...]"

  if [ "$DRY_RUN" = "1" ]; then continue; fi

  nohup "${pin[@]}" "$PY" examples/benchmark/domains/run_taskset_bo.py \
      --optimizer "$OPTIMIZER" --n-trials "$N_TRIALS" --seed "$SEED" --no-dummy \
      --training-steps "$STEPS" --eval-every-n "$EVAL_EVERY" --num-replicas "$REPLICAS" \
      --objective-metric "$METRIC" --n-startup-trials "$N_STARTUP" \
      --task-ids "$ids" --output-dir "$OUT" \
      > "$log" 2>&1 &
  echo "$!" >> "$OUT/worker.pids"
done

echo
if [ "$DRY_RUN" = "1" ]; then
  echo "DRY RUN complete - no processes started, no files written under $OUT"
  echo "to launch for real, re-run the same command without DRY_RUN=1"
else
  echo "launched $WORKERS workers; pids in $OUT/worker.pids"
  echo "monitor:  tail -f $OUT/worker0.log"
  echo "progress: grep -c best_objective $OUT/worker*.log"
fi
