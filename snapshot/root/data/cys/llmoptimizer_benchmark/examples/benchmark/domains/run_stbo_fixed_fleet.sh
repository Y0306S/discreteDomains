#!/usr/bin/env bash
# STBO reference runs on the CORRECTED pipeline: per-reference init pools, deterministic
# acquisition budget, feasibility assertion. See docs/BOLT_PEPTIDE_BO_FIX_PLAN.md and
# docs/BOLT_PEPTIDE_SPLIT.md.
#
# Protocol: 1000 init points per task, 100 rounds x 10 candidates = 1000 acquisition calls.
# Default task range is the TRAIN split (pep_000..pep_899); the LLM fine-tuning corpus.
#
# Worker density: measured 1 -> 2 workers/GPU = +32% throughput, 2 -> 3 = +7% while
# inflating per-task latency 79%. So 2 per GPU.
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
PY="${PY:-/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python}"
GEN="${GEN:-/root/data/cys/BOLT/optimization/peptides/apex_oracle/init_data/generated}"
LOGDIR="${LOGDIR:-/root/data/logs}"
OUT_ROOT="${OUT_ROOT:-/root/data/cys/blockdata_archive/domain_protein/stbo_bsz10_r100_fixed}"

N_TASKS="${N_TASKS:-900}"             # pep_000 .. pep_(N_TASKS-1); 900 = train split
# TASK_IDS: optional explicit comma-separated task list. When set it overrides N_TASKS and
# is sharded across workers here (run_tasks() ignores sharding when --task-ids is given).
TASK_IDS="${TASK_IDS:-}"
BUDGET="${BUDGET:-1000}"              # acquisition calls per task (100 rounds x bsz 10)
BSZ="${BSZ:-10}"
E2E_FREQ="${E2E_FREQ:-33}"
INIT_POINTS="${INIT_POINTS:-1000}"
BUDGET_MODE="${BUDGET_MODE:-acquisition}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
THREADS="${THREADS:-4}"
BASE_CPU="${BASE_CPU:-64}"

INIT_DATA="${INIT_DATA:-$GEN/refs_0_1000_init.txt}"
INIT_SCORES="${INIT_SCORES:-$GEN/refs_0_1000_scores.csv}"
INIT_OFFSET="${INIT_OFFSET:-0}"

for f in "$INIT_DATA" "$INIT_SCORES"; do
  [[ -s "$f" ]] || { echo "FATAL: missing init pool $f -- run tools/build_bolt_init_pools.py" >&2; exit 1; }
done

mkdir -p "$OUT_ROOT" "$LOGDIR"
export PYTHONPATH="$REPO"
export PROTEIN_BO80_BOLT_ROOT="$OUT_ROOT"
export BOLT_MIN_FREE_GB="${BOLT_MIN_FREE_GB:-50}"
export TQDM_DISABLE=1                  # tqdm bars were >90% of the old 17MB/shard logs

IFS=',' read -r -a GPUS <<<"$GPU_IDS"
NGPU="${#GPUS[@]}"
N_SHARDS=$(( NGPU * WORKERS_PER_GPU ))

MASTER="$LOGDIR/stbo_fixed_launch.log"
PIDFILE="$LOGDIR/stbo_fixed.pids"
cd "$REPO"

{
  echo "[stbo-fixed] $(date -Is)"
  echo "  out_root      = $OUT_ROOT"
  if [[ -n "$TASK_IDS" ]]; then
    echo "  tasks         = explicit list (n=$(tr ',' '\n' <<<"$TASK_IDS" | grep -c .))"
  else
    echo "  tasks         = pep_000..pep_$(printf '%03d' $((N_TASKS-1)))  (n=$N_TASKS)"
  fi
  echo "  budget        = $BUDGET ($BUDGET_MODE)  bsz=$BSZ  rounds=$((BUDGET/BSZ))  e2e_freq=$E2E_FREQ"
  echo "  init pool     = $INIT_DATA (offset $INIT_OFFSET, $INIT_POINTS pts/task)"
  echo "  gpus          = $GPU_IDS  workers/gpu=$WORKERS_PER_GPU  shards=$N_SHARDS  threads=$THREADS"
} | tee "$MASTER"
: >"$PIDFILE"

shard=0
for ((g=0; g<NGPU; g++)); do
  for ((w=0; w<WORKERS_PER_GPU; w++)); do
    gpu="${GPUS[$g]}"
    cpu_start=$(( BASE_CPU + shard * THREADS ))
    cpu_end=$(( cpu_start + THREADS - 1 ))
    log="$LOGDIR/stbo_fixed_s${shard}.log"
    echo "[stbo-fixed] shard=$shard gpu=$gpu cpus=${cpu_start}-${cpu_end} -> $log" | tee -a "$MASTER"
    # When TASK_IDS is given, shard the list here (run_tasks ignores --n-shards then).
    SEL_ARGS=(--n-tasks "$N_TASKS" --n-shards "$N_SHARDS" --shard-index "$shard")
    if [[ -n "$TASK_IDS" ]]; then
      sub="$(tr ',' '\n' <<<"$TASK_IDS" | grep . | awk -v s="$shard" -v n="$N_SHARDS" 'NR%n==s%n' | paste -sd, -)"
      if [[ -z "$sub" ]]; then shard=$(( shard + 1 )); continue; fi
      SEL_ARGS=(--task-ids "$sub")
    fi
    CUDA_VISIBLE_DEVICES="$gpu" \
    OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
    OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS" \
    taskset -c "${cpu_start}-${cpu_end}" nohup "$PY" -u \
      examples/benchmark/domains/run_protein_bolt_lolbo.py \
        --run-mode continue_bolt \
        --max-oracle-budget "$BUDGET" --n-trials "$BUDGET" --budget-mode "$BUDGET_MODE" \
        --bolt-init-points "$INIT_POINTS" --bsz "$BSZ" --e2e-freq "$E2E_FREQ" \
        --oracle-mode bolt_native --skip-done \
        "${SEL_ARGS[@]}" --device-id 0 \
        --init-data-path "$INIT_DATA" --init-scores-path "$INIT_SCORES" \
        --init-offset-helper "$INIT_OFFSET" \
      >>"$log" 2>&1 &
    echo "$!" >>"$PIDFILE"
    shard=$(( shard + 1 ))
  done
done

echo "[stbo-fixed] pids: $(tr '\n' ' ' <"$PIDFILE")" | tee -a "$MASTER"
echo "[stbo-fixed] monitor: python $REPO/tools/stbo_status.py" | tee -a "$MASTER"
