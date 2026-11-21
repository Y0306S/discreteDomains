#!/usr/bin/env bash
# Continue BOLT LOLBO from checkpoint -> 25k oracle budget.
# Two workers per GPU (8 shards), each pinned to a disjoint CPU core range.
# Tasks already past this budget (including finished 50k/200k runs) are skipped via --skip-done.
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
PY="${PY:-/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python}"
LOGDIR="${LOGDIR:-/root/data/logs}"
MAX_ORACLE_BUDGET="${MAX_ORACLE_BUDGET:-25000}"
N_TASKS="${N_TASKS:-1000}"
N_SHARDS="${N_SHARDS:-8}"
LOLBO_THREADS="${LOLBO_THREADS:-8}"
RL_CORES_PER_JOB="${RL_CORES_PER_JOB:-8}"
mkdir -p "$LOGDIR"
export PYTHONPATH="$REPO"

MASTER="$LOGDIR/bo_protein_bolt_lolbo_continue_parallel.log"
PIDFILE="$LOGDIR/bo_protein_bolt_lolbo_continue.pids"

pin_rl_jobs() {
  local pids=()
  mapfile -t pids < <(pgrep -f 'run_p5_rl_bo_prefix_b.py' || true)
  local base=0
  local pid cpus
  local max_rl_cores=64
  for pid in "${pids[@]}"; do
    if ((base >= max_rl_cores)); then
      taskset -apc 63 "$pid" >/dev/null 2>&1 || true
      continue
    fi
    cpus="${base}-$((base + RL_CORES_PER_JOB - 1))"
    taskset -apc "$cpus" "$pid" >/dev/null 2>&1 || true
    base=$((base + RL_CORES_PER_JOB))
  done
  if ((${#pids[@]} > 0)); then
    echo "[bolt-lolbo-continue] pinned ${#pids[@]} RL jobs to cores 0-$((base - 1))" | tee -a "$MASTER"
  fi
}

pin_rl_jobs
echo "[bolt-lolbo-continue] $(date -Is) RESUME ${N_TASKS} tasks continue_bolt -> ${MAX_ORACLE_BUDGET} oracle calls, ${N_SHARDS} shards (2/GPU), ${LOLBO_THREADS} CPU threads each, LOLBO cores 64-127" | tee -a "$MASTER"
: >"$PIDFILE"

# Materialize checkpoint.pt from existing 500-call histories if needed.
"$PY" - <<'PY'
import os
from pathlib import Path
from mindware.domains.protein.bolt_lolbo_checkpoint import ensure_checkpoint
from mindware.domains.paths import protein_bo80_bolt_root
root = protein_bo80_bolt_root()
n_tasks = int(os.environ.get("N_TASKS", "1000"))
for i in range(n_tasks):
    hist = root / f"pep_{i:03d}/bolt_native/seed1/history.json"
    if hist.is_file():
        ensure_checkpoint(hist)
print("checkpoints ready")
PY

launch() {
  local gpu="$1" shard="$2" cpus="$3" log="$4"
  CUDA_VISIBLE_DEVICES="$gpu" \
  OMP_NUM_THREADS="$LOLBO_THREADS" \
  MKL_NUM_THREADS="$LOLBO_THREADS" \
  OPENBLAS_NUM_THREADS="$LOLBO_THREADS" \
  NUMEXPR_NUM_THREADS="$LOLBO_THREADS" \
  TORCH_NUM_THREADS="$LOLBO_THREADS" \
  taskset -c "$cpus" "$PY" -u "$REPO/examples/benchmark/domains/run_protein_bolt_lolbo.py" \
    --n-tasks "$N_TASKS" --n-trials 500 --device-id 0 --skip-done \
    --run-mode continue_bolt --max-oracle-budget "$MAX_ORACLE_BUDGET" \
    --oracle-mode bolt_native --bolt-init-points 1000 --bsz 50 --max-string-length 30 \
    --shard-index "$shard" --n-shards "$N_SHARDS" \
    >>"$log" 2>&1 &
  echo "$!" >>"$PIDFILE"
}

# GPU 0-3 × 2 workers. RL on cores 0-63 (pinned above). LOLBO on 64-127 (8 cores each).
# NOTE: superseded for ops by run_protein_bolt_lolbo_continue_gpus023.sh (GPUs 0,2,3 only).
launch 0 0 "64-71" "$LOGDIR/bo_protein_bolt_lolbo_continue_gpu0a.log"
launch 0 4 "72-79" "$LOGDIR/bo_protein_bolt_lolbo_continue_gpu0b.log"
launch 1 1 "80-87" "$LOGDIR/bo_protein_bolt_lolbo_continue_gpu1a.log"
launch 1 5 "88-95" "$LOGDIR/bo_protein_bolt_lolbo_continue_gpu1b.log"
launch 2 2 "96-103" "$LOGDIR/bo_protein_bolt_lolbo_continue_gpu2a.log"
launch 2 6 "104-111" "$LOGDIR/bo_protein_bolt_lolbo_continue_gpu2b.log"
launch 3 3 "112-119" "$LOGDIR/bo_protein_bolt_lolbo_continue_gpu3a.log"
launch 3 7 "120-127" "$LOGDIR/bo_protein_bolt_lolbo_continue_gpu3b.log"

echo "pids: $(tr '\n' ' ' <"$PIDFILE")" | tee -a "$MASTER"
wait $(tr '\n' ' ' <"$PIDFILE")
echo "[bolt-lolbo-continue] $(date -Is) all shards finished" | tee -a "$MASTER"
