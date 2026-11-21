#!/usr/bin/env bash
# Continue BOLT LOLBO -> 25k on GPUs 0, 2, 3 only (never GPU 1).
# GPU 0: default 0 workers (usually ~77GB used by other jobs).
# GPU 2: 1 worker (shard 6). GPU 3: 4 workers (shards 3,7,0,4) — shares GPU with TPE.
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
PY="${PY:-/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python}"
LOGDIR="${LOGDIR:-/root/data/logs}"
MAX_ORACLE_BUDGET="${MAX_ORACLE_BUDGET:-25000}"
N_TASKS="${N_TASKS:-1000}"
N_SHARDS="${N_SHARDS:-8}"
LOLBO_THREADS="${LOLBO_THREADS:-8}"
WORKERS_GPU0="${WORKERS_GPU0:-0}"
WORKERS_GPU2="${WORKERS_GPU2:-1}"
WORKERS_GPU3="${WORKERS_GPU3:-4}"
mkdir -p "$LOGDIR"
export PYTHONPATH="$REPO"

MASTER="$LOGDIR/bo_protein_bolt_lolbo_continue_gpus023.log"
PIDFILE="$LOGDIR/bo_protein_bolt_lolbo_continue_gpus023.pids"
CPU_BASE=64

pin_rl_jobs() {
  local pids=()
  mapfile -t pids < <(pgrep -f 'run_p5_rl_bo_prefix_b.py' || true)
  local base=0 pid cpus
  for pid in "${pids[@]}"; do
    cpus="${base}-$((base + 7))"
    taskset -apc "$cpus" "$pid" >/dev/null 2>&1 || true
    base=$((base + 8))
  done
}

pin_rl_jobs
echo "[lolbo-g023] $(date -Is) GPUs 0/2/3 workers=${WORKERS_GPU0}/${WORKERS_GPU2}/${WORKERS_GPU3} budget=${MAX_ORACLE_BUDGET}" | tee "$MASTER"
: >"$PIDFILE"

"$PY" - <<'PY'
import os
from pathlib import Path
from mindware.domains.protein.bolt_lolbo_checkpoint import ensure_checkpoint
from mindware.domains.paths import protein_bo80_bolt_root
root = protein_bo80_bolt_root()
for i in range(int(os.environ.get("N_TASKS", "1000"))):
    hist = root / f"pep_{i:03d}/bolt_native/seed1/history.json"
    if hist.is_file():
        ensure_checkpoint(hist)
print("checkpoints ready")
PY

launch_shard() {
  local gpu="$1" shard="$2" cpus="$3" log="$4"
  echo "[lolbo-g023] gpu=${gpu} shard=${shard} cpus=${cpus} -> ${log}" | tee -a "$MASTER"
  CUDA_VISIBLE_DEVICES="$gpu" \
  OMP_NUM_THREADS="$LOLBO_THREADS" MKL_NUM_THREADS="$LOLBO_THREADS" \
  OPENBLAS_NUM_THREADS="$LOLBO_THREADS" NUMEXPR_NUM_THREADS="$LOLBO_THREADS" \
  TORCH_NUM_THREADS="$LOLBO_THREADS" \
  taskset -c "$cpus" "$PY" -u "$REPO/examples/benchmark/domains/run_protein_bolt_lolbo.py" \
    --n-tasks "$N_TASKS" --n-trials 500 --device-id 0 --skip-done \
    --run-mode continue_bolt --max-oracle-budget "$MAX_ORACLE_BUDGET" \
    --oracle-mode bolt_native --bolt-init-points 1000 --bsz 50 --max-string-length 30 \
    --shard-index "$shard" --n-shards "$N_SHARDS" \
    >>"$log" 2>&1 &
  echo "$!" >>"$PIDFILE"
}

next_cpus() {
  local start=$CPU_BASE
  CPU_BASE=$((CPU_BASE + 8))
  echo "${start}-$((start + 7))"
}

# GPU 0 shards (only if WORKERS_GPU0>0 and VRAM allows)
GPU0_SHARDS=(0 4)
for ((w=0; w<WORKERS_GPU0 && w<${#GPU0_SHARDS[@]}; w++)); do
  launch_shard 0 "${GPU0_SHARDS[$w]}" "$(next_cpus)" "$LOGDIR/bo_protein_bolt_lolbo_g0_${GPU0_SHARDS[$w]}.log"
done

# GPU 2
GPU2_SHARDS=(6 2)
for ((w=0; w<WORKERS_GPU2 && w<${#GPU2_SHARDS[@]}; w++)); do
  launch_shard 2 "${GPU2_SHARDS[$w]}" "$(next_cpus)" "$LOGDIR/bo_protein_bolt_lolbo_g2_${GPU2_SHARDS[$w]}.log"
done

# GPU 3 (extra capacity; TPE uses cores 0-15 on same GPU)
GPU3_SHARDS=(3 7 0 4)
for ((w=0; w<WORKERS_GPU3 && w<${#GPU3_SHARDS[@]}; w++)); do
  launch_shard 3 "${GPU3_SHARDS[$w]}" "$(next_cpus)" "$LOGDIR/bo_protein_bolt_lolbo_g3_${GPU3_SHARDS[$w]}.log"
done

echo "[lolbo-g023] pids: $(tr '\n' ' ' <"$PIDFILE")" | tee -a "$MASTER"
