#!/usr/bin/env bash
# Continue BOLT LOLBO -> 25k on GPUs 0-3 (2 workers/GPU = 8 shards).
# CPU: RL ~0-63, TPE ~8-23, LOLBO ~64-95 (4 threads/worker).
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
PY="${PY:-/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python}"
LOGDIR="${LOGDIR:-/root/data/logs}"
MAX_ORACLE_BUDGET="${MAX_ORACLE_BUDGET:-25000}"
N_TASKS="${N_TASKS:-1000}"
N_SHARDS="${N_SHARDS:-8}"
LOLBO_THREADS="${LOLBO_THREADS:-4}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
mkdir -p "$LOGDIR"
export PYTHONPATH="$REPO"

MASTER="$LOGDIR/bo_protein_bolt_lolbo_continue_gpus012.log"
PIDFILE="$LOGDIR/bo_protein_bolt_lolbo_continue_gpus012.pids"

pin_rl_jobs() {
  local pids=()
  mapfile -t pids < <(pgrep -f 'train_p5_online_grpo.py' || true)
  local base=0 pid cpus
  for pid in "${pids[@]}"; do
    cpus="${base}-$((base + 7))"
    taskset -apc "$cpus" "$pid" >/dev/null 2>&1 || true
    base=$((base + 8))
  done
  if ((${#pids[@]} > 0)); then
    echo "[lolbo-g012] pinned ${#pids[@]} RL jobs to cores 0-$((base - 1))" | tee -a "$MASTER"
  fi
}

pin_tpe_jobs() {
  local pids=()
  mapfile -t pids < <(pgrep -f 'run_bolt_bo_suite.py.*tr25k_strict' || true)
  local pid
  local core=8
  for pid in "${pids[@]}"; do
    taskset -apc "${core}-$((core + 7))" "$pid" >/dev/null 2>&1 || true
    core=$((core + 8))
  done
  if ((${#pids[@]} > 0)); then
    echo "[lolbo-g012] pinned ${#pids[@]} TPE jobs starting core 8" | tee -a "$MASTER"
  fi
}

pin_rl_jobs
pin_tpe_jobs
echo "[lolbo-g012] $(date -Is) GPUs 0-3 x${WORKERS_PER_GPU} workers threads=${LOLBO_THREADS} budget=${MAX_ORACLE_BUDGET}" | tee -a "$MASTER"
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
  echo "[lolbo-g012] gpu=${gpu} shard=${shard} cpus=${cpus} -> ${log}" | tee -a "$MASTER"
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

# Full 8-shard layout: GPUs 0-3 × 2 workers (GPU3 shares with TPE).
declare -a PLAN=(
  "0:0:64-67"
  "0:4:68-71"
  "1:1:72-75"
  "1:5:76-79"
  "2:2:80-83"
  "2:6:84-87"
  "3:3:88-91"
  "3:7:92-95"
)

for entry in "${PLAN[@]}"; do
  IFS=: read -r gpu shard cpus <<<"$entry"
  launch_shard "$gpu" "$shard" "$cpus" "$LOGDIR/bo_protein_bolt_lolbo_g${gpu}_${shard}.log"
done

echo "[lolbo-g012] pids: $(tr '\n' ' ' <"$PIDFILE")" | tee -a "$MASTER"
