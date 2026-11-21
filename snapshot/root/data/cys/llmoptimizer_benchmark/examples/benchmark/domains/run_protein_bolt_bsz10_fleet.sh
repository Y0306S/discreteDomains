#!/usr/bin/env bash
# LOLBO bsz=10 fleet: 1000 peptides, 1500 post-init oracle calls (150 rounds × 10).
# e2e_freq=50 (same ~500-call stall as bsz=50 + e2e_freq=10).
# 4 GPUs × 2 workers/GPU (8 shards), skip already-finished tasks.
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
PY="${PY:-/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python}"
LOGDIR="/root/data/logs"
OUT_ROOT="${OUT_ROOT:-/root/data/cys/blockdata_archive/domain_protein/bo80_bolt_bsz10_pilot5}"
N_TASKS="${N_TASKS:-1000}"
N_SHARDS="${N_SHARDS:-8}"
LOLBO_THREADS="${LOLBO_THREADS:-8}"
mkdir -p "$OUT_ROOT" "$LOGDIR"
export PYTHONPATH="$REPO"
export PROTEIN_BO80_BOLT_ROOT="$OUT_ROOT"
export BOLT_MIN_FREE_GB="${BOLT_MIN_FREE_GB:-30}"

COMMON=(
  examples/benchmark/domains/run_protein_bolt_lolbo.py
  --run-mode continue_bolt
  --max-oracle-budget 1500
  --n-trials 1500
  --bolt-init-points 1000
  --bsz 10
  --e2e-freq 50
  --oracle-mode bolt_native
  --skip-done
  --n-tasks "$N_TASKS"
  --n-shards "$N_SHARDS"
)

MASTER="$LOGDIR/bolt_bsz10_fleet_launch.log"
PIDFILE="$LOGDIR/bolt_bsz10_fleet.pids"
cd "$REPO"

echo "[bsz10-fleet] $(date -Is) n_tasks=$N_TASKS shards=$N_SHARDS out=$OUT_ROOT (${LOLBO_THREADS} threads/worker, 2/GPU)" | tee "$MASTER"
: >"$PIDFILE"

launch_shard() {
  local shard="$1" gpu="$2" cpus="$3" log="$4"
  echo "[bsz10-fleet] shard=$shard gpu=$gpu cpus=$cpus -> $log" | tee -a "$MASTER"
  CUDA_VISIBLE_DEVICES="$gpu" \
  OMP_NUM_THREADS="$LOLBO_THREADS" \
  MKL_NUM_THREADS="$LOLBO_THREADS" \
  OPENBLAS_NUM_THREADS="$LOLBO_THREADS" \
  NUMEXPR_NUM_THREADS="$LOLBO_THREADS" \
  TORCH_NUM_THREADS="$LOLBO_THREADS" \
  taskset -c "$cpus" nohup "$PY" -u "${COMMON[@]}" \
    --shard-index "$shard" --device-id 0 \
    >>"$log" 2>&1 &
  echo "$!" >>"$PIDFILE"
}

# GPU 0-3 × 2 workers. LOLBO on cores 64-127 (8 cores each).
launch_shard 0 0 "64-71"  "$LOGDIR/bolt_bsz10_fleet_s0_gpu0a.log"
launch_shard 4 0 "72-79"  "$LOGDIR/bolt_bsz10_fleet_s4_gpu0b.log"
launch_shard 1 1 "80-87"  "$LOGDIR/bolt_bsz10_fleet_s1_gpu1a.log"
launch_shard 5 1 "88-95"  "$LOGDIR/bolt_bsz10_fleet_s5_gpu1b.log"
launch_shard 2 2 "96-103" "$LOGDIR/bolt_bsz10_fleet_s2_gpu2a.log"
launch_shard 6 2 "104-111" "$LOGDIR/bolt_bsz10_fleet_s6_gpu2b.log"
launch_shard 3 3 "112-119" "$LOGDIR/bolt_bsz10_fleet_s3_gpu3a.log"
launch_shard 7 3 "120-127" "$LOGDIR/bolt_bsz10_fleet_s7_gpu3b.log"

echo "[bsz10-fleet] pids: $(tr '\n' ' ' <"$PIDFILE")" | tee -a "$MASTER"
echo "[bsz10-fleet] monitor: tail -f $LOGDIR/bolt_bsz10_fleet_s0_gpu0a.log" | tee -a "$MASTER"
