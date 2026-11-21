#!/usr/bin/env bash
# LOLBO bsz=10 fleet (constraint_index=pep_idx patched): 1000 peptides, 1000 post-init calls (100 rounds × 10).
# Default: 1 worker per GPU on GPUs 0-3. Override GPU_IDS if some cards are busy.
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
PY="${PY:-/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python}"
LOGDIR="/root/data/logs"
OUT_ROOT="${OUT_ROOT:-/root/data/cys/blockdata_archive/domain_protein/bo80_bolt_bsz10_r100}"
N_TASKS="${N_TASKS:-1000}"
N_SHARDS="${N_SHARDS:-}"
LOLBO_THREADS="${LOLBO_THREADS:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
# Extra LOLBO shards pinned to the last GPU in GPU_IDS (default 0 = one shard per GPU only).
EXTRA_ON_LAST_GPU="${EXTRA_ON_LAST_GPU:-0}"
mkdir -p "$OUT_ROOT" "$LOGDIR"
export PYTHONPATH="$REPO"
export PROTEIN_BO80_BOLT_ROOT="$OUT_ROOT"
export BOLT_MIN_FREE_GB="${BOLT_MIN_FREE_GB:-30}"

MASTER="$LOGDIR/bolt_bsz10_r100_fleet_launch.log"
PIDFILE="$LOGDIR/bolt_bsz10_r100_fleet.pids"
cd "$REPO"

IFS=',' read -r -a GPUS <<<"$GPU_IDS"
NGPU="${#GPUS[@]}"
if [[ -z "$N_SHARDS" ]]; then
  N_SHARDS=$((NGPU + EXTRA_ON_LAST_GPU))
fi

COMMON=(
  examples/benchmark/domains/run_protein_bolt_lolbo.py
  --run-mode continue_bolt
  --max-oracle-budget 1000
  --n-trials 1000
  --bolt-init-points 1000
  --bsz 10
  --e2e-freq 33
  --oracle-mode bolt_native
  --skip-done
  --n-tasks "$N_TASKS"
  --n-shards "$N_SHARDS"
)

echo "[bsz10-r100] $(date -Is) n_tasks=$N_TASKS shards=$N_SHARDS out=$OUT_ROOT gpus=${GPU_IDS} extra_on_last_gpu=$EXTRA_ON_LAST_GPU" | tee "$MASTER"
: >"$PIDFILE"

launch_shard() {
  local shard="$1" gpu="$2" cpus="$3" log="$4"
  echo "[bsz10-r100] shard=$shard gpu=$gpu cpus=$cpus -> $log" | tee -a "$MASTER"
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

# Shards 0..NGPU-2 → one per GPU; remaining shards → last GPU (default: 3 workers on GPU 3).
gpu_for_shard() {
  local shard="$1"
  if (( shard < NGPU - 1 )); then
    echo "${GPUS[$shard]}"
  else
    echo "${GPUS[$((NGPU - 1))]}"
  fi
}

BASE_CPU=64
for ((shard=0; shard<N_SHARDS; shard++)); do
  gpu="$(gpu_for_shard "$shard")"
  cpu_start=$((BASE_CPU + shard * 8))
  cpu_end=$((cpu_start + 7))
  launch_shard "$shard" "$gpu" "${cpu_start}-${cpu_end}" "$LOGDIR/bolt_bsz10_r100_fleet_s${shard}.log"
done

echo "[bsz10-r100] pids: $(tr '\n' ' ' <"$PIDFILE")" | tee -a "$MASTER"
echo "[bsz10-r100] monitor: tail -f $LOGDIR/bolt_bsz10_r100_fleet_s0.log" | tee -a "$MASTER"
