#!/usr/bin/env bash
# BOLT-native peptide BO: bacteria_0 objective + 75% template similarity (seed_0 protocol).
# Runs 4 shards sequentially on GPU 3 to avoid OOM alongside existing TPE jobs on GPUs 0-2.
set -euo pipefail

PY="${PY:-/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python}"
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
SCRIPT="$REPO/examples/benchmark/domains/run_protein_bolt_lolbo.py"
LOGDIR="/root/data/logs"
mkdir -p "$LOGDIR"
GPU="${BOLT_NATIVE_GPU:-3}"
N_TRIALS="${BOLT_NATIVE_TRIALS:-100}"

export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES="$GPU"

MASTER="$LOGDIR/bo_protein_bolt_native_parallel.log"
echo "[bolt-native] $(date -Is) gpu=$GPU n_trials=$N_TRIALS sequential 4 shards" | tee "$MASTER"

run_shard() {
  local shard="$1"
  echo "[bolt-native] shard $shard start $(date -Is)" | tee -a "$MASTER"
  "$PY" -u "$SCRIPT" \
    --oracle-mode bolt_native \
    --n-trials "$N_TRIALS" \
    --seed 1 \
    --device-id 0 \
    --n-tasks 1000 \
    --shard-index "$shard" \
    --n-shards 4 \
    --min-similarity 0.75 \
    --skip-done \
    >>"$LOGDIR/bo_protein_bolt_native_s${shard}.log" 2>&1
  echo "[bolt-native] shard $shard done $(date -Is)" | tee -a "$MASTER"
}

for s in 0 1 2 3; do
  run_shard "$s"
done

echo "[bolt-native] $(date -Is) all shards finished" | tee -a "$MASTER"
