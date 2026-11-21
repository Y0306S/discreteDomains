#!/usr/bin/env bash
# BOLT seed_0 oracle (bacteria_0 MIC + 75% template similarity) via fast TPE in VAE space.
# 1000 tasks x 500 trials, 4 GPUs x 250 tasks, 4 CPU threads/shard to reduce contention.
set -euo pipefail

PY="${PY:-/root/data/cys/blockdata_archive/miniconda3/envs/hpo_benchmark/bin/python}"
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
LOGDIR="/root/data/logs"
mkdir -p "$LOGDIR"
THREADS="${PEPTIDE_BO_THREADS:-4}"
MASTER="$LOGDIR/bo_protein_bolt_native_tpe_parallel.log"

export PYTHONPATH="$REPO"

echo "[protein-bolt-native-tpe] $(date -Is) launching 4 shards threads=$THREADS n_trials=500" | tee "$MASTER"

launch() {
  local gpu="$1" offset="$2" log="$3"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$REPO/examples/benchmark/domains/run_bolt_bo_suite.py" \
    --domain protein --n-datasets 250 --offset "$offset" --n-trials 500 --device cuda:0 \
    --optimizer tpe --seed 1 --cpu-threads "$THREADS" --skip-done \
    --protein-oracle bolt_native --min-similarity 0.75 --bench-subdir bo80_bolt_native \
    >>"$log" 2>&1 &
  echo "$!"
}

p0=$(launch 0 0 "$LOGDIR/bo_protein_bolt_native_tpe_gpu0.log")
p1=$(launch 1 250 "$LOGDIR/bo_protein_bolt_native_tpe_gpu1.log")
p2=$(launch 2 500 "$LOGDIR/bo_protein_bolt_native_tpe_gpu2.log")
p3=$(launch 3 750 "$LOGDIR/bo_protein_bolt_native_tpe_gpu3.log")

echo "pids gpu0=$p0 gpu1=$p1 gpu2=$p2 gpu3=$p3" | tee -a "$MASTER"
wait "$p0" "$p1" "$p2" "$p3"
echo "[protein-bolt-native-tpe] $(date -Is) all shards finished" | tee -a "$MASTER"
