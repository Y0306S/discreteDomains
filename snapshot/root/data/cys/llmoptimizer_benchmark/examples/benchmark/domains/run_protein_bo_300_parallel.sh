#!/usr/bin/env bash
# Resume peptide TPE from 240 → 300 trials. Disjoint offsets, skip-done, 8 CPU threads.
set -euo pipefail

PY="${PY:-/root/data/cys/blockdata_archive/miniconda3/envs/hpo_benchmark/bin/python}"
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
LOGDIR="/root/data/logs"
mkdir -p "$LOGDIR"
THREADS="${PEPTIDE_BO_THREADS:-8}"
MASTER="$LOGDIR/bo_protein_300_parallel.log"

export PYTHONPATH="$REPO"

echo "[protein-300] $(date -Is) launching 4 shards threads=$THREADS n_trials=300" | tee "$MASTER"

CUDA_VISIBLE_DEVICES=0 "$PY" -u "$REPO/examples/benchmark/domains/run_bolt_bo_suite.py" \
  --domain protein --n-datasets 250 --offset 0 --n-trials 300 --device cuda:0 \
  --optimizer tpe --seed 1 --cpu-threads "$THREADS" --skip-done \
  >>"$LOGDIR/bo_protein_300_gpu0.log" 2>&1 &
p0=$!

CUDA_VISIBLE_DEVICES=1 "$PY" -u "$REPO/examples/benchmark/domains/run_bolt_bo_suite.py" \
  --domain protein --n-datasets 250 --offset 250 --n-trials 300 --device cuda:0 \
  --optimizer tpe --seed 1 --cpu-threads "$THREADS" --skip-done \
  >>"$LOGDIR/bo_protein_300_gpu1.log" 2>&1 &
p1=$!

CUDA_VISIBLE_DEVICES=2 "$PY" -u "$REPO/examples/benchmark/domains/run_bolt_bo_suite.py" \
  --domain protein --n-datasets 250 --offset 500 --n-trials 300 --device cuda:0 \
  --optimizer tpe --seed 1 --cpu-threads "$THREADS" --skip-done \
  >>"$LOGDIR/bo_protein_300_gpu2.log" 2>&1 &
p2=$!

CUDA_VISIBLE_DEVICES=3 "$PY" -u "$REPO/examples/benchmark/domains/run_bolt_bo_suite.py" \
  --domain protein --n-datasets 250 --offset 750 --n-trials 300 --device cuda:0 \
  --optimizer tpe --seed 1 --cpu-threads "$THREADS" --skip-done \
  >>"$LOGDIR/bo_protein_300_gpu3.log" 2>&1 &
p3=$!

echo "pids gpu0=$p0 gpu1=$p1 gpu2=$p2 gpu3=$p3" | tee -a "$MASTER"
wait "$p0" "$p1" "$p2" "$p3"
echo "[protein-300] $(date -Is) all shards finished" | tee -a "$MASTER"
