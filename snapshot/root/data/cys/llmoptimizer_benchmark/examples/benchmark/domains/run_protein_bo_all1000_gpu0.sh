#!/usr/bin/env bash
# Resume TPE BO on all 1000 BOLT start sequences: pep_000 .. pep_999.
# Extends existing 80-trial histories to 160 trials (no re-eval of the first 80).
# Uses --skip-done: skips tasks already at >=160 observations.
# GPU0 only — do not share with query C3 on GPUs 1–3.
set -euo pipefail

PY="${PY:-/root/miniconda3/envs/hpo_benchmark/bin/python}"
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
LOGDIR="/root/data/logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/bo_protein_all1000_gpu0_160.log"

export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES=0

exec 9>"$LOGDIR/bo_protein_gpu0.lock"
echo "[protein-all1000-160] $(date -Is) waiting for lock..."
flock 9
echo "[protein-all1000-160] $(date -Is) lock acquired"

while pgrep -f 'run_bolt_bo_suite.py.*--domain protein' >/dev/null 2>&1; do
  echo "[protein-all1000-160] $(date -Is) waiting for existing protein BO process..."
  sleep 30
done

{
  echo "[protein-all1000-160] $(date -Is) resume offsets 0..900 n_trials=160"
  for offset in 0 100 200 300 400 500 600 700 800 900; do
    echo "[protein-all1000-160] $(date -Is) offset=$offset n_datasets=100"
    "$PY" "$REPO/examples/benchmark/domains/run_bolt_bo_suite.py" \
      --domain protein \
      --n-datasets 100 \
      --offset "$offset" \
      --n-trials 160 \
      --device cuda:0 \
      --optimizer tpe \
      --seed 1 \
      --skip-done
    echo "[protein-all1000-160] $(date -Is) offset=$offset done"
  done
  echo "[protein-all1000-160] $(date -Is) all 1000 slots finished at 160 trials"
} >>"$LOG" 2>&1
