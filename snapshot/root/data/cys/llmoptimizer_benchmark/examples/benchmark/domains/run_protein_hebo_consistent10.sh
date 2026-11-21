#!/usr/bin/env bash
# HEBO vs TPE on 10 AMP tasks where TPE kept improving across 240 trials.
set -euo pipefail
REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
PY="${PY:-/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python}"
LOGDIR="${LOGDIR:-/root/data/logs}"
mkdir -p "$LOGDIR"
cd "$REPO"
export PYTHONPATH="$REPO"
TASKS="pep_009,pep_791,pep_231,pep_291,pep_277,pep_726,pep_012,pep_813,pep_474,pep_124"
LOG="$LOGDIR/bo_protein_hebo_consistent10.log"
echo "[hebo-10] $(date -Is) n_trials=240 tasks=$TASKS" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" -u \
  examples/benchmark/domains/run_bolt_bo_suite.py \
  --domain protein --n-trials 240 --optimizer hebo --seed 1 \
  --device cuda:0 --cpu-threads 8 --skip-done \
  --task-ids "$TASKS" \
  2>&1 | tee -a "$LOG"
echo "[hebo-10] $(date -Is) done" | tee -a "$LOG"
