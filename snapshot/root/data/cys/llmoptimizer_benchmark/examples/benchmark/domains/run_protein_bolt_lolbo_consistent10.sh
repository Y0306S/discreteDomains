#!/usr/bin/env bash
# Full BOLT LOLBO (GP + TuRBO trust region + E2E VAE) on consistent-10 peptides × 240 trials.
set -euo pipefail
REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
PY="${PY:-/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python3}"
LOGDIR="${LOGDIR:-/root/data/logs}"
mkdir -p "$LOGDIR"
cd "$REPO"
export PYTHONPATH="$REPO"
TASKS="pep_009,pep_791,pep_231,pep_291,pep_277,pep_726,pep_012,pep_813,pep_474,pep_124"
LOG="$LOGDIR/bo_protein_bolt_lolbo_consistent10.log"
echo "[bolt-lolbo-10] $(date -Is) n_trials=240 tasks=$TASKS" | tee "$LOG"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" -u \
  examples/benchmark/domains/run_protein_bolt_lolbo.py \
  --n-trials 240 --seed 1 --device-id 0 --skip-done \
  --task-ids "$TASKS" \
  2>&1 | tee -a "$LOG"
echo "[bolt-lolbo-10] $(date -Is) done" | tee -a "$LOG"
