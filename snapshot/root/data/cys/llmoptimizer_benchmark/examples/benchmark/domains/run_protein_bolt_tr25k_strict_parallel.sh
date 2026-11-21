#!/usr/bin/env bash
# Trust-region TPE with LOLBO-style strict decode (no feasibility repair).
# Writes to bo80_bolt_tr25k_strict (old repair runs stay in bo80_bolt_tr25k).
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
PY="${PY:-/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python}"
LOGDIR="${LOGDIR:-/root/data/logs}"
GPU="${CUDA_VISIBLE_DEVICES:-3}"
THREADS="${PEPTIDE_BO_THREADS:-8}"
mkdir -p "$LOGDIR"
export PYTHONPATH="$REPO"

cd "$REPO"

COMMON=(
  examples/benchmark/domains/run_bolt_bo_suite.py
  --domain protein
  --n-trials 25000
  --device cuda:0
  --optimizer tr_tpe
  --seed 1
  --cpu-threads "$THREADS"
  --skip-done
  --protein-oracle bolt_native
  --min-similarity 0.75
  --constraint-index 0
  --bench-subdir bo80_bolt_tr25k_strict
  --bolt-init-points 1000
  --max-string-length 30
  --bolt-strict-decode
)

launch() {
  local tid="$1" cpus="$2" log="$3"
  echo "[tr25k-strict] $(date -Is) start $tid cpus=$cpus log=$log" | tee -a "$LOGDIR/tpe_tr25k_strict_launch.log"
  CUDA_VISIBLE_DEVICES="$GPU" \
  OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
  taskset -c "$cpus" "$PY" -u "${COMMON[@]}" --task-ids "$tid" >>"$log" 2>&1 &
  echo "$!" >>"$LOGDIR/tpe_tr25k_strict.pids"
}

: >"$LOGDIR/tpe_tr25k_strict.pids"
launch pep_050 "0-7" "$LOGDIR/tpe_tr25k_strict_pep050.log"
launch pep_051 "8-15" "$LOGDIR/tpe_tr25k_strict_pep051.log"
echo "[tr25k-strict] pids: $(tr '\n' ' ' <"$LOGDIR/tpe_tr25k_strict.pids")" | tee -a "$LOGDIR/tpe_tr25k_strict_launch.log"
