#!/usr/bin/env bash
# Trust-region TPE aligned with LOLBO: strict decode (no repair), E2E every 10 stalls, dup resample.
# For feasibility repair (snap to template), use run_protein_bolt_tr25k_lolbo_tpe_repair_parallel.sh
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
  --bench-subdir bo80_bolt_tr25k_lolbo_tpe
  --bolt-init-points 1000
  --max-string-length 30
  --bolt-strict-decode
)

launch() {
  local tid="$1" cpus="$2" log="$3"
  echo "[lolbo-tpe] $(date -Is) start $tid cpus=$cpus log=$log" | tee -a "$LOGDIR/tpe_tr25k_lolbo_tpe_launch.log"
  CUDA_VISIBLE_DEVICES="$GPU" \
  OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
  taskset -c "$cpus" "$PY" -u "${COMMON[@]}" --task-ids "$tid" >>"$log" 2>&1 &
  echo "$!" >>"$LOGDIR/tpe_tr25k_lolbo_tpe.pids"
}

: >"$LOGDIR/tpe_tr25k_lolbo_tpe.pids"
launch pep_050 "8-15" "$LOGDIR/tpe_tr25k_lolbo_tpe_pep050.log"
launch pep_051 "16-23" "$LOGDIR/tpe_tr25k_lolbo_tpe_pep051.log"
echo "[lolbo-tpe] pids: $(tr '\n' ' ' <"$LOGDIR/tpe_tr25k_lolbo_tpe.pids")" | tee -a "$LOGDIR/tpe_tr25k_lolbo_tpe_launch.log"
