#!/usr/bin/env bash
# Vanilla plain TPE (no trust region) + feasible repair + novel-seq dedup.
# LOLBO-aligned: 1000 seed_0 init + 25k post-init on pep_050 / pep_051.
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
PY="${PY:-/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python}"
LOGDIR="${LOGDIR:-/root/data/logs}"
GPU="${CUDA_VISIBLE_DEVICES:-1}"
THREADS="${PEPTIDE_BO_THREADS:-4}"
E2E_FREQ="${E2E_FREQ:-400}"
BENCH_SUBDIR="${BENCH_SUBDIR:-bo80_bolt_vanilla_tpe_dedup_cmp050051}"
HIST_SUFFIX="${HIST_SUFFIX:-tpe_vanilla_dedup_bolt_native.json}"
mkdir -p "$LOGDIR"
export PYTHONPATH="$REPO"

cd "$REPO"

COMMON=(
  examples/benchmark/domains/run_bolt_bo_suite.py
  --domain protein
  --n-trials 25000
  --device cuda:0
  --optimizer tpe
  --allow-vanilla-tpe
  --seed 1
  --cpu-threads "$THREADS"
  --skip-done
  --protein-oracle bolt_native
  --min-similarity 0.75
  --constraint-index 0
  --bench-subdir "$BENCH_SUBDIR"
  --bolt-init-points 1000
  --max-string-length 30
  --e2e-freq "$E2E_FREQ"
  --bolt-reject-dup-sequences
  --bolt-history-suffix "$HIST_SUFFIX"
)

MASTER="$LOGDIR/tpe_vanilla_dedup_cmp050051_launch.log"
PIDFILE="$LOGDIR/tpe_vanilla_dedup_cmp050051.pids"
: >"$PIDFILE"

launch() {
  local task_id="$1" cpus="$2" log="$3"
  echo "[vanilla-tpe-dedup] $(date -Is) task=${task_id} gpu=${GPU} cpus=${cpus} -> ${log}" | tee -a "$MASTER"
  CUDA_VISIBLE_DEVICES="$GPU" \
  OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
  NUMEXPR_NUM_THREADS="$THREADS" \
  taskset -c "$cpus" "$PY" -u "${COMMON[@]}" \
    --task-ids "$task_id" \
    >>"$log" 2>&1 &
  echo "$!" >>"$PIDFILE"
}

# CPU 56-63 (outside RL 0-23 and LOLBO 64-95); GPU 1 by default.
launch pep_050 "56-59" "$LOGDIR/tpe_vanilla_dedup_cmp_pep050.log"
launch pep_051 "60-63" "$LOGDIR/tpe_vanilla_dedup_cmp_pep051.log"

echo "[vanilla-tpe-dedup] pids: $(tr '\n' ' ' <"$PIDFILE")" | tee -a "$MASTER"
