#!/usr/bin/env bash
# Full BOLT benchmark: trust-region TPE + feasible repair + novel-seq dedup.
# LOLBO-aligned budget: 1000 seed_0 init + 25k post-init oracle calls per task.
# Valid peptide: min_similarity>=0.75 vs REFERENCE_SEQUENCE[0], standard AAs, <=1 Cys.
# Distinct from prior runs (strict / repair-only pilots) via --bolt-reject-dup-sequences
# and bench subdir bo80_bolt_tr25k_lolbo_tpe_repair_dedup.
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
PY="${PY:-/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python}"
LOGDIR="${LOGDIR:-/root/data/logs}"
GPU="${CUDA_VISIBLE_DEVICES:-3}"
THREADS="${PEPTIDE_BO_THREADS:-4}"
E2E_FREQ="${E2E_FREQ:-400}"
N_TASKS="${N_TASKS:-1000}"
N_SHARDS="${N_SHARDS:-8}"
mkdir -p "$LOGDIR"
export PYTHONPATH="$REPO"

MASTER="$LOGDIR/tpe_tr25k_lolbo_tpe_repair_dedup_launch.log"
PIDFILE="$LOGDIR/tpe_tr25k_lolbo_tpe_repair_dedup.pids"

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
  --bench-subdir bo80_bolt_tr25k_lolbo_tpe_repair_dedup
  --bolt-init-points 1000
  --max-string-length 30
  --e2e-freq "$E2E_FREQ"
  --bolt-reject-dup-sequences
)

per_shard=$(( (N_TASKS + N_SHARDS - 1) / N_SHARDS ))

echo "[tpe-repair-dedup] $(date -Is) gpu=${GPU} shards=${N_SHARDS} tasks=${N_TASKS} per_shard=${per_shard} threads=${THREADS} e2e_freq=${E2E_FREQ}" | tee "$MASTER"
: >"$PIDFILE"

launch_shard() {
  local shard="$1" offset="$2" cpus="$3" log="$4"
  echo "[tpe-repair-dedup] shard=${shard} offset=${offset} n=${per_shard} cpus=${cpus} -> ${log}" | tee -a "$MASTER"
  CUDA_VISIBLE_DEVICES="$GPU" \
  OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
  NUMEXPR_NUM_THREADS="$THREADS" \
  taskset -c "$cpus" "$PY" -u "${COMMON[@]}" \
    --n-datasets "$per_shard" --offset "$offset" \
    >>"$log" 2>&1 &
  echo "$!" >>"$PIDFILE"
}

# CPU 24-55 (8 shards x 4 cores); avoids RL 0-23 and LOLBO 64-95.
declare -a PLAN=(
  "0:0:24-27"
  "1:125:28-31"
  "2:250:32-35"
  "3:375:36-39"
  "4:500:40-43"
  "5:625:44-47"
  "6:750:48-51"
  "7:875:52-55"
)

for entry in "${PLAN[@]}"; do
  IFS=: read -r shard offset cpus <<<"$entry"
  if (( offset >= N_TASKS )); then
    continue
  fi
  launch_shard "$shard" "$offset" "$cpus" "$LOGDIR/tpe_tr25k_lolbo_tpe_repair_dedup_s${shard}.log"
done

echo "[tpe-repair-dedup] pids: $(tr '\n' ' ' <"$PIDFILE")" | tee -a "$MASTER"
