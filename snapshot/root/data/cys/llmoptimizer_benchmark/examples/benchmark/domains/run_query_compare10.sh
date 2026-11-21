#!/usr/bin/env bash
# 10 CEB tasks × 80 trials: trust-region TPE (pg 5432) and native BOLT LOLBO (pg 5433).
# Baseline: existing arm_c_bo_v2 CMA-ES+e2e LOLBO on the same tasks.
set -euo pipefail
REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
PY="${PY:-/root/data/cys/blockdata_archive/miniconda3/envs/hpo_benchmark/bin/python}"
LOGDIR="${LOGDIR:-/root/data/logs}"
mkdir -p "$LOGDIR"
cd "$REPO"
export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1
TASKS="CEB_11A114,CEB_11A53,CEB_11A97,CEB_11A13,CEB_11A94,CEB_11A99,CEB_11A133,CEB_11A16,CEB_11B27,CEB_2A75"

echo "[query-10] $(date -Is) tr_tpe + bolt_lolbo n_trials=80" | tee "$LOGDIR/query_compare10.log"

CUDA_VISIBLE_DEVICES=0 nohup "$PY" -u examples/benchmark/domains/run_query_arm_c_bo.py \
  --optimizer tr_tpe --n-trials 80 --seed 1 --device cuda:0 --skip-done \
  --task-ids "$TASKS" \
  --dsn "host=127.0.0.1 port=5432 dbname=imdb user=imdb password=imdb" \
  >>"$LOGDIR/query_tr_tpe_consistent10.log" 2>&1 &
echo "tr_tpe pid=$! log=$LOGDIR/query_tr_tpe_consistent10.log"

CUDA_VISIBLE_DEVICES=1 nohup "$PY" -u examples/benchmark/domains/run_query_bolt_lolbo.py \
  --n-trials 80 --seed 1 --device cuda:0 --skip-done \
  --task-ids "$TASKS" \
  --dsn "host=127.0.0.1 port=5433 dbname=imdb user=imdb password=imdb" \
  >>"$LOGDIR/query_bolt_lolbo_consistent10.log" 2>&1 &
echo "bolt_lolbo pid=$! log=$LOGDIR/query_bolt_lolbo_consistent10.log"
