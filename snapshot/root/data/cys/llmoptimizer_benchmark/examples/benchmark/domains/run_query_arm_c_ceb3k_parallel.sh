#!/usr/bin/env bash
# Arm C BO: 3033 CEB-3k tasks × 80 LOLBO, 3-way shard on isolated IMDB clones.
# Skip-done: first 100 already in arm_c_bo_v2.
set -euo pipefail
export PYTHONPATH=/root/data/cys/blockdata_archive/llmoptimizer_benchmark
export PYTHONUNBUFFERED=1
PY=/root/miniconda3/envs/hpo_benchmark/bin/python
SCRIPT=/root/data/cys/blockdata_archive/llmoptimizer_benchmark/examples/benchmark/domains/run_query_arm_c_bo.py
LOG=/root/data/logs
mkdir -p "$LOG"
MASTER="$LOG/query_arm_c_ceb3k_parallel.log"

N_TASKS="${N_TASKS:-3033}"
N_TRIALS="${N_TRIALS:-80}"
CKPT="${CKPT:-/root/data/cys/BayesQO/bayes_lqo/vae/CEB_64.ckpt}"
OUT="${OUT:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark/benchmark_data/domain_query/arm_c_bo_v2}"

echo "[arm-c-3k] $(date -Is) n_tasks=$N_TASKS n_trials=$N_TRIALS ckpt=$CKPT" | tee "$MASTER"

CUDA_VISIBLE_DEVICES=2 nohup "$PY" -u "$SCRIPT" \
  --n-tasks "$N_TASKS" --n-shards 3 --shard-index 0 --n-trials "$N_TRIALS" \
  --arm C --ckpt "$CKPT" --out-root "$OUT" --skip-done --device cuda:0 \
  --dsn "host=127.0.0.1 port=5432 dbname=imdb user=imdb password=imdb" \
  >>"$LOG/query_arm_c_ceb3k_s0_pg5432.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=3 nohup "$PY" -u "$SCRIPT" \
  --n-tasks "$N_TASKS" --n-shards 3 --shard-index 1 --n-trials "$N_TRIALS" \
  --arm C --ckpt "$CKPT" --out-root "$OUT" --skip-done --device cuda:0 \
  --dsn "host=127.0.0.1 port=5433 dbname=imdb user=imdb password=imdb" \
  >>"$LOG/query_arm_c_ceb3k_s1_pg5433.log" 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=1 nohup "$PY" -u "$SCRIPT" \
  --n-tasks "$N_TASKS" --n-shards 3 --shard-index 2 --n-trials "$N_TRIALS" \
  --arm C --ckpt "$CKPT" --out-root "$OUT" --skip-done --device cuda:0 \
  --dsn "host=127.0.0.1 port=5434 dbname=imdb user=imdb password=imdb" \
  >>"$LOG/query_arm_c_ceb3k_s2_pg5434.log" 2>&1 &
PID2=$!

echo "pids gpu2/5432=$PID0 gpu3/5433=$PID1 gpu1/5434=$PID2" | tee -a "$MASTER"
echo "logs $LOG/query_arm_c_ceb3k_s{0,1,2}_pg543{2,3,4}.log" | tee -a "$MASTER"
wait "$PID0" "$PID1" "$PID2"
echo "[arm-c-3k] $(date -Is) all shards finished" | tee -a "$MASTER"
