#!/usr/bin/env bash
# Arm C BO: 100 CEB tasks, 3-way shard across isolated Postgres (5432/5433/5434).
set -euo pipefail
export PYTHONPATH=/root/data/cys/blockdata_archive/llmoptimizer_benchmark
export PYTHONUNBUFFERED=1
PY=/root/miniconda3/envs/hpo_benchmark/bin/python
SCRIPT=/root/data/cys/blockdata_archive/llmoptimizer_benchmark/examples/benchmark/domains/run_query_arm_c_bo.py
LOG=/root/data/logs
mkdir -p "$LOG"

N_TASKS="${N_TASKS:-100}"
N_TRIALS="${N_TRIALS:-80}"

echo "[arm-c-100] $(date -Is) launch n_tasks=$N_TASKS n_trials=$N_TRIALS"

# Shard 0 → pg5432 → GPU 2
CUDA_VISIBLE_DEVICES=2 nohup "$PY" -u "$SCRIPT" \
  --n-tasks "$N_TASKS" --n-shards 3 --shard-index 0 --n-trials "$N_TRIALS" \
  --device cuda:0 \
  --dsn "host=127.0.0.1 port=5432 dbname=imdb user=imdb password=imdb" \
  > "$LOG/query_arm_c_s0_pg5432.log" 2>&1 &
PID0=$!
echo "[arm-c-100] shard0 pid=$PID0 log=$LOG/query_arm_c_s0_pg5432.log gpu=2 port=5432"

# Shard 1 → pg5433 → GPU 3
CUDA_VISIBLE_DEVICES=3 nohup "$PY" -u "$SCRIPT" \
  --n-tasks "$N_TASKS" --n-shards 3 --shard-index 1 --n-trials "$N_TRIALS" \
  --device cuda:0 \
  --dsn "host=127.0.0.1 port=5433 dbname=imdb user=imdb password=imdb" \
  > "$LOG/query_arm_c_s1_pg5433.log" 2>&1 &
PID1=$!
echo "[arm-c-100] shard1 pid=$PID1 log=$LOG/query_arm_c_s1_pg5433.log gpu=3 port=5433"

# Shard 2 → pg5434 → GPU 1 (headroom; GPU0 busy with peptide BO)
CUDA_VISIBLE_DEVICES=1 nohup "$PY" -u "$SCRIPT" \
  --n-tasks "$N_TASKS" --n-shards 3 --shard-index 2 --n-trials "$N_TRIALS" \
  --device cuda:0 \
  --dsn "host=127.0.0.1 port=5434 dbname=imdb user=imdb password=imdb" \
  > "$LOG/query_arm_c_s2_pg5434.log" 2>&1 &
PID2=$!
echo "[arm-c-100] shard2 pid=$PID2 log=$LOG/query_arm_c_s2_pg5434.log gpu=1 port=5434"

echo "[arm-c-100] PIDs: $PID0 $PID1 $PID2"
echo "[arm-c-100] index: benchmark_data/domain_query/arm_c_bo_v2/index.jsonl"
