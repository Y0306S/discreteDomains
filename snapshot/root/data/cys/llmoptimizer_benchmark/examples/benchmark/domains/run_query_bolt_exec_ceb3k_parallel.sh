#!/usr/bin/env bash
# Paper-style CEB-3K: BOLT execution oracle × 80 trials.
#   - bolt_lolbo (GP + TuRBO / BoTorch) with 49-point Bao init
#   - tpe ablation
# 3-way shard on isolated IMDB clones (ports 5432/5433/5434).
set -euo pipefail
export PYTHONPATH=/root/data/cys/blockdata_archive/llmoptimizer_benchmark
export PYTHONUNBUFFERED=1

PY_LOLBO=/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python
PY_TPE=/root/data/cys/blockdata_archive/miniconda3/envs/hpo_benchmark/bin/python
LOLBO=/root/data/cys/blockdata_archive/llmoptimizer_benchmark/examples/benchmark/domains/run_query_bolt_lolbo.py
TPE=/root/data/cys/blockdata_archive/llmoptimizer_benchmark/examples/benchmark/domains/run_query_bolt_tpe.py
LOG=/root/data/logs
mkdir -p "$LOG"
MASTER="$LOG/query_bolt_exec_ceb3k_parallel.log"

N_TASKS="${N_TASKS:-3033}"
N_TRIALS="${N_TRIALS:-80}"
TIMEOUT="${EXECUTION_TIMEOUT:-100}"
CKPT="${CKPT:-/root/data/cys/BayesQO/bayes_lqo/vae/CEB_64.ckpt}"
OUT="${OUT:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark/benchmark_data/domain_query/arm_c_bo_v2/bolt_execute}"

echo "[bolt-exec-3k] $(date -Is) n_tasks=$N_TASKS n_trials=$N_TRIALS timeout=${TIMEOUT}s" | tee "$MASTER"

# LOLBO shards (GPUs 2/3/1)
CUDA_VISIBLE_DEVICES=2 nohup "$PY_LOLBO" -u "$LOLBO" \
  --n-tasks "$N_TASKS" --n-shards 3 --shard-index 0 --n-trials "$N_TRIALS" \
  --execution-timeout "$TIMEOUT" --ckpt "$CKPT" --out-root "$OUT" --skip-done --device cuda:0 \
  --dsn "host=127.0.0.1 port=5432 dbname=imdb user=imdb password=imdb" \
  >>"$LOG/query_bolt_lolbo_ceb3k_s0_pg5432.log" 2>&1 &
PID_L0=$!

CUDA_VISIBLE_DEVICES=3 nohup "$PY_LOLBO" -u "$LOLBO" \
  --n-tasks "$N_TASKS" --n-shards 3 --shard-index 1 --n-trials "$N_TRIALS" \
  --execution-timeout "$TIMEOUT" --ckpt "$CKPT" --out-root "$OUT" --skip-done --device cuda:0 \
  --dsn "host=127.0.0.1 port=5433 dbname=imdb user=imdb password=imdb" \
  >>"$LOG/query_bolt_lolbo_ceb3k_s1_pg5433.log" 2>&1 &
PID_L1=$!

CUDA_VISIBLE_DEVICES=1 nohup "$PY_LOLBO" -u "$LOLBO" \
  --n-tasks "$N_TASKS" --n-shards 3 --shard-index 2 --n-trials "$N_TRIALS" \
  --execution-timeout "$TIMEOUT" --ckpt "$CKPT" --out-root "$OUT" --skip-done --device cuda:0 \
  --dsn "host=127.0.0.1 port=5434 dbname=imdb user=imdb password=imdb" \
  >>"$LOG/query_bolt_lolbo_ceb3k_s2_pg5434.log" 2>&1 &
PID_L2=$!

# TPE shards (CPU decode; execution-bound)
nohup "$PY_TPE" -u "$TPE" \
  --n-tasks "$N_TASKS" --n-shards 3 --shard-index 0 --n-trials "$N_TRIALS" \
  --execution-timeout "$TIMEOUT" --ckpt "$CKPT" --out-root "$OUT" --skip-done --device cpu \
  --dsn "host=127.0.0.1 port=5432 dbname=imdb user=imdb password=imdb" \
  >>"$LOG/query_bolt_tpe_ceb3k_s0_pg5432.log" 2>&1 &
PID_T0=$!

nohup "$PY_TPE" -u "$TPE" \
  --n-tasks "$N_TASKS" --n-shards 3 --shard-index 1 --n-trials "$N_TRIALS" \
  --execution-timeout "$TIMEOUT" --ckpt "$CKPT" --out-root "$OUT" --skip-done --device cpu \
  --dsn "host=127.0.0.1 port=5433 dbname=imdb user=imdb password=imdb" \
  >>"$LOG/query_bolt_tpe_ceb3k_s1_pg5433.log" 2>&1 &
PID_T1=$!

nohup "$PY_TPE" -u "$TPE" \
  --n-tasks "$N_TASKS" --n-shards 3 --shard-index 2 --n-trials "$N_TRIALS" \
  --execution-timeout "$TIMEOUT" --ckpt "$CKPT" --out-root "$OUT" --skip-done --device cpu \
  --dsn "host=127.0.0.1 port=5434 dbname=imdb user=imdb password=imdb" \
  >>"$LOG/query_bolt_tpe_ceb3k_s2_pg5434.log" 2>&1 &
PID_T2=$!

echo "lolbo pids s0/s1/s2=$PID_L0 $PID_L1 $PID_L2" | tee -a "$MASTER"
echo "tpe   pids s0/s1/s2=$PID_T0 $PID_T1 $PID_T2" | tee -a "$MASTER"
echo "out=$OUT logs=$LOG/query_bolt_{lolbo,tpe}_ceb3k_s*.log" | tee -a "$MASTER"
