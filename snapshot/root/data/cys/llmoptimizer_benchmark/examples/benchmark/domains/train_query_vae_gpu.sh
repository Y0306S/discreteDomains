#!/usr/bin/env bash
# Query CEB VAE on a chosen GPU (default physical GPU 3).
set -euo pipefail

GPU="${1:-3}"
export CUDA_VISIBLE_DEVICES="$GPU"
PY="${PY:-/root/miniconda3/envs/hpo_benchmark/bin/python}"
CEB_CKPT="/root/data/cys/BOLT/optimization/query_plans/query_plan_optimization/vae/CEB_64.ckpt"
LOG="/root/data/logs/query_vae_gpu${GPU}.log"

exec > >(tee -a "$LOG") 2>&1
echo "[query-vae] $(date -Is) start GPU=$GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

export PYTHONPATH="/root/data/cys/BOLT/optimization/query_plans/query_plan_optimization:${PYTHONPATH:-}"
cd /root/data/cys/BOLT/optimization/query_plans/query_plan_optimization

"$PY" /root/data/cys/BOLT/scripts/train_ceb_vae.py \
  --d-neck 32 \
  --max-epochs 50 \
  --batch-size 512 \
  --num-workers 8 \
  --gpus 1 \
  --gpu-mem-fraction 1.0 \
  --out "$CEB_CKPT"

echo "[query-vae] $(date -Is) done -> $CEB_CKPT"
