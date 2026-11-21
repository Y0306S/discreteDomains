#!/usr/bin/env bash
# Sequential ProteinGym stack on GPU 1 only: VAE → surrogates → BO probe.
# Skips stages whose artifacts already exist.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/root/data/cys/blockdata_archive/llmoptimizer_benchmark
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENDPOINT="${HF_HUB_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/data/cys/blockdata_archive/home-spill/dot-cache/huggingface}"
export PROTEINGYM_ESM_LOCAL="${PROTEINGYM_ESM_LOCAL:-/root/data/cys/blockdata_archive/home-spill/dot-cache/huggingface/hub/models--facebook--esm2_t12_35M_UR50D/snapshots/6fbf070e65b0b7291e7bbcd451118c216cff79d8}"
PY=/root/miniconda3/envs/hpo_benchmark/bin/python
REPO=/root/data/cys/blockdata_archive/llmoptimizer_benchmark
LOG=/root/data/logs
VAE_DIR="$REPO/history_data/domain_protein/proteingym/vae"
mkdir -p "$LOG" "$VAE_DIR"

echo "[pgym-gpu1] $(date -Is) start CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

if [[ -f "$VAE_DIR/gfp_vae.pt" ]]; then
  echo "[pgym-gpu1] $(date -Is) skip GFP VAE (exists)"
else
  echo "[pgym-gpu1] $(date -Is) train GFP VAE"
  $PY -u "$REPO/examples/benchmark/domains/train_proteingym_vae.py" \
    --landscape gfp --device cuda:0 --epochs 25 \
    2>&1 | tee "$LOG/proteingym_vae_gfp_gpu1.log"
fi

if [[ -f "$VAE_DIR/aav_vae.pt" ]]; then
  echo "[pgym-gpu1] $(date -Is) skip AAV VAE (exists)"
else
  echo "[pgym-gpu1] $(date -Is) train AAV VAE"
  $PY -u "$REPO/examples/benchmark/domains/train_proteingym_vae.py" \
    --landscape aav --device cuda:0 --epochs 20 --batch-size 16 \
    2>&1 | tee "$LOG/proteingym_vae_aav_gpu1.log"
fi

if [[ -f "$REPO/history_data/domain_protein/proteingym/gfp/esm_fitness_head.pt" ]]; then
  echo "[pgym-gpu1] $(date -Is) skip GFP surrogate (exists)"
else
  echo "[pgym-gpu1] $(date -Is) train GFP surrogate"
  $PY -u "$REPO/examples/benchmark/domains/train_proteingym_surrogate.py" \
    --landscape gfp --device cuda:0 \
    2>&1 | tee "$LOG/proteingym_surr_gfp_gpu1.log"
fi

if [[ -f "$REPO/history_data/domain_protein/proteingym/aav/esm_fitness_head.pt" ]]; then
  echo "[pgym-gpu1] $(date -Is) skip AAV surrogate (exists)"
else
  echo "[pgym-gpu1] $(date -Is) train AAV surrogate"
  $PY -u "$REPO/examples/benchmark/domains/train_proteingym_surrogate.py" \
    --landscape aav --device cuda:0 \
    2>&1 | tee "$LOG/proteingym_surr_aav_gpu1.log"
fi

echo "[pgym-gpu1] $(date -Is) BO probe GFP+AAV"
$PY -u "$REPO/examples/benchmark/domains/run_proteingym_bo_suite.py" \
  --landscape both --device cuda:0 --n-starts 4 --n-trials 40 \
  --optimizer tpe --baselines --seed 1 \
  2>&1 | tee "$LOG/proteingym_bo_gpu1.log"

echo "[pgym-gpu1] $(date -Is) ALL DONE"
