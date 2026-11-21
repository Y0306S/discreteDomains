#!/usr/bin/env bash
set -euo pipefail
CAMPAIGN=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DISABLE_VERSION_CHECK=1 WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=""
export HF_HOME="$CAMPAIGN/hf_cache" HF_DATASETS_CACHE="$CAMPAIGN/hf_cache/datasets" TMPDIR="$CAMPAIGN/scratch"
exec /root/data/cys/baseline/envs/taskset_qwen35_fast_20261030/bin/python -u "$CAMPAIGN/code/ml_orpo_train.py" prepare --campaign "$CAMPAIGN"
