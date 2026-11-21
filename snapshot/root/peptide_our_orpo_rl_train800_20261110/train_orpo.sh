#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to one admitted training GPU}"
CAMPAIGN=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DISABLE_VERSION_CHECK=1 WANDB_MODE=disabled
export HF_HOME="$CAMPAIGN/hf_cache" HF_DATASETS_CACHE="$CAMPAIGN/hf_cache/datasets" TMPDIR="$CAMPAIGN/scratch"
exec /root/data/cys/baseline/envs/taskset_qwen35_fast_20261030/bin/python -u "$CAMPAIGN/code/train_checked.py" --campaign "$CAMPAIGN"
