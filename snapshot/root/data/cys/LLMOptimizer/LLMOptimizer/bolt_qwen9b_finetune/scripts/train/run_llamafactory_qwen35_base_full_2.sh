#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export MPLCONFIGDIR=/tmp/matplotlib-llamafactory-qwen35-2
export no_proxy="${no_proxy:-}"
export http_proxy="${http_proxy:-}"
export https_proxy="${https_proxy:-}"

source /jizhicfs/lhy/activate
conda activate llamafactory_qwen35_repo

cd /jizhicfs/pkuhetu/bcxu/LlamaFactory
llamafactory-cli train \
  /jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/configs/ml/llamafactory_qwen35_9b_base_lora_sft_2.yaml \
  "$@"
