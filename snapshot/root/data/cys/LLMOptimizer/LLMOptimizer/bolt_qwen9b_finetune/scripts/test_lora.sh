#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,6,7

source /jizhicfs/johnnyslin/anaconda3/bin/activate llamafactory_qwen35
cd /jizhicfs/pkuhetu/bcxu/LlamaFactory

LOG="/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/lora_test.log"
CONFIG="/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/configs/ml/llamafactory_qwen35_9b_full_sft_4.yaml"

echo "[$(date)] Starting LoRA test on GPUs $CUDA_VISIBLE_DEVICES" > "$LOG"
llamafactory-cli train "$CONFIG" >> "$LOG" 2>&1
echo "[$(date)] Training exited with code $?" >> "$LOG"
