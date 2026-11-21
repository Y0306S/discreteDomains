#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FORCE_TORCHRUN=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

source /jizhicfs/johnnyslin/anaconda3/bin/activate llamafactory_qwen35
cd /jizhicfs/pkuhetu/bcxu/LlamaFactory

LOG="/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/full_sft_8gpu.log"
CONFIG="/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/configs/ml/llamafactory_qwen35_9b_full_sft_4.yaml"

echo "[$(date)] Starting FULL SFT on ALL 8 GPUs" > "$LOG"
echo "[$(date)] Config: $CONFIG" >> "$LOG"
echo "[$(date)] Environment: llamafactory_qwen35" >> "$LOG"
echo "[$(date)] Batch size per device: 8" >> "$LOG"
echo "[$(date)] Effective batch size: $((8 * 8)) = 64" >> "$LOG"
echo "[$(date)] Workers: preprocessing=16, dataloader=4" >> "$LOG"

llamafactory-cli train "$CONFIG" >> "$LOG" 2>&1
echo "[$(date)] Finished with exit code $?" >> "$LOG"
