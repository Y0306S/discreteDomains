#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
export FORCE_TORCHRUN=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

source /root/miniconda3/etc/profile.d/conda.sh && conda activate llamafactory_qwen35
cd /root/LLMOptimizer/LlamaFactory

CONFIG_DIR="/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/configs/dl/multigpu"
LOG_DIR="/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune/dl"
mkdir -p "$LOG_DIR"

# Use all 8 GPUs for each job via DDP
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 1) LoRA SFT dl - 8 GPUs, grad_acc=2 to keep effective batch=16
CONFIG1="$CONFIG_DIR/llamafactory_qwen35_9b_base_lora_sft_dl.yaml"
LOG1="$LOG_DIR/lora_sft_dl.log"
echo "[$(date)] ===== Starting LoRA SFT dl on 8 GPUs (config 1) =====" | tee "$LOG1"
llamafactory-cli train "$CONFIG1" >> "$LOG1" 2>&1
echo "[$(date)] ===== Finished LoRA SFT dl with exit code $? =====" | tee -a "$LOG1"

# 2) LoRA SFT 2 dl - 8 GPUs, grad_acc=1 to keep effective batch=16
CONFIG2="$CONFIG_DIR/llamafactory_qwen35_9b_base_lora_sft_2_dl.yaml"
LOG2="$LOG_DIR/lora_sft_2_dl.log"
echo "[$(date)] ===== Starting LoRA SFT 2 dl on 8 GPUs (config 2) =====" | tee "$LOG2"
llamafactory-cli train "$CONFIG2" >> "$LOG2" 2>&1
echo "[$(date)] ===== Finished LoRA SFT 2 dl with exit code $? =====" | tee -a "$LOG2"

# 3) Full SFT 4 dl - DeepSpeed ZeRO-3 on all 8 GPUs (original config)
CONFIG3="/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/configs/dl/llamafactory_qwen35_9b_full_sft_4_dl.yaml"
LOG3="$LOG_DIR/full_sft_4_dl.log"
echo "[$(date)] ===== Starting Full SFT 4 dl on 8 GPUs (config 3) =====" | tee "$LOG3"
llamafactory-cli train "$CONFIG3" >> "$LOG3" 2>&1
echo "[$(date)] ===== Finished Full SFT 4 dl with exit code $? =====" | tee -a "$LOG3"

echo "[$(date)] ===== All dl SFT runs completed ====="
