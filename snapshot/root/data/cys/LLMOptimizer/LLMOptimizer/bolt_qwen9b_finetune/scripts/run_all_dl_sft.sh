#!/bin/bash
set -e
export PYTHONUNBUFFERED=1

source /root/miniconda3/etc/profile.d/conda.sh && conda activate llamafactory_qwen35
cd /root/LLMOptimizer/LlamaFactory

CONFIG_DIR="/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/configs/dl"
LOG_DIR="/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune/dl"
mkdir -p "$LOG_DIR"

# 1) LoRA SFT dl - small, run on single GPU
CONFIG1="$CONFIG_DIR/llamafactory_qwen35_9b_base_lora_sft_dl.yaml"
LOG1="$LOG_DIR/lora_sft_dl.log"
echo "[$(date)] ===== Starting LoRA SFT dl (config 1) =====" | tee "$LOG1"
export CUDA_VISIBLE_DEVICES=0
llamafactory-cli train "$CONFIG1" >> "$LOG1" 2>&1
echo "[$(date)] ===== Finished LoRA SFT dl with exit code $? =====" | tee -a "$LOG1"

# 2) LoRA SFT 2 dl - small, run on single GPU
CONFIG2="$CONFIG_DIR/llamafactory_qwen35_9b_base_lora_sft_2_dl.yaml"
LOG2="$LOG_DIR/lora_sft_2_dl.log"
echo "[$(date)] ===== Starting LoRA SFT 2 dl (config 2) =====" | tee "$LOG2"
export CUDA_VISIBLE_DEVICES=0
llamafactory-cli train "$CONFIG2" >> "$LOG2" 2>&1
echo "[$(date)] ===== Finished LoRA SFT 2 dl with exit code $? =====" | tee -a "$LOG2"

# 3) Full SFT 4 dl - needs DeepSpeed + multiple GPUs
CONFIG3="$CONFIG_DIR/llamafactory_qwen35_9b_full_sft_4_dl.yaml"
LOG3="$LOG_DIR/full_sft_4_dl.log"
echo "[$(date)] ===== Starting Full SFT 4 dl (config 3) =====" | tee "$LOG3"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FORCE_TORCHRUN=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
llamafactory-cli train "$CONFIG3" >> "$LOG3" 2>&1
echo "[$(date)] ===== Finished Full SFT 4 dl with exit code $? =====" | tee -a "$LOG3"

echo "[$(date)] ===== All dl SFT runs completed ====="
