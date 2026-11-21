#!/bin/bash
# Watcher: polls GPU availability and launches training when all 8 GPUs are free

LOG_FILE="/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/training_run.log"
CONFIG="/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/configs/ml/llamafactory_qwen35_9b_full_sft_4.yaml"

source /jizhicfs/johnnyslin/anaconda3/bin/activate llamafactory_qwen35

echo "[$(date)] Watcher started. Polling GPU availability every 30s..." >> "$LOG_FILE"

while true; do
    # Count GPUs with >100 MiB memory usage (i.e., actively in use)
    BUSY_GPUS=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 100 {count++} END {print count+0}')
    
    if [ "$BUSY_GPUS" -eq 0 ]; then
        echo "[$(date)] All GPUs are free. Launching training..." >> "$LOG_FILE"
        cd /jizhicfs/pkuhetu/bcxu/LlamaFactory
        llamafactory-cli train "$CONFIG" >> "$LOG_FILE" 2>&1
        echo "[$(date)] Training process exited with code $?. Watcher exiting." >> "$LOG_FILE"
        break
    else
        echo "[$(date)] $BUSY_GPUS GPU(s) still busy. Waiting..." >> "$LOG_FILE"
    fi
    
    sleep 30
done
