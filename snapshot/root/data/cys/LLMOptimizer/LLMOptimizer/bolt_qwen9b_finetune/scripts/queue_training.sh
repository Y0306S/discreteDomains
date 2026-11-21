#!/bin/bash
# Queues the SFT job to run automatically when all GPUs are free.
# Usage: nohup bash /jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/scripts/queue_training.sh &

LOG_FILE="/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/queued_training.log"
CONFIG="/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/configs/ml/llamafactory_qwen35_9b_full_sft_4.yaml"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Queue watcher started." >> "$LOG_FILE"

while true; do
    # Count GPUs with more than 100 MiB memory used
    BUSY_COUNT=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 100 {count++} END {print count+0}')

    if [ "$BUSY_COUNT" -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] All GPUs free. Launching training now..." >> "$LOG_FILE"
        source /jizhicfs/johnnyslin/anaconda3/bin/activate llamafactory_qwen35
        cd /jizhicfs/pkuhetu/bcxu/LlamaFactory
        llamafactory-cli train "$CONFIG" >> "$LOG_FILE" 2>&1
        EXIT_CODE=$?
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training finished with exit code $EXIT_CODE. Exiting watcher." >> "$LOG_FILE"
        break
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $BUSY_COUNT GPU(s) still busy. Waiting 30s..." >> "$LOG_FILE"
    fi

    sleep 30
done
