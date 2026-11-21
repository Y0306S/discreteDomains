#!/bin/bash
# Auto-evaluation monitor script
# Runs predictions and OpenBox eval when each training finishes

LOG_FILE="/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/eval_runs/auto_monitor.log"
CONDA_BASE="/jizhicfs/johnnyslin/anaconda3"
PROJECT_DIR="/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune"
OUTPUTS_DIR="$PROJECT_DIR/outputs"
EVAL_DIR="$PROJECT_DIR/eval_runs"
DATA_FILE="$PROJECT_DIR/data/ml_test_20.jsonl"
BASE_MODEL="/jizhicfs/pkuhetu/bcxu/models/Qwen3.5-9B-Base"

source "$CONDA_BASE/etc/profile.d/conda.sh"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Models to monitor: config_name -> output_dir -> model_label
# Format: "config_path|output_dir|model_label|type(lora/full)"
MODELS=(
    "configs/dl/llamafactory_qwen35_9b_base_lora_sft_dl.yaml|llamafactory_qwen35_9b_base_lora_sft_dl|dl_lora_sft|lora"
    "configs/dl/llamafactory_qwen35_9b_base_lora_sft_2_dl.yaml|llamafactory_qwen35_9b_base_lora_sft_2_dl|dl_lora_sft_2|lora"
    "configs/dl/llamafactory_qwen35_9b_base_lora_sft_3_dl.yaml|llamafactory_qwen35_9b_base_lora_sft_3_dl|dl_lora_sft_3|lora"
    "configs/dl/llamafactory_qwen35_9b_base_lora_sft_4_dl.yaml|llamafactory_qwen35_9b_base_lora_sft_4_dl|dl_lora_sft_4|lora"
    "configs/dl/llamafactory_qwen35_9b_base_lora_sft_5_dl.yaml|llamafactory_qwen35_9b_base_lora_sft_5_dl|dl_lora_sft_5|lora"
    "configs/dl/llamafactory_qwen35_9b_base_lora_sft_6_dl.yaml|llamafactory_qwen35_9b_base_lora_sft_6_dl|dl_lora_sft_6|lora"
    "configs/dl/llamafactory_qwen35_9b_full_sft_4_dl.yaml|llamafactory_qwen35_9b_full_sft_4_dl|dl_full_sft_4|full"
    "configs/dl/llamafactory_qwen35_9b_full_sft_4_dl_rec.yaml|llamafactory_qwen35_9b_full_sft_4_dl_rec|dl_full_sft_4_rec|full"
    "configs/ml/llamafactory_qwen35_9b_lora_recommended5.yaml|llamafactory_qwen35_9b_lora_recommended5|ml_lora_rec5|lora"
)

# Track which models have been processed
PROCESSED_FILE="$EVAL_DIR/.processed_models"
touch "$PROCESSED_FILE"

is_processed() {
    grep -q "^$1$" "$PROCESSED_FILE" 2>/dev/null
}

mark_processed() {
    echo "$1" >> "$PROCESSED_FILE"
}

# Check if training is done (has final model files)
is_training_done() {
    local output_dir="$1"
    local model_type="$2"
    
    if [ "$model_type" = "lora" ]; then
        # LoRA: check for adapter_model.safetensors
        if [ -f "$OUTPUTS_DIR/$output_dir/adapter_model.safetensors" ]; then
            # Also check if file is still being written (size stable)
            local size1=$(stat -c%s "$OUTPUTS_DIR/$output_dir/adapter_model.safetensors" 2>/dev/null)
            sleep 5
            local size2=$(stat -c%s "$OUTPUTS_DIR/$output_dir/adapter_model.safetensors" 2>/dev/null)
            if [ "$size1" = "$size2" ]; then
                return 0
            fi
        fi
    else
        # Full SFT: check for model.safetensors
        if [ -f "$OUTPUTS_DIR/$output_dir/model.safetensors" ]; then
            local size1=$(stat -c%s "$OUTPUTS_DIR/$output_dir/model.safetensors" 2>/dev/null)
            sleep 5
            local size2=$(stat -c%s "$OUTPUTS_DIR/$output_dir/model.safetensors" 2>/dev/null)
            if [ "$size1" = "$size2" ]; then
                return 0
            fi
        fi
    fi
    return 1
}

# Run predictions for a model
run_predictions() {
    local config_path="$1"
    local output_dir="$2"
    local model_label="$3"
    local model_type="$4"
    
    local pred_file="$EVAL_DIR/${model_label}_test_predictions.jsonl"
    local log_file="$EVAL_DIR/gen_${model_label}.log"
    
    log "Starting predictions for $model_label"
    
    conda activate llamafactory_qwen35
    export PYTHONNOUSERSITE=1
    
    if [ "$model_type" = "lora" ]; then
        # LoRA: use base model + adapter
        CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/eval/evaluate_config_cli.py" \
            --generate qwen \
            --dataset-jsonl "$DATA_FILE" \
            --predictions-out "$pred_file" \
            --base-model "$BASE_MODEL" \
            --adapter-dir "$OUTPUTS_DIR/$output_dir" \
            --model-label "$model_label" \
            --max-new-tokens 256 \
            --num-samples 3 \
            --temperature 0.1 \
            --top-p 0.9 \
            > "$log_file" 2>&1
    else
        # Full SFT: use output dir as base model
        CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/eval/evaluate_config_cli.py" \
            --generate qwen \
            --dataset-jsonl "$DATA_FILE" \
            --predictions-out "$pred_file" \
            --base-model "$OUTPUTS_DIR/$output_dir" \
            --model-label "$model_label" \
            --max-new-tokens 256 \
            --num-samples 3 \
            --temperature 0.1 \
            --top-p 0.9 \
            > "$log_file" 2>&1
    fi
    
    log "Predictions done for $model_label"
}

# Run OpenBox eval for a model
run_eval() {
    local model_label="$1"
    
    local pred_file="$EVAL_DIR/${model_label}_test_predictions.jsonl"
    local eval_file="$EVAL_DIR/${model_label}_test_openbox_eval.json"
    local log_file="$EVAL_DIR/eval_${model_label}.log"
    
    if [ ! -f "$pred_file" ]; then
        log "ERROR: Predictions not found for $model_label"
        return 1
    fi
    
    log "Starting OpenBox eval for $model_label"
    
    conda activate openbox_eval
    
    python "$PROJECT_DIR/scripts/eval/evaluate_config_cli.py" \
        --dataset-jsonl "$DATA_FILE" \
        --predictions-jsonl "$pred_file" \
        --output-json "$eval_file" \
        > "$log_file" 2>&1
    
    log "OpenBox eval done for $model_label"
}

# Main monitoring loop
log "Auto-eval monitor started"
log "Monitoring ${#MODELS[@]} models"

while true; do
    for model_info in "${MODELS[@]}"; do
        IFS='|' read -r config_path output_dir model_label model_type <<< "$model_info"
        
        if is_processed "$model_label"; then
            continue
        fi
        
        if is_training_done "$output_dir" "$model_type"; then
            log "Training done for $model_label"
            
            # Run predictions
            run_predictions "$config_path" "$output_dir" "$model_label" "$model_type"
            
            # Run eval
            run_eval "$model_label"
            
            mark_processed "$model_label"
            log "Completed $model_label"
        fi
    done
    
    # Check if all done
    all_done=true
    for model_info in "${MODELS[@]}"; do
        IFS='|' read -r config_path output_dir model_label model_type <<< "$model_info"
        if ! is_processed "$model_label"; then
            all_done=false
            break
        fi
    done
    
    if $all_done; then
        log "All models processed!"
        break
    fi
    
    # Sleep before next check
    sleep 300  # Check every 5 minutes
done

log "Monitor exiting"
