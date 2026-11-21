#!/bin/bash
# Run OpenBox evaluation for all models on restored ml_test_20.jsonl

set -e

EVAL_CLI="/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/scripts/eval/evaluate_config_cli.py"
TEST_DATA="/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/data/ml_test_20.jsonl"
EVAL_DIR="/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/eval_runs"
REPORTS_DIR="$EVAL_DIR/reports"

mkdir -p "$REPORTS_DIR"

# Models and their prediction files
# Format: "model_name|predictions_jsonl|output_json"
MODELS=(
    "claude_opus_4_8|$EVAL_DIR/claude_opus_4_8_test_predictions.jsonl|$EVAL_DIR/claude_opus_4_8_test_openbox_eval.json"
    "gemini_3.1_pro_preview|$EVAL_DIR/gemini_3.1_pro_preview_test_predictions.jsonl|$EVAL_DIR/gemini_3.1_pro_preview_test_openbox_eval.json"
    "gemini_31_flash|$EVAL_DIR/gemini_31_flash_test_predictions.jsonl|$EVAL_DIR/gemini_31_flash_test_openbox_eval.json"
    "gpt_5.5|$EVAL_DIR/gpt_5.5_test_predictions.jsonl|$EVAL_DIR/gpt_5.5_test_openbox_eval.json"
    "qwen_base|$EVAL_DIR/qwen_base_test_predictions.jsonl|$EVAL_DIR/qwen_base_test_openbox_eval.json"
    "qwen_base_full_sft_4|$EVAL_DIR/qwen_base_full_sft_4_test_predictions.jsonl|$EVAL_DIR/qwen_base_full_sft_4_test_openbox_eval.json"
    "qwen_base_lora_sft|$EVAL_DIR/qwen_base_lora_sft_test_predictions.jsonl|$EVAL_DIR/qwen_base_lora_sft_test_openbox_eval.json"
    "qwen_base_lora_sft_3|$EVAL_DIR/qwen_base_lora_sft_3_test_predictions.jsonl|$EVAL_DIR/qwen_base_lora_sft_3_test_openbox_eval.json"
    "qwen_base_lora_sft_2_dl|$EVAL_DIR/qwen_base_lora_sft_2_dl_test_predictions.jsonl|$EVAL_DIR/qwen_base_lora_sft_2_dl_test_openbox_eval.json"
    "qwen_base_lora_sft_3_dl|$EVAL_DIR/qwen_base_lora_sft_3_dl_test_predictions.jsonl|$EVAL_DIR/qwen_base_lora_sft_3_dl_test_openbox_eval.json"
    "qwen_base_lora_sft_4_dl|$EVAL_DIR/qwen_base_lora_sft_4_dl_test_predictions.jsonl|$EVAL_DIR/qwen_base_lora_sft_4_dl_test_openbox_eval.json"
    "qwen_base_lora_sft_5_dl|$EVAL_DIR/qwen_base_lora_sft_5_dl_test_predictions.jsonl|$EVAL_DIR/qwen_base_lora_sft_5_dl_test_openbox_eval.json"
    "qwen_base_lora_sft_6_dl|$EVAL_DIR/qwen_base_lora_sft_6_dl_test_predictions.jsonl|$EVAL_DIR/qwen_base_lora_sft_6_dl_test_openbox_eval.json"
    "qwen_base_lora_sft_dl|$EVAL_DIR/qwen_base_lora_sft_dl_test_predictions.jsonl|$EVAL_DIR/qwen_base_lora_sft_dl_test_openbox_eval.json"
    "qwen_full_sft_recommended|$EVAL_DIR/qwen_full_sft_recommended_test_predictions.jsonl|$EVAL_DIR/qwen_full_sft_recommended_test_openbox_eval.json"
    "qwen_lora_recommended5|$EVAL_DIR/qwen_lora_recommended5_test_predictions.jsonl|$EVAL_DIR/qwen_lora_recommended5_test_openbox_eval.json"
    "lora_recommended3|$EVAL_DIR/lora_recommended3_test_predictions.jsonl|$EVAL_DIR/lora_recommended3_test_openbox_eval.json"
)

# Run evaluation for each model
for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_name pred_file out_file <<< "$entry"
    
    if [ ! -f "$pred_file" ]; then
        echo "Skipping $model_name: prediction file not found: $pred_file"
        continue
    fi
    
    echo "========================================"
    echo "Evaluating: $model_name"
    echo "Predictions: $pred_file"
    echo "Output: $out_file"
    echo "========================================"
    
    python "$EVAL_CLI" \
        --dataset-jsonl "$TEST_DATA" \
        --predictions-jsonl "$pred_file" \
        --output-json "$out_file" \
        || echo "FAILED: $model_name"
    
    echo ""
done

echo "========================================"
echo "All evaluations complete"
echo "========================================"
