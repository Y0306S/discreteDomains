#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=2
source /jizhicfs/johnnyslin/anaconda3/bin/activate llamafactory_qwen35
python /jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/scripts/eval/evaluate_dl_structural.py \
    --test-data /jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/data/dl_test.jsonl \
    --base-model /jizhicfs/pkuhetu/bcxu/models/Qwen3.5-9B-Base \
    --adapter-dir /jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/outputs/llamafactory_qwen35_9b_base_lora_sft_2_dl \
    --model-label qwen_lora_sft_2_dl \
    --output-jsonl /jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/eval_runs/dl_qwen_lora_sft_2_dl_predictions.jsonl \
    --output-json /jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/eval_runs/dl_qwen_lora_sft_2_dl_structural_eval.json \
    --max-new-tokens 768
