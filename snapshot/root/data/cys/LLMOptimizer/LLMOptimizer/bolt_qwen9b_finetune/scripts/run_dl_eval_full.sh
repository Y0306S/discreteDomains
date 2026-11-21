#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=3
source /jizhicfs/johnnyslin/anaconda3/bin/activate llamafactory_qwen35
python /jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/scripts/eval/evaluate_dl_structural.py \
    --test-data /jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/data/dl_test.jsonl \
    --base-model /jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/outputs/llamafactory_qwen35_9b_base_full_sft_4_dl \
    --model-label qwen_full_sft_4_dl \
    --output-jsonl /jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/eval_runs/dl_qwen_full_sft_4_dl_predictions.jsonl \
    --output-json /jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/eval_runs/dl_qwen_full_sft_4_dl_structural_eval.json \
    --max-new-tokens 768
