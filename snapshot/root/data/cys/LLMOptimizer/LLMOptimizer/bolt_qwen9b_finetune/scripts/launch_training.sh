#!/bin/bash
source /jizhicfs/johnnyslin/anaconda3/bin/activate llamafactory_qwen35
cd /jizhicfs/pkuhetu/bcxu/LlamaFactory
export FORCE_TORCHRUN=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
llamafactory-cli train /jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/configs/ml/llamafactory_qwen35_9b_full_sft_4.yaml 2>&1 | tee /jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/training_live.log
