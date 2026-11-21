#!/bin/bash
set -e

cd /jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune
source /jizhicfs/johnnyslin/anaconda3/bin/activate llamafactory_qwen35

# Critical fix: remove ghost flash-linear-attention dir if present
rm -rf /jizhicfs/johnnyslin/anaconda3/envs/llamafactory_qwen35/lib/python3.11/site-packages/fla

# Wait for any lingering distributed ports to clear
sleep 5

echo "===== Job 1/3: LoRA SFT DL (bsz=1, grad_acc=16, 3 epochs) ====="
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.run \
  --nproc_per_node=4 \
  --master_port=29510 \
  -m llamafactory.cli train \
  --config configs/dl/llamafactory_qwen35_9b_base_lora_sft_dl.yaml

echo "===== Job 2/3: LoRA SFT 2 DL (bsz=2, grad_acc=8, 5 epochs) ====="
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.run \
  --nproc_per_node=4 \
  --master_port=29510 \
  -m llamafactory.cli train \
  --config configs/dl/llamafactory_qwen35_9b_base_lora_sft_2_dl.yaml

echo "===== Job 3/3: Full SFT 4 DL (bsz=8, DeepSpeed Z3, 5 epochs) ====="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
llamafactory-cli train \
  --config configs/dl/llamafactory_qwen35_9b_full_sft_4_dl.yaml

echo "===== All jobs completed ====="
