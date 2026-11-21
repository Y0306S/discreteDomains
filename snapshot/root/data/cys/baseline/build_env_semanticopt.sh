#!/usr/bin/env bash
set -euo pipefail
export PATH=/root/data/cys/blockdata_archive/miniconda3/bin:$PATH
ENVDIR=/root/data/cys/baseline/envs/bl_semanticopt
rm -rf "$ENVDIR"
conda create -y -p "$ENVDIR" python=3.11
P="$ENVDIR/bin/python"
$P -m pip install --upgrade pip
# SemanticOpt paper trains with Unsloth assistant-only LoRA SFT (rank/alpha 16, lr 1e-5,
# cosine, bf16, bs 64, 1 epoch, ctx 24576).
$P -m pip install "torch==2.6.0"
$P -m pip install unsloth unsloth_zoo
$P -m pip install "trl" "peft" "transformers" "datasets" "accelerate" "bitsandbytes" optuna openai
echo "=== VERSION CHECK ==="
$P -c "
import torch; print('torch',torch.__version__,'cuda',torch.cuda.is_available())
import transformers, trl, peft; print('transformers',transformers.__version__,'trl',trl.__version__,'peft',peft.__version__)
import unsloth; print('unsloth OK')
"
