# LLaMA-Factory Repo-Aligned Setup

This setup follows the local LLaMA-Factory documentation instead of the earlier ad hoc training scripts.

## What The Repo Says

- Run commands from `/jizhicfs/pkuhetu/bcxu/LlamaFactory`.
- Install from source with `pip install -e .`.
- Register custom datasets in `dataset_info.json` under the configured `dataset_dir`.
- Launch training with `llamafactory-cli train <yaml>`.
- Select GPUs with `CUDA_VISIBLE_DEVICES`.
- The CUDA Docker baseline is Python 3.11, PyTorch 2.6.0, CUDA 12.4, and Flash-attn 2.7.4.

## Clean Environment

Do not reuse the previous `llamafactory_qwen35` environment. It has mixed Torch, Triton, and FLA versions from debugging.

Create a fresh repo-aligned environment:

```bash
source /jizhicfs/lhy/activate
conda create -y -n llamafactory_qwen35_repo python=3.11
conda activate llamafactory_qwen35_repo

export PYTHONNOUSERSITE=1
python -m pip install --upgrade pip packaging wheel setuptools editables "hatchling>=1.18.0"
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

cd /jizhicfs/pkuhetu/bcxu/LlamaFactory
python -m pip install --no-build-isolation -e .
python -m pip install --no-build-isolation -r requirements/metrics.txt

# Required by Transformers' Qwen3.5 implementation.
# LLaMA-Factory requires >=0.4.1; 0.4.1 failed on H20/Triton backward here, 0.4.2 passed smoke training.
python -m pip install --no-deps flash-linear-attention==0.4.2 fla-core==0.4.2
```

Optional, matching the Docker image more closely:

```bash
python -m pip install --no-build-isolation \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

## Verify Environment

```bash
source /jizhicfs/lhy/activate
conda activate llamafactory_qwen35_repo
export PYTHONNOUSERSITE=1

python - <<'PY'
import torch
import triton
import transformers
from fla.modules.convolution import causal_conv1d
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("triton", triton.__version__)
print("transformers", transformers.__version__)
print("fla imports ok")
PY

python -m pip check
```

## Smoke Test

The smoke test uses one visible GPU by default, GPU 4:

```bash
cd /jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune
CUDA_VISIBLE_DEVICES=4 ./scripts/train/run_llamafactory_qwen35_base_smoke.sh
```

To use GPUs 4-7:

```bash
cd /jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune
CUDA_VISIBLE_DEVICES=4,5,6,7 ./scripts/train/run_llamafactory_qwen35_base_smoke.sh
```

## Full Run

```bash
cd /jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune
CUDA_VISIBLE_DEVICES=4,5,6,7 ./scripts/train/run_llamafactory_qwen35_base_full.sh
```

## Monitor

```bash
nvidia-smi
tail -f /jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/outputs/llamafactory_qwen35_9b_base_lora_sft/trainer_log.jsonl
```

If the full run is launched with `nohup`, monitor its log instead:

```bash
tail -f logs/train_llamafactory_qwen35_full.log
```
