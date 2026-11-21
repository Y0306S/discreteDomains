#!/usr/bin/env bash
set -euo pipefail
export PATH=/root/data/cys/blockdata_archive/miniconda3/bin:$PATH
ENVDIR=/root/data/cys/baseline/envs/bl_bora
rm -rf "$ENVDIR"
# Readme asks for python 3.11.8
conda create -y -p "$ENVDIR" python=3.11
P="$ENVDIR/bin/python"
$P -m pip install --upgrade pip
# requirements.txt pins
$P -m pip install "openai~=1.57.0" "python_dotenv~=1.0.1" "pandas~=2.2.3" \
    "scikit-learn~=1.5.1" "numpy~=1.26.4" "xgboost~=2.1.1" "matplotlib==3.9.2" \
    colorama tiktoken
# torch CPU wheel is enough: bora/ only imports torch for tensor plumbing, the
# real-world benchmarks that need CUDA are not part of our four scenarios
$P -m pip install "torch~=2.4.1" --index-url https://mirrors.aliyun.com/pytorch-wheels/cpu/ \
   || $P -m pip install "torch~=2.4.1"
# real-world benchmark extras (crop/solar tasks from the paper); optional for us
$P -m pip install "PCSE~=6.0.8" "pvlib~=0.11.1" || echo "WARN: PCSE/pvlib skipped (paper's real-world tasks only)"
echo "=== VERSION CHECK ==="
$P -c "import openai, numpy, sklearn, torch, pandas; print('openai',openai.__version__,'numpy',numpy.__version__,'sklearn',sklearn.__version__,'torch',torch.__version__)"
