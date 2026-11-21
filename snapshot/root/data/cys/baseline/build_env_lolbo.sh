#!/usr/bin/env bash
set -euo pipefail
export PATH=/root/data/cys/blockdata_archive/miniconda3/bin:$PATH
ENVDIR=/root/data/cys/baseline/envs/bl_lolbo_exact
rm -rf "$ENVDIR"
conda create -y -p "$ENVDIR" python=3.9
P="$ENVDIR/bin/python"
$P -m pip install --upgrade pip
# exact LOL-BO stack (lolbo_env.yml): torch 1.11.0 is the cu113 build on PyPI
$P -m pip install "numpy==1.22.3" "torch==1.11.0" "gpytorch==1.6.0" "botorch==0.6.3.1" "selfies==2.0.0" \
                  "scipy<1.11" pandas fire tqdm
echo "=== VERSION CHECK ==="
$P -c "import torch,gpytorch,botorch,numpy;print('torch',torch.__version__,'cuda',torch.version.cuda,'gpytorch',gpytorch.__version__,'botorch',botorch.__version__,'numpy',numpy.__version__)"
