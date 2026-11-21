#!/usr/bin/env bash
set -euo pipefail
export PATH=/root/data/cys/blockdata_archive/miniconda3/bin:$PATH
SRC=/root/data/cys/baseline/envs/bl_lolbo_exact
ENVDIR=/root/data/cys/baseline/envs/bl_fsbo_exact
rm -rf "$ENVDIR"
# clone the LOL-BO stack: torch 1.11.0+cu113 / gpytorch 1.6.0 / numpy 1.22.3.
# FSBO's README asks gpytorch 1.4.2 / torch 1.8.1 / numpy 1.20.3, but PyPI torch 1.8.1 is a
# cu102 build with no sm_80 support, so it cannot run on this A800 host. numpy must stay
# <1.23 because fsbo_modules.py calls np.asscalar.
conda create -y -p "$ENVDIR" --clone "$SRC"
P="$ENVDIR/bin/python"
$P -m pip install xgboost tensorboard networkx matplotlib
echo "=== VERSION CHECK ==="
$P -c "import torch,gpytorch,numpy,xgboost,sklearn;print('torch',torch.__version__,'gpytorch',gpytorch.__version__,'numpy',numpy.__version__,'xgboost',xgboost.__version__,'sklearn',sklearn.__version__);print('cuda',torch.cuda.is_available())"
