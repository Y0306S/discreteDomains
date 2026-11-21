#!/usr/bin/env bash
# Env for the LOL-BO baseline running inside the benchmark harness.
#
# Cloned from openbox_eval -- the env the harness's classical optimizers (smac/tpe/
# cmaes/hebo) run in -- so the *oracle* (sklearn 1.3.2, lightgbm 4.6.0, xgboost 2.1.4,
# catboost 1.2.8) is byte-identical to the other baselines. The clone then gains
# gpytorch/botorch, which openbox_eval lacks. Cloning rather than mutating
# openbox_eval keeps the env that produced the method's published numbers untouched.
set -euo pipefail
export PATH=/root/data/cys/blockdata_archive/miniconda3/bin:$PATH
SRC=/root/data/cys/blockdata_archive/miniconda3/envs/openbox_eval
ENVDIR=/root/data/cys/baseline/envs/bl_lolbo_harness
rm -rf "$ENVDIR"
conda create -y -p "$ENVDIR" --clone "$SRC"
P="$ENVDIR/bin/python"

# Hold the oracle + numerics stack exactly where the clone put it, so adding
# gpytorch/botorch cannot silently upgrade torch/numpy/scipy/sklearn and change
# what the objective function computes.
C=/tmp/lolbo_harness_constraints.txt
cat > "$C" <<'EOF'
torch==2.0.1
numpy==1.24.4
scipy==1.15.3
scikit-learn==1.3.2
lightgbm==4.6.0
xgboost==2.1.4
catboost==1.2.8
optuna==4.9.0
ConfigSpace==0.6.1
EOF
# botorch 0.9.2 / gpytorch 1.11 are the newest pair supporting torch 2.0.1
$P -m pip install -c "$C" "gpytorch==1.11" "botorch==0.9.2"

echo "=== VERSION CHECK ==="
$P -c "
import torch, gpytorch, botorch, numpy, scipy, sklearn, lightgbm, xgboost, catboost, optuna
print('torch',torch.__version__,'gpytorch',gpytorch.__version__,'botorch',botorch.__version__)
print('numpy',numpy.__version__,'scipy',scipy.__version__,'sklearn',sklearn.__version__)
print('lightgbm',lightgbm.__version__,'xgboost',xgboost.__version__,'catboost',catboost.__version__)
"
