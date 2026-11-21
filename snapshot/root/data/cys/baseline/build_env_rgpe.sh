#!/usr/bin/env bash
set -euo pipefail
export PATH=/root/data/cys/blockdata_archive/miniconda3/bin:$PATH
ENVDIR=/root/data/cys/baseline/envs/bl_rgpe
if [ ! -x "$ENVDIR/bin/python" ]; then
  # scripts/install.sh pins: py3.7 numpy1.18.1 scipy1.4.1 sklearn0.22.1 cython0.29.13 pandas0.25
  conda create -y -p "$ENVDIR" python=3.7 numpy=1.18.1 scipy=1.4.1 scikit-learn=0.22.1 \
      swig cython=0.29.13 pandas=0.25
fi
# env's own bin must be on PATH so pyrfr's setup.py finds swig
export PATH="$ENVDIR/bin:$PATH"
P="$ENVDIR/bin/python"
command -v swig
$P -m pip install --upgrade "pip<24"
$P -m pip install "ConfigSpace==0.4.11" "pyrfr==0.8.0"
$P -m pip install "smac[all]==0.12.3"
$P -m pip install "torch==1.5.0" "botorch==0.2.5" lockfile
echo "=== VERSION CHECK ==="
$P -c "import smac,ConfigSpace,botorch,torch,numpy,scipy,sklearn,pyrfr;print('smac',smac.__version__,'CS',ConfigSpace.__version__,'botorch',botorch.__version__,'torch',torch.__version__,'np',numpy.__version__,'scipy',scipy.__version__,'sk',sklearn.__version__)"
