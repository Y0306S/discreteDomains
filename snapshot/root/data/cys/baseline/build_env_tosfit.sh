#!/usr/bin/env bash
set -euo pipefail
export PATH=/root/data/cys/blockdata_archive/miniconda3/bin:$PATH
ENVDIR=/root/data/cys/baseline/envs/bl_tosfit
rm -rf "$ENVDIR"
conda create -y -p "$ENVDIR" python=3.11
P="$ENVDIR/bin/python"
$P -m pip install --upgrade pip
# environment.yaml pinned stack
$P -m pip install "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0"
$P -m pip install "transformers==4.51.3" "sentence_transformers==4.1.0" "datasets==3.3.2" \
    "accelerate==1.6.0" "wandb==0.19.6" "bert-score==0.3.13" "biopython==1.85" \
    "scipy==1.15.3" "RestrictedPython==8.0" "lexicalrichness==0.5.1" "ipykernel==6.29.5"
# quantum-circuit benchmark deps (needed for full repo import surface)
$P -m pip install "qiskit==2.0.2" "qiskit-aer==0.17.1" "openfermion==1.7.1" "openfermionpyscf==0.5" || \
  echo "WARN: quantum extras failed (only affects the Hamiltonian benchmark)"
$P -m pip install "deepspeed==0.16" || echo "WARN: deepspeed failed (optional)"
echo "=== VERSION CHECK ==="
$P -c "import torch,transformers;print('torch',torch.__version__,'cuda_avail',torch.cuda.is_available(),'transformers',transformers.__version__)"
