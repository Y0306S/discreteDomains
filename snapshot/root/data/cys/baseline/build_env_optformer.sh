#!/usr/bin/env bash
set -euo pipefail
export PATH=/root/data/cys/blockdata_archive/miniconda3/bin:$PATH
ENVDIR=/root/data/cys/baseline/envs/bl_optformer
R=/root/data/cys/baseline/repos
C=/root/data/cys/baseline/optformer_constraints.txt
rm -rf "$ENVDIR"
conda create -y -p "$ENVDIR" python=3.10
P="$ENVDIR/bin/python"
$P -m pip install --upgrade pip
PIP="$P -m pip install -c $C"

# Era-matched stack for OptFormer @ neurips22 (Jul 2023). t5x main and modern flax both
# require py>=3.12 and track git HEAD, so we pin releases and install t5x from a
# Jul-2023 worktree with --no-deps.
$PIP "protobuf==3.20.3" "numpy==1.24.3"
$PIP "jax==0.4.23" "jaxlib==0.4.23"
$PIP "tensorflow==2.13.1" "tensorflow-text==2.13.0"
$PIP "flax==0.7.5" "optax==0.1.7" "chex==0.1.85" "orbax-checkpoint==0.4.4" "clu==0.0.12" \
     "tensorstore==0.1.45" "fiddle==0.2.5" gin-config "ml-collections==0.1.1" \
     "nest-asyncio>=1.5.6" "absl-py>=1.0.0" cached_property jaxtyping matplotlib pytest \
     sentencepiece
# deps that optformer/data/tasks.py pulls in via t5.data -- installed BEFORE t5 itself
$PIP babel nltk sacrebleu rouge-score "scikit-learn==1.3.2" "transformers==4.33.2" \
     editdistance mesh-tensorflow
$PIP "google-vizier==0.0.20"
$PIP "seqio==0.0.13"
$P -m pip install -e "$R/t5x-2023-07" --no-deps
$P -m pip install -e "$R/optformer-neurips22" --no-deps
# t5 0.9.4 hard-depends on seqio-nightly, which would evict seqio==0.0.13 and drag in
# TF 2.21. Install it --no-deps and LAST, so no later resolve "repairs" its metadata.
$P -m pip install "t5==0.9.4" --no-deps
echo "=== VERSION CHECK ==="
$P -c "
import jax, jaxlib, flax, seqio, t5x, tensorflow as tf, vizier
print('jax',jax.__version__,'jaxlib',jaxlib.__version__,'flax',flax.__version__)
print('tf',tf.__version__,'seqio',seqio.__version__)
print('cpu devices',jax.devices())
import t5.data; print('t5.data OK')
import optformer; print('optformer OK')
"
