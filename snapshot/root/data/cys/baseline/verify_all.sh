#!/usr/bin/env bash
# Verify every baseline env + repo is importable and functional.
# Usage: bash /root/data/cys/baseline/verify_all.sh
B=/root/data/cys/baseline
R=$B/repos
E=$B/envs
pass=0; fail=0
hdr() { printf '\n\033[1m═══ %s ═══\033[0m\n' "$1"; }
chk() { if [ "$1" = 0 ]; then echo "  ✅ $2"; pass=$((pass+1)); else echo "  ❌ $2"; fail=$((fail+1)); fi; }

hdr "1. Random Search  (in-harness, openbox advisor_type=random)"
/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python -c "
from mindware.components.optimizers.random_search_optimizer import RandomSearchOptimizer
from openbox.optimizer.generic_smbo import SMBO
print('   RandomSearchOptimizer + openbox SMBO OK')
" 2>/dev/null; chk $? "Random Search importable"

hdr "2. LOL-BO  (nataliemaus/lolbo)"
(cd $R/lolbo && PYTHONPATH=. $E/bl_lolbo_exact/bin/python $B/tests/smoke_lolbo.py) 2>&1 | grep -E "SMOKE|FAIL"
chk ${PIPESTATUS[0]} "LOL-BO trust-region DKL-GP loop"

hdr "3. TPE  (optuna TPESampler, in-harness)"
/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python -c "
import optuna
from mindware.components.optimizers.pt4opt_tpe_optimizer import TPEOptimizer
s = optuna.samplers.TPESampler(seed=1)
st = optuna.create_study(sampler=s, direction='minimize')
st.optimize(lambda t: (t.suggest_float('x',-5,5)-0.3)**2, n_trials=25)
print('   TPE 25 trials, best x=%.3f' % st.best_params['x'])
" 2>/dev/null; chk $? "TPE / Latent-TPE backend"

hdr "4. FSBO  (releaunifreiburg/FSBO)"
(cd $R/FSBO && PYTHONPATH=. $E/bl_fsbo_exact/bin/python $B/tests/smoke_fsbo.py) 2>&1 | grep -E "SMOKE|FAIL"
chk ${PIPESTATUS[0]} "FSBO meta-train + meta-test"

hdr "5. RGPE  (automl/transfer-hpo-framework)"
(cd $R/transfer-hpo-framework && PYTHONPATH=. $E/bl_rgpe/bin/python $B/tests/smoke_rgpe.py) 2>&1 | grep -E "SMOKE|FAIL"
chk ${PIPESTATUS[0]} "RGPE(TAF) rank weighting"

hdr "6. OptFormer  (google-research/optformer @ neurips22)"
$E/bl_optformer/bin/python -c "
import jax, t5x, optformer, seqio, tensorflow as tf
print('   jax',jax.__version__,'tf',tf.__version__,'seqio',seqio.__version__)
print('   devices',jax.devices())
" 2>/dev/null; chk $? "OptFormer + T5X stack"

hdr "7. ToSFiT  (IBM/thompson-sampling-via-fine-tuning-of-llms)"
(cd $R/tosfit && PYTHONPATH=. $E/bl_tosfit/bin/python -c "
import importlib, pkgutil, src
bad=[]
for m in pkgutil.walk_packages(src.__path__,'src.'):
    try: importlib.import_module(m.name)
    except Exception as e: bad.append(m.name)
assert not bad, bad
from src.benchmarks.biology.reward import protein_stability
from Bio.SeqUtils.ProtParam import ProteinAnalysis
print('   all src modules import; instability_index=%.2f' % ProteinAnalysis('MINDLLDISRIISGK').instability_index())
") 2>/dev/null; chk $? "ToSFiT modules + protein reward"

hdr "8. BORA  (Ablatif6c/bora-the-explorer)"
(cd $R/bora-the-explorer && PYTHONPATH=. $E/bl_bora/bin/python $B/tests/smoke_bora.py) 2>&1 | grep -E "SMOKE|FAIL"
chk ${PIPESTATUS[0]} "BORA policy + BO backend"

hdr "9. BOLT  (Yimeng-Zeng/BOLT)"
(cd /root/data/cys/BOLT && PYTHONPATH=optimization/peptides /root/data/cys/envs/bolt_clean_venv/bin/python -c "
from lolbo.lolbo import LOLBOState
from lolbo.latent_space_objective import LatentSpaceObjective
import os
vae='optimization/peptides/uniref_vae/saved_models/dim128_k1_kl0001_eff256_dff256_pious-sea-2_model_state_epoch_118.pkl'
assert os.path.exists(vae), 'missing uniref VAE ckpt'
print('   BOLT peptide LOL-BO + uniref VAE ckpt present')
") 2>/dev/null; chk $? "BOLT peptide stack"

hdr "10. SemanticOpt  (no upstream repo - Unsloth LoRA SFT path)"
$E/bl_semanticopt/bin/python -c "
import torch, unsloth, trl, peft, transformers
print('   torch',torch.__version__,'GPUs',torch.cuda.device_count())
print('   unsloth + trl',trl.__version__,'+ peft',peft.__version__)
" 2>/dev/null | grep -v "^$"; chk $? "SemanticOpt training stack"

printf '\n\033[1mTOTAL: %d passed, %d failed\033[0m\n' $pass $fail
[ $fail -eq 0 ]
