#!/usr/bin/env bash
set -euo pipefail

worker=/root/data/logs/peptide_bo_release_20261003_v4/tools/peptide_bo_campaign.py
campaign=/root/data/logs/peptide_bo_verified_20261003_v4
python=/root/data/cys/envs/bolt_clean_venv/bin/python

if [[ $# -eq 0 ]]; then
    exec ssh -o ConnectTimeout=10 LLMOptimizer3 "$python $worker verify --campaign $campaign"
fi

if [[ $# -ne 3 || $1 != --launch || ! $2 =~ ^[0-9]+$ || ! $3 =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    printf 'Usage: %s [--launch GPU CPU_IDS]\n' "$0" >&2
    exit 2
fi

exec ssh -o ConnectTimeout=10 LLMOptimizer3 \
    "env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE $python -u $worker worker --campaign $campaign --gpu $2 --cpus $3 --split all --launch"
