#!/usr/bin/env bash
# Sequential ProteinGym stack on GPU 0 only: VAE → surrogates → BO probe.
# Skips stages whose artifacts already exist.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/root/data/cys/blockdata_archive/llmoptimizer_benchmark
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENDPOINT="${HF_HUB_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/data/cys/blockdata_archive/home-spill/dot-cache/huggingface}"
export PROTEINGYM_ESM_LOCAL="${PROTEINGYM_ESM_LOCAL:-/root/data/cys/blockdata_archive/home-spill/dot-cache/huggingface/hub/models--facebook--esm2_t12_35M_UR50D/snapshots/6fbf070e65b0b7291e7bbcd451118c216cff79d8}"
PY=/root/miniconda3/envs/hpo_benchmark/bin/python
REPO=/root/data/cys/blockdata_archive/llmoptimizer_benchmark
LOG=/root/data/logs
VAE_DIR="$REPO/history_data/domain_protein/proteingym/vae"
mkdir -p "$LOG" "$VAE_DIR"

echo "[pgym-gpu0] $(date -Is) start CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Mark GB1 deferred; GFP/AAV ready
$PY - << 'PY'
import json
from pathlib import Path
root = Path("/root/data/cys/blockdata_archive/llmoptimizer_benchmark/history_data/domain_protein/proteingym")
for ls, ready, note in [
    ("gfp", True, "Sarkisyan 2016; primary VAE+BO landscape"),
    ("aav", True, "Sinai 2021 AAV2; primary VAE+BO landscape"),
    ("gb1", False, "Full SPG1 448aa — not 55aa GB1 domain; BO deferred"),
]:
    meta_p = root / ls / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
    scored = root / ls / "scored_meta.json"
    if scored.is_file():
        meta.update(json.loads(scored.read_text()))
    meta["bo_ready"] = ready
    meta["note"] = note
    meta["msa_used"] = False
    meta_p.write_text(json.dumps(meta, indent=2) + "\n")
    print(ls, "bo_ready", ready, "n_scored", meta.get("n_scored") or meta.get("split_counts"))
PY

if [[ -f "$VAE_DIR/gfp_vae.pt" ]]; then
  echo "[pgym-gpu0] $(date -Is) skip GFP VAE (exists)"
else
  echo "[pgym-gpu0] $(date -Is) train GFP VAE"
  $PY -u "$REPO/examples/benchmark/domains/train_proteingym_vae.py" \
    --landscape gfp --device cuda:0 --epochs 25 \
    2>&1 | tee "$LOG/proteingym_vae_gfp_gpu0.log"
fi

if [[ -f "$VAE_DIR/aav_vae.pt" ]]; then
  echo "[pgym-gpu0] $(date -Is) skip AAV VAE (exists)"
else
  echo "[pgym-gpu0] $(date -Is) train AAV VAE"
  $PY -u "$REPO/examples/benchmark/domains/train_proteingym_vae.py" \
    --landscape aav --device cuda:0 --epochs 20 --batch-size 16 \
    2>&1 | tee "$LOG/proteingym_vae_aav_gpu0.log"
fi

if [[ -f "$REPO/history_data/domain_protein/proteingym/gfp/esm_fitness_head.pt" ]]; then
  echo "[pgym-gpu0] $(date -Is) skip GFP surrogate (exists)"
else
  echo "[pgym-gpu0] $(date -Is) train GFP surrogate"
  $PY -u "$REPO/examples/benchmark/domains/train_proteingym_surrogate.py" \
    --landscape gfp --device cuda:0 \
    2>&1 | tee "$LOG/proteingym_surr_gfp_gpu0.log"
fi

if [[ -f "$REPO/history_data/domain_protein/proteingym/aav/esm_fitness_head.pt" ]]; then
  echo "[pgym-gpu0] $(date -Is) skip AAV surrogate (exists)"
else
  echo "[pgym-gpu0] $(date -Is) train AAV surrogate"
  $PY -u "$REPO/examples/benchmark/domains/train_proteingym_surrogate.py" \
    --landscape aav --device cuda:0 \
    2>&1 | tee "$LOG/proteingym_surr_aav_gpu0.log"
fi

echo "[pgym-gpu0] $(date -Is) BO probe GFP+AAV"
$PY -u "$REPO/examples/benchmark/domains/run_proteingym_bo_suite.py" \
  --landscape both --device cuda:0 --n-starts 4 --n-trials 40 \
  --optimizer tpe --baselines --seed 1 \
  2>&1 | tee "$LOG/proteingym_bo_gpu0.log"

echo "[pgym-gpu0] $(date -Is) ALL DONE"
