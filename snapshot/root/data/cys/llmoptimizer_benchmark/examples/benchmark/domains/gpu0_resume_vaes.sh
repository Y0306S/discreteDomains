#!/usr/bin/env bash
# Resume UniRef AMP VAE (epoch 40 -> 118) then train query CEB VAE on GPU 0.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export WANDB_SILENT=true

PY="${PY:-/root/miniconda3/envs/hpo_benchmark/bin/python}"
APEX_GEN="/root/data/cys/APEXGo/generation"
RESUME_CKPT="$APEX_GEN/saved_models/uniref_offline/uniref_offline_model_state_epoch_40.pkl"
BOLT_CKPT="/root/data/cys/BOLT/optimization/peptides/uniref_vae/saved_models/dim128_k1_kl0001_eff256_dff256_pious-sea-2_model_state_epoch_118.pkl"
CEB_CKPT="/root/data/cys/BOLT/optimization/query_plans/query_plan_optimization/vae/CEB_64.ckpt"
ENC="/root/data/cys/BOLT/optimization/query_plans/query_plan_optimization/training_data/encoded_ceb"
LOGDIR="/root/data/logs"
mkdir -p "$LOGDIR"

exec 9>"$LOGDIR/gpu0.lock"
if ! flock -n 9; then
  echo "[gpu0-resume] another job holds $LOGDIR/gpu0.lock" >&2
  exit 1
fi

echo "[gpu0-resume] $(date -Is) start — UniRef resume then query VAE"
echo "[gpu0-resume] UniRef: load $RESUME_CKPT epochs 40->118 batch=1024 workers=16"
if [[ ! -s "$RESUME_CKPT" ]]; then
  echo "[gpu0-resume] missing resume ckpt: $RESUME_CKPT" >&2
  exit 1
fi
NCSV=$(find "$ENC" -maxdepth 1 -name '*.csv' 2>/dev/null | wc -l)
echo "[gpu0-resume] encoded_ceb CSVs: $NCSV"

cd "$APEX_GEN"
"$PY" train_vae.py \
  --load_ckpt "$RESUME_CKPT" \
  --start_epoch 40 \
  --max_epochs 118 \
  --lr 0.0002 \
  --batch_size 1024 \
  --num_workers 16 \
  --k 1 \
  --d_model 128 \
  --dropout 0.05 \
  --kl_factor 0.0001 \
  --encoder_dim_feedforward 256 \
  --decoder_dim_feedforward 256 \
  --encoder_num_layers 6 \
  --decoder_num_layers 6 \
  --wandb_entity ""

FINAL="$APEX_GEN/saved_models/uniref_offline/uniref_offline_model_state_epoch_118.pkl"
if [[ -s "$FINAL" ]]; then
  cp "$FINAL" "$BOLT_CKPT"
  echo "[gpu0-resume] installed UniRef -> $BOLT_CKPT"
else
  echo "[gpu0-resume] WARNING: missing $FINAL" >&2
fi

if [[ "$NCSV" -lt 1 ]]; then
  echo "[gpu0-resume] no encoded_ceb CSVs; skip query VAE" >&2
fi

# Query VAE runs separately on GPU 3 when TRAIN_QUERY_VAE=0 (parallel mode).
if [[ "${TRAIN_QUERY_VAE:-1}" == "1" ]] && [[ "$NCSV" -ge 1 ]]; then
  echo "[gpu0-resume] query VAE: 50 epochs d_neck=32 batch=512 workers=8"
  export PYTHONPATH="/root/data/cys/BOLT/optimization/query_plans/query_plan_optimization:${PYTHONPATH:-}"
  cd /root/data/cys/BOLT/optimization/query_plans/query_plan_optimization
  "$PY" /root/data/cys/BOLT/scripts/train_ceb_vae.py \
    --d-neck 32 \
    --max-epochs 50 \
    --batch-size 512 \
    --num-workers 8 \
    --gpus 1 \
    --gpu-mem-fraction 1.0 \
    --out "$CEB_CKPT"
  echo "[gpu0-resume] query VAE done"
elif [[ "${TRAIN_QUERY_VAE:-1}" == "0" ]]; then
  echo "[gpu0-resume] skip query VAE (TRAIN_QUERY_VAE=0; running elsewhere)"
fi

echo "[gpu0-resume] $(date -Is) all done"
