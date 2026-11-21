#!/usr/bin/env bash
# Sequential GPU-0 pipeline:
#   1) AMP UniRef VAE (118 ep)
#   2) Query CEB VAE (50 ep, after EXPLAIN CSVs)
#   3) AMP + query BO (TPE 80 × 100)
#   4) Structure-track VAEs (GFP / AAV / GB1) last
# CPU plan collection is started separately. Do not launch a second copy.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export WANDB_SILENT=true
export QUERYOPT_PG_DSN="${QUERYOPT_PG_DSN:-host=127.0.0.1 port=5432 dbname=imdb user=imdb password=imdb}"
export DB_HOST="${DB_HOST:-127.0.0.1}"
export DB_PORT="${DB_PORT:-5432}"
export DB_USER="${DB_USER:-imdb}"
export DB_PASSWORD="${DB_PASSWORD:-imdb}"
export DB_NAME="${DB_NAME:-imdb}"

PY="${PY:-/root/miniconda3/envs/hpo_benchmark/bin/python}"
QP_PY="${QP_PY:-/root/miniconda3/envs/hpo_benchmark/bin/python}"
APEX_GEN="/root/data/cys/APEXGo/generation"
UNIREF_CSV="$APEX_GEN/data/uniref-cropped.csv"
BOLT_CKPT="/root/data/cys/BOLT/optimization/peptides/uniref_vae/saved_models/dim128_k1_kl0001_eff256_dff256_pious-sea-2_model_state_epoch_118.pkl"
AUTHOR_CKPT="/root/data/cys/BOLT/optimization/peptides/uniref_vae/saved_models/dim128_k1_kl0001_eff256_dff256_pious-sea-2_model_state_epoch_118_pretrained_authors.pkl"
ENC="/root/data/cys/BOLT/optimization/query_plans/query_plan_optimization/training_data/encoded_ceb"
CEB_CKPT="/root/data/cys/BOLT/optimization/query_plans/query_plan_optimization/vae/CEB_64.ckpt"
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
PG="$REPO/history_data/domain_protein/proteingym"
LOGDIR="/root/data/logs"
mkdir -p "$LOGDIR"

exec 9>"$LOGDIR/gpu0.lock"
if ! flock -n 9; then
  echo "[gpu0] another GPU-0 job holds $LOGDIR/gpu0.lock" >&2
  exit 1
fi

echo "[gpu0] $(date -Is) pipeline start CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} (physical GPU 0 only)"
echo "[gpu0] order: UniRef AMP VAE → query VAE → AMP/query BO → structure VAEs"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -i 0 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
fi

wait_file() {
  local path="$1" min_bytes="${2:-1000}" timeout_s="${3:-0}"
  echo "[gpu0] waiting for $path (>= ${min_bytes} bytes) timeout=${timeout_s}s"
  local start
  start=$(date +%s)
  while true; do
    if [[ -s "$path" ]]; then
      local sz
      sz=$(stat -c%s "$path" 2>/dev/null || echo 0)
      if [[ "$sz" -ge "$min_bytes" ]]; then
        return 0
      fi
    fi
    if [[ "$timeout_s" -gt 0 ]]; then
      local now
      now=$(date +%s)
      if (( now - start >= timeout_s )); then
        echo "[gpu0] timeout waiting for $path"
        return 1
      fi
    fi
    sleep 30
  done
}

wait_dir_csvs() {
  local dir="$1"
  echo "[gpu0] waiting for CSVs in $dir"
  while true; do
    local n
    n=$(find "$dir" -maxdepth 1 -type f -name '*.csv' 2>/dev/null | wc -l)
    if [[ "$n" -ge 1 ]]; then
      return 0
    fi
    sleep 30
  done
}

train_one_landscape() {
  local ls_name="$1"
  local csv="$PG/$ls_name/vae_sequences.csv"
  if [[ ! -s "$csv" ]]; then
    csv="$PG/$ls_name/sequences.csv"
  fi
  if [[ ! -s "$csv" ]]; then
    echo "[gpu0] skip landscape $ls_name (no sequences csv)"
    return 0
  fi
  local out="$PG/$ls_name/vae/epoch_118.pkl"
  local meta="${out%.pkl}.meta.json"
  if [[ -s "$meta" ]]; then
    echo "[gpu0] skip landscape $ls_name (complete $meta)"
    return 0
  fi
  local max_len=256 bs=128
  case "$ls_name" in
    gb1) max_len=480; bs=64 ;;
    gfp) max_len=256; bs=128 ;;
    aav) max_len=800; bs=32 ;;
  esac
  echo "[gpu0] landscape VAE $ls_name max_len=$max_len batch=$bs on GPU 0"
  "$PY" "$REPO/examples/benchmark/domains/train_landscape_vae.py" \
    --csv "$csv" \
    --out "$out" \
    --max-epochs 118 \
    --batch-size "$bs" \
    --max-string-length "$max_len"
}

# Keep the published author weights even if we overwrite BOLT_CKPT with a local retrain.
if [[ -s "$BOLT_CKPT" && ! -s "$AUTHOR_CKPT" ]]; then
  cp "$BOLT_CKPT" "$AUTHOR_CKPT"
  echo "[gpu0] preserved author UniRef ckpt -> $AUTHOR_CKPT"
fi

# --- 1. AMP UniRef VAE, 118 epochs (CSV is on disk from Hugging Face) ---
if wait_file "$UNIREF_CSV" 100000000 120; then
  echo "[gpu0] UniRef AMP VAE 118 epochs on GPU 0 ($(du -h "$UNIREF_CSV" | awk '{print $1}'))"
  cd "$APEX_GEN"
  "$PY" train_vae.py \
    --lr 0.0002 \
    --batch_size 512 \
    --k 1 \
    --d_model 128 \
    --dropout 0.05 \
    --kl_factor 0.0001 \
    --encoder_dim_feedforward 256 \
    --decoder_dim_feedforward 256 \
    --encoder_num_layers 6 \
    --decoder_num_layers 6 \
    --max_epochs 118 \
    --wandb_entity ""
  NEW_CKPT=$(ls -1t "$APEX_GEN"/saved_models/*/*epoch_118.pkl 2>/dev/null | head -1 || true)
  if [[ -n "${NEW_CKPT:-}" && -s "$NEW_CKPT" ]]; then
    cp "$NEW_CKPT" "$BOLT_CKPT"
    echo "[gpu0] installed UniRef epoch-118 -> $BOLT_CKPT from $NEW_CKPT"
  else
    echo "[gpu0] WARNING: no epoch_118 pkl found under $APEX_GEN/saved_models; keeping $BOLT_CKPT" >&2
  fi
else
  echo "[gpu0] UniRef CSV unavailable; AMP BO will use existing ckpt at $BOLT_CKPT"
fi

# --- 2. Query VAE after encoded EXPLAIN CSVs exist ---
wait_dir_csvs "$ENC"
echo "[gpu0] training query VAE d_neck=32 50 epochs on GPU 0"
export PYTHONPATH="/root/data/cys/BOLT/optimization/query_plans/query_plan_optimization:${PYTHONPATH:-}"
cd /root/data/cys/BOLT/optimization/query_plans/query_plan_optimization
"$QP_PY" /root/data/cys/BOLT/scripts/train_ceb_vae.py \
  --d-neck 32 \
  --max-epochs 50 \
  --gpus 1 \
  --gpu-mem-fraction 1.0 \
  --out "$CEB_CKPT"

# --- 3. AMP + query BO first (not structure) ---
cd "$REPO"
"$PY" examples/benchmark/domains/make_tune_lock_split.py --domain protein
"$PY" examples/benchmark/domains/build_domain_experience.py --domain protein
"$PY" examples/benchmark/domains/make_tune_lock_split.py --domain query
"$PY" examples/benchmark/domains/build_domain_experience.py --domain query
echo "[gpu0] 8-trial smoke AMP + query on GPU 0"
"$PY" examples/benchmark/domains/smoke_paper_oracles.py \
  --n-trials 8 --device cuda:0 --skip-dms --skip-esmfold
echo "[gpu0] TPE BO 80×100 AMP then query on GPU 0"
"$PY" examples/benchmark/domains/run_bolt_bo_suite.py \
  --domain both --n-datasets 100 --n-trials 80 --device cuda:0 --skip-done

# --- 4. Structure-track landscape VAEs last ---
for ls_name in gfp aav gb1; do
  train_one_landscape "$ls_name"
done
for ls_name in gfp gb1 aav; do
  mkdir -p "$REPO/history_data/domain_protein/structure/$ls_name"
done
echo "[gpu0] structure smoke (DMS + ESMFold) on GPU 0"
"$PY" examples/benchmark/domains/smoke_paper_oracles.py \
  --n-trials 8 --device cuda:0 --skip-amp --skip-query

echo "[gpu0] $(date -Is) pipeline done"
