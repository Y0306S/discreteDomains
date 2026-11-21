#!/usr/bin/env bash
# Queue: CEB query-plan VAE then optional peptide VAE. Single GPU 0, full device memory.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT/assets_cache/vae_train_logs}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

# Paper VAEs fit on one A800; keep GPUs 1–3 free for other jobs.
TRAIN_GPUS="${TRAIN_GPUS:-0}"
GPU_MEM_FRACTION="${GPU_MEM_FRACTION:-1.0}"
CEB_EPOCHS="${CEB_EPOCHS:-50}"
CEB_BATCH="${CEB_BATCH:-256}"
RETRAIN_PEPTIDE_VAE="${RETRAIN_PEPTIDE_VAE:-0}"
SYNTHETIC_SMOKE="${SYNTHETIC_SMOKE:-0}"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG_DIR/queue_${TS}.log"; }

log "=== VAE training queue start ==="
log "TRAIN_GPUS=$TRAIN_GPUS GPU_MEM_FRACTION=$GPU_MEM_FRACTION RETRAIN_PEPTIDE_VAE=$RETRAIN_PEPTIDE_VAE"

QP="$ROOT/optimization/query_plans"
if command -v uv >/dev/null 2>&1; then
  PY=(uv run --directory "$QP" python)
else
  PY=(poetry -C "$QP" run python)
fi

ENC="$ROOT/optimization/query_plans/query_plan_optimization/training_data/encoded_ceb"
CEB_OUT="$ROOT/optimization/query_plans/query_plan_optimization/vae/CEB_64.ckpt"

# --- Phase 1: CEB VAE ---
if [[ -f "$CEB_OUT" ]] && [[ "$(stat -c%s "$CEB_OUT")" -gt 1000000 ]]; then
  log "CEB_64.ckpt already exists ($(stat -c%s "$CEB_OUT") bytes) — skip train (delete to retrain)"
else
  if [[ ! -d "$ENC" ]] || [[ -z "$(ls -A "$ENC"/*.csv 2>/dev/null)" ]]; then
    if [[ "$SYNTHETIC_SMOKE" == "1" ]]; then
      log "No encoded_ceb CSVs; running SYNTHETIC_SMOKE train (not production quality)"
    else
      log "No encoded plan data at $ENC"
      log "Start IMDB docker, then: export DB_HOST=... && bash $ROOT/scripts/collect_ceb_plans_for_vae.sh"
      log "Or set SYNTHETIC_SMOKE=1 to test the training loop only."
      exit 1
    fi
  fi

  # Use first GPU in TRAIN_GPUS for Lightning (single-process train; mem fraction caps footprint)
  PRIMARY_GPU="${TRAIN_GPUS%%,*}"
  log "Phase 1: train CEB VAE on GPU $PRIMARY_GPU (mem fraction $GPU_MEM_FRACTION)"
  EXTRA=()
  [[ "$SYNTHETIC_SMOKE" == "1" ]] && EXTRA+=(--synthetic-smoke)

  (
    export CUDA_VISIBLE_DEVICES="$PRIMARY_GPU"
    cd "$ROOT"
    "${PY[@]}" scripts/train_ceb_vae.py \
      --encoded-dir "$ENC" \
      --out "$CEB_OUT" \
      --max-epochs "$CEB_EPOCHS" \
      --batch-size "$CEB_BATCH" \
      --gpu-mem-fraction "$GPU_MEM_FRACTION" \
      --gpus 1 \
      "${EXTRA[@]}" \
      >"$LOG_DIR/ceb_vae_${TS}.log" 2>&1
  )
  log "Phase 1 done — see $LOG_DIR/ceb_vae_${TS}.log"
  bash "$ROOT/scripts/link_query_vae_ckpt.sh" "$CEB_OUT" || true
fi

# --- Phase 2: Peptide VAE (optional; you already have APEXGo epoch-118 weights) ---
PEPTIDE_CKPT="$ROOT/optimization/peptides/uniref_vae/saved_models/dim128_k1_kl0001_eff256_dff256_pious-sea-2_model_state_epoch_118.pkl"
if [[ "$RETRAIN_PEPTIDE_VAE" != "1" ]]; then
  log "Phase 2 skipped (RETRAIN_PEPTIDE_VAE=0). Using existing $PEPTIDE_CKPT"
  log "=== Queue complete ==="
  exit 0
fi

APEX_GEN="/root/data/cys/APEXGo/generation"
if [[ ! -f "$APEX_GEN/train_vae.py" ]]; then
  log "APEXGo generation/train_vae.py not found — cannot retrain peptide VAE"
  exit 1
fi

PEP_GPU="${PEP_GPU:-${TRAIN_GPUS%%,*}}"
log "Phase 2: peptide VAE retrain on GPU $PEP_GPU (queued after CEB)"
(
  export CUDA_VISIBLE_DEVICES="$PEP_GPU"
  export WANDB_MODE="${WANDB_MODE:-disabled}"
  cd "$APEX_GEN"
  # Conservative batch for ~30% VRAM on A800
  python train_vae.py \
    --batch_size 128 \
    --max_epochs "${PEPTIDE_VAE_EPOCHS:-200}" \
    --compute_val_freq 3 \
    --wandb_entity "${WANDB_ENTITY:-local}" \
    >"$LOG_DIR/peptide_vae_${TS}.log" 2>&1
  # Copy latest epoch pkl into BOLT if present
  LATEST=$(ls -t saved_models/*/*_epoch_*.pkl 2>/dev/null | head -1 || true)
  if [[ -n "$LATEST" ]]; then
    cp -f "$LATEST" "$PEPTIDE_CKPT"
    log "Copied peptide checkpoint -> $PEPTIDE_CKPT"
  fi
)
log "Phase 2 done — see $LOG_DIR/peptide_vae_${TS}.log"
log "=== Queue complete ==="
