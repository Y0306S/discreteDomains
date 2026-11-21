#!/usr/bin/env bash
# Download peptide VAE + APEX oracle weights from Yimeng-Zeng/APEXGo (public).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APEXGO="${APEXGO_BASE:-https://github.com/Yimeng-Zeng/APEXGo/raw/main}"

VAE_DIR="$ROOT/optimization/peptides/uniref_vae/saved_models"
VAE_FILE="dim128_k1_kl0001_eff256_dff256_pious-sea-2_model_state_epoch_118.pkl"
APEX_DIR="$ROOT/optimization/peptides/apex_oracle/APEX_pathogen_models"

mkdir -p "$VAE_DIR" "$APEX_DIR"

download() {
  local rel="$1"
  local dest="$2"
  if [[ -f "$dest" ]] && [[ "$(stat -c%s "$dest" 2>/dev/null || echo 0)" -gt 1000000 ]]; then
    echo "[skip] $(basename "$dest")"
    return 0
  fi
  echo "[get] $rel"
  mkdir -p "$(dirname "$dest")"
  curl -fL --retry 5 --retry-delay 5 --connect-timeout 30 --max-time 7200 --continue-at - -o "$dest" "$APEXGO/$rel"
}

download "optimization/uniref_vae/saved_models/$VAE_FILE" "$VAE_DIR/$VAE_FILE"

APEX_MODELS=(
  'APEX_2&256&2048&1e-05&0.0&0.1'
  'APEX_3&128&1024&1e-06&0.001&2.0'
  'APEX_3&128&2048&1e-06&0.0&1.0'
  'APEX_3&128&2048&1e-06&0.001&0.1'
  'APEX_3&128&2048&1e-06&0.001&2.0'
  'APEX_3&256&2048&1e-05&0.001&0.1'
  'APEX_3&256&2048&1e-06&0.001&0.1'
  'APEX_3&256&512&1e-06&0.0&2.0'
)

for name in "${APEX_MODELS[@]}"; do
  # GitHub raw URLs need encoding for '&'
  enc=$(python3 -c "import urllib.parse; print(urllib.parse.quote('''$name''', safe=''))")
  download "optimization/apex_oracle/APEX_pathogen_models/$enc" "$APEX_DIR/$name"
done

echo "[prepare_peptide_assets] Done."
ls -lh "$VAE_DIR/$VAE_FILE"
ls -lh "$APEX_DIR" | head -12
