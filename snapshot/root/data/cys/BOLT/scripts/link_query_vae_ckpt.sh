#!/usr/bin/env bash
# Symlink query-plan VAE checkpoint names expected by different scripts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VAE_DIR="$ROOT/optimization/query_plans/query_plan_optimization/vae"
CKPT="${1:-}"

if [[ -z "$CKPT" ]]; then
  if [[ -f "$VAE_DIR/CEB_64.ckpt" ]] && [[ "$(stat -c%s "$VAE_DIR/CEB_64.ckpt")" -gt 1000000 ]]; then
    CKPT="$VAE_DIR/CEB_64.ckpt"
  else
    echo "Usage: $0 /path/to/CEB_64.ckpt"
    echo "  Or place checkpoint at $VAE_DIR/CEB_64.ckpt first."
    exit 1
  fi
fi

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT"
  exit 1
fi

ln -sfn "$(realpath "$CKPT")" "$VAE_DIR/CEB_64.ckpt"
ln -sfn "$(realpath "$CKPT")" "$VAE_DIR/64.ckpt"
echo "[link_query_vae] CEB_64.ckpt and 64.ckpt -> $CKPT"
