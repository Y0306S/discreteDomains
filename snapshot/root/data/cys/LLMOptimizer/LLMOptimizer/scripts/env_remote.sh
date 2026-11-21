#!/usr/bin/env bash

# Source this file after connecting to a PKU-Hetu node:
#   source env_remote.sh

set -euo pipefail

SHARED_ROOT="/jizhicfs/pkuhetu"
WORKDIR="${SHARED_ROOT}/bcxu"
PROJECT_DIR="${WORKDIR}/warm_start_agent"
CONDA_ACTIVATE="/jizhicfs/lhy/activate"

if [[ -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}"
else
  echo "[WARN] Shared conda activate script not found: ${CONDA_ACTIVATE}"
fi

if [[ -f "${WORKDIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${WORKDIR}/.env"
  set +a
fi

if [[ -d "${PROJECT_DIR}" ]]; then
  cd "${PROJECT_DIR}"
elif [[ -d "${WORKDIR}" ]]; then
  cd "${WORKDIR}"
fi

NODE_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"

export WORKDIR

echo "[OK] Remote environment initialized."
echo "      host=$(hostname)"
echo "      ip=${NODE_IP:-unknown}"
echo "      workdir=$(pwd)"
echo "      gpus=${GPU_COUNT:-0}"
