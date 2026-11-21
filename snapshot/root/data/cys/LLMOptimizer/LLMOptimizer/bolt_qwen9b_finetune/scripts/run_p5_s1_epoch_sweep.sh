#!/bin/bash
# Train P5 S1a / S1b from base Qwen3.5-9B (no S0 stack).
# Conservative (2.5 ep) vs standard (7.5 ep); cutoff_len=6016.
#
# TRAIN_MODE (default ddp2 — best for 4× A800):
#   ddp2  — 2-GPU DDP per job; batch ep2p5 pair then ep7p5 pair (all GPUs busy)
#   single4 — 4 single-GPU jobs in parallel (ep2p5 GPUs idle before ep7p5 done)
#   single2 — legacy 2 single-GPU jobs at a time
#
# Usage:
#   TRAIN_MODE=ddp2 bash scripts/run_p5_s1_epoch_sweep.sh
#   CONFIG=... GPUS=0,1 bash scripts/run_p5_s1_epoch_sweep.sh
#
set -euo pipefail

REPO="/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune"
cd "$REPO"

PYTHON="${PYTHON:-/root/miniconda3/envs/llamafactory_qwen35/bin/python}"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-/root/miniconda3/envs/llamafactory_qwen35/bin/llamafactory-cli}"
CONDA_BIN="$(dirname "$LLAMAFACTORY_CLI")"
CONFIG_DIR="$REPO/configs/ml"
LOG_DIR="${LOG_DIR:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune/p5_s1_epoch_sweep}"
TRAIN_MODE="${TRAIN_MODE:-ddp2}"
mkdir -p "$LOG_DIR"

if [[ ! -d /root/eb-public/huggingface-models/Qwen/Qwen3.5-9B ]]; then
  echo "[error] Missing base model Qwen3.5-9B" >&2
  exit 1
fi

"$PYTHON" scripts/generate_p5_s1_epoch_configs.py

ALL_CONFIGS=(
  llamafactory_qwen35_9b_lora_p5_s1a_ep2p5.yaml
  llamafactory_qwen35_9b_lora_p5_s1b_ep2p5.yaml
  llamafactory_qwen35_9b_lora_p5_s1a_ep7p5.yaml
  llamafactory_qwen35_9b_lora_p5_s1b_ep7p5.yaml
)

VISIBLE_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
NUM_GPUS="${NUM_GPUS:-$VISIBLE_GPUS}"
if [[ "$NUM_GPUS" -gt 4 ]]; then NUM_GPUS=4; fi
if [[ "$NUM_GPUS" -lt 1 ]]; then NUM_GPUS=1; fi

output_dir_for() {
  grep '^output_dir:' "$CONFIG_DIR/$1" | awk '{print $2}'
}

adapter_done() {
  [[ -f "$(output_dir_for "$1")/adapter_model.safetensors" ]]
}

latest_checkpoint() {
  local out="$1"
  local ckpt
  ckpt="$(ls -d "$out"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)"
  [[ -n "$ckpt" && -d "$ckpt" ]] && echo "$ckpt"
}

prepare_run_cfg() {
  local path="$1"
  local stem="$2"
  local ckpt="$3"
  local use_ddp="$4"
  local tmp_cfg
  tmp_cfg="$(mktemp "/tmp/${stem}.XXXX.yaml")"
  cp "$path" "$tmp_cfg"
  if [[ "$use_ddp" == "1" ]]; then
    # Keep effective batch = 2*8=16 on 1 GPU → 2*2*4=16 on 2-GPU DDP.
    sed -i 's/gradient_accumulation_steps: 8/gradient_accumulation_steps: 4/' "$tmp_cfg"
  fi
  if [[ -n "$ckpt" ]]; then
    sed -i "s|^resume_from_checkpoint: null|resume_from_checkpoint: ${ckpt}|" "$tmp_cfg"
  fi
  echo "$tmp_cfg"
}

run_one() {
  local cfg="$1"
  local gpus="$2"
  local master_port="${3:-}"
  local path="$CONFIG_DIR/$cfg"
  local stem="${cfg%.yaml}"
  local log="$LOG_DIR/${stem}.log"
  local out
  out="$(output_dir_for "$cfg")"
  mkdir -p "$out"

  if adapter_done "$cfg"; then
    echo "[$(date -Is)] SKIP $cfg — adapter already at $out"
    return 0
  fi

  local ckpt=""
  ckpt="$(latest_checkpoint "$out" || true)"
  local n_gpus
  n_gpus="$(echo "$gpus" | tr ',' '\n' | wc -l)"
  local use_ddp=0
  [[ "$n_gpus" -ge 2 ]] && use_ddp=1

  local tmp_cfg
  tmp_cfg="$(prepare_run_cfg "$path" "$stem" "$ckpt" "$use_ddp")"
  local run_cfg="$tmp_cfg"

  if [[ -n "$ckpt" ]]; then
    echo "[$(date -Is)] RESUME $cfg from $ckpt on GPUs $gpus (ddp=$use_ddp)"
  else
    echo "[$(date -Is)] START $cfg on GPUs $gpus (ddp=$use_ddp)"
  fi
  echo "log: $log"

  local -a env_args=(
    "PATH=${CONDA_BIN}:${PATH}"
    "CUDA_VISIBLE_DEVICES=${gpus}"
    "PYTHONUNBUFFERED=1"
    "TOKENIZERS_PARALLELISM=false"
    "OMP_NUM_THREADS=4"
  )
  if [[ "$use_ddp" == "1" ]]; then
    env_args+=(
      "FORCE_TORCHRUN=1"
      "NCCL_P2P_DISABLE=1"
      "NCCL_IB_DISABLE=1"
      "MASTER_PORT=${master_port:-29500}"
    )
  fi

  env "${env_args[@]}" "$LLAMAFACTORY_CLI" train "$run_cfg" >>"$log" 2>&1
  local rc=$?
  rm -f "$tmp_cfg"
  if [[ $rc -ne 0 ]]; then
    echo "[error] $cfg failed (exit $rc); see $log" >&2
    return $rc
  fi
}

if [[ -n "${CONFIG:-}" ]]; then
  run_one "$CONFIG" "${GPUS:-0}" "${MASTER_PORT:-29500}"
  exit 0
fi

echo "[sweep] TRAIN_MODE=$TRAIN_MODE NUM_GPUS=$NUM_GPUS (visible=$VISIBLE_GPUS)"

run_batch_ddp2() {
  local cfg_a="$1" cfg_b="$2" port_a="$3" port_b="$4"
  local pids=()
  if ! adapter_done "$cfg_a"; then
    run_one "$cfg_a" "0,1" "$port_a" &
    pids+=($!)
  fi
  if ! adapter_done "$cfg_b"; then
    run_one "$cfg_b" "2,3" "$port_b" &
    pids+=($!)
  fi
  if [[ ${#pids[@]} -gt 0 ]]; then
    for pid in "${pids[@]}"; do wait "$pid"; done
  fi
}

case "$TRAIN_MODE" in
  ddp2)
    if [[ "$NUM_GPUS" -lt 4 ]]; then
      echo "[error] TRAIN_MODE=ddp2 needs 4 GPUs (have $NUM_GPUS)" >&2
      exit 1
    fi
    echo "[batch 1/2] ep2p5 conservative: s1a on 0,1 + s1b on 2,3 (2-GPU DDP each)"
    run_batch_ddp2 "${ALL_CONFIGS[0]}" "${ALL_CONFIGS[1]}" 29501 29502
    echo "[batch 2/2] ep7p5 standard: s1a on 0,1 + s1b on 2,3 (2-GPU DDP each)"
    run_batch_ddp2 "${ALL_CONFIGS[2]}" "${ALL_CONFIGS[3]}" 29503 29504
    ;;
  single4)
    if [[ "$NUM_GPUS" -ge 4 ]]; then
      pids=()
      for i in 0 1 2 3; do
        run_one "${ALL_CONFIGS[$i]}" "$i" &
        pids+=($!)
      done
      for pid in "${pids[@]}"; do wait "$pid"; done
    else
      echo "[error] TRAIN_MODE=single4 needs 4 GPUs" >&2
      exit 1
    fi
    ;;
  single2|*)
    run_one "${ALL_CONFIGS[0]}" 0 &
    pid_a=$!
    run_one "${ALL_CONFIGS[1]}" 1 &
    pid_b=$!
    wait "$pid_a" "$pid_b"
    run_one "${ALL_CONFIGS[2]}" 0 &
    pid_a=$!
    run_one "${ALL_CONFIGS[3]}" 1 &
    pid_b=$!
    wait "$pid_a" "$pid_b"
    ;;
esac

echo ""
echo "[Done] S1 epoch sweep complete."
echo "Weights:"
for cfg in "${ALL_CONFIGS[@]}"; do
  out="$(output_dir_for "$cfg")"
  if adapter_done "$cfg"; then
    echo "  OK  $out"
  else
    ckpt="$(latest_checkpoint "$out" || true)"
    if [[ -n "$ckpt" ]]; then
      echo "  INCOMPLETE (checkpoint) $out -> $ckpt"
    else
      echo "  MISSING $out"
    fi
  fi
done
