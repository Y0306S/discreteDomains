#!/usr/bin/env bash
# ORPO HP sweep on s1b_ep7p5 + p5_s2_dpo_rationale (same recipe as dpo_rationale_G).
#
# Locked:
#   init: s1b_ep7p5, loss=orpo, epochs=1, grad_accum=8, pref_ftx=0, seed=42
# Sweep (9 runs):
#   learning_rate ∈ {2e-6, 5e-6, 1e-5}
#   pref_beta     ∈ {0.05, 0.1, 0.3}
#
# Then algo40 eval P5/H0 ptllm_tpe_qwen with rationale_json.
#
# Usage:
#   bash examples/benchmark/run_s1b_rg_orpo_hp_sweep.sh
#   TRAIN_ONLY=1 bash ...
#   EVAL_ONLY=1 bash ...
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

CFG_DIR="${CFG_DIR:-/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/configs/ml/s1b_rg_orpo_sweep}"
TEMPLATE="${TEMPLATE:-/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/configs/ml/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_rationale_G.yaml}"
INIT="${INIT:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5}"
WEIGHT_ROOT="${WEIGHT_ROOT:-/root/data/cys/blockdata_archive/model-weights}"
LOGDIR="${LOGDIR:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
LF="${LLAMAFACTORY_CLI:-/root/miniconda3/envs/llamafactory_qwen35/bin/llamafactory-cli}"
LF_ROOT="${LF_ROOT:-/root/LLMOptimizer/LlamaFactory}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
PIPE="$LOGDIR/s1b_rg_orpo_hp_sweep_${TS}.log"
MANIFEST="${MANIFEST:-benchmark_data/rl_task_selection/s1b_rg_orpo_hp_sweep_manifest.json}"
WANDB_PROJECT="${WANDB_PROJECT:-llmoptimizer}"
WANDB_ENTITY="${WANDB_ENTITY:-cys36-nanyang-technological-university-singapore}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-15000}"
EVAL_RESUME="${EVAL_RESUME:-1}"
SEED="${SEED:-1}"
TRAIN_ONLY="${TRAIN_ONLY:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"

LRS=(2e-6 5e-6 1e-5)
BETAS=(0.05 0.1 0.3)

# shellcheck source=examples/benchmark/orpo_resume_helpers.sh
source "$REPO/examples/benchmark/orpo_resume_helpers.sh"

mkdir -p "$LOGDIR" "$CFG_DIR"
export PATH="/root/miniconda3/envs/llamafactory_qwen35/bin:$PATH"
export WANDB_PROJECT WANDB_ENTITY
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

tag_lr() {
  case "$1" in
    2e-6|2.0e-6) echo "2e6" ;;
    5e-6|5.0e-6) echo "5e6" ;;
    1e-5|1.0e-5) echo "1e5" ;;
    *) echo "$1" | tr -d '-' | tr '.' 'p' ;;
  esac
}

tag_beta() {
  case "$1" in
    0.05) echo "05" ;;
    0.1|0.10) echo "10" ;;
    0.3|0.30) echo "30" ;;
    *) echo "$1" | tr -d '.' ;;
  esac
}

arm_id() {
  local lr="$1" beta="$2"
  echo "s1b_rg_orpo_lr$(tag_lr "$lr")_b$(tag_beta "$beta")"
}

adapter_path() {
  echo "$WEIGHT_ROOT/llamafactory_qwen35_9b_lora_p5_$(arm_id "$1" "$2")"
}

write_yaml() {
  local lr="$1" beta="$2"
  local arm out_dir yaml lr_sci
  arm="$(arm_id "$lr" "$beta")"
  out_dir="$(adapter_path "$lr" "$beta")"
  yaml="$CFG_DIR/${arm}.yaml"
  # normalize scientific form for yaml
  case "$lr" in
    2e-6) lr_sci="2.0e-6" ;;
    5e-6) lr_sci="5.0e-6" ;;
    1e-5) lr_sci="1.0e-5" ;;
    *) lr_sci="$lr" ;;
  esac
  cat >"$yaml" <<EOF
### model
# ORPO HP sweep arm: lr=${lr_sci} pref_beta=${beta} (from dpo_rationale_G)
model_name_or_path: /root/eb-public/huggingface-models/Qwen/Qwen3.5-9B
adapter_name_or_path: ${INIT}
trust_remote_code: true

### method
stage: dpo
do_train: true
finetuning_type: lora
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target: all
pref_beta: ${beta}
pref_loss: orpo
pref_ftx: 0.0

### dataset
dataset_dir: /root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/llamafactory_data
dataset: p5_s2_dpo_rationale
template: qwen3_5_nothink
cutoff_len: 6016
preprocessing_num_workers: 8
dataloader_num_workers: 2
overwrite_cache: true
packing: false

### output
output_dir: ${out_dir}
logging_steps: 5
save_steps: 100
plot_loss: true
overwrite_output_dir: true
save_only_model: false
save_total_limit: 2
report_to: wandb
run_name: ${arm}

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: ${lr_sci}
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
gradient_checkpointing: true
ddp_timeout: 180000000
resume_from_checkpoint: null
seed: 42
EOF
  echo "$yaml"
}

train_one() {
  local lr="$1" beta="$2" gpu="$3"
  local arm out_dir yaml tlog mode ckpt
  arm="$(arm_id "$lr" "$beta")"
  out_dir="$(adapter_path "$lr" "$beta")"
  tlog="$LOGDIR/s1b_rg_orpo_train_${arm}_${TS}.log"

  if [[ -f "$out_dir/adapter_model.safetensors" ]]; then
    log "SKIP train $arm (adapter exists)"
    return 0
  fi

  yaml="$(write_yaml "$lr" "$beta")"
  mode="$(orpo_patch_yaml_resume "$yaml" "$out_dir")"
  if [[ "$mode" == "resume" ]]; then
    ckpt="$(orpo_latest_checkpoint "$out_dir")"
    log "RESUME train $arm from $ckpt"
  fi
  log "TRAIN $arm gpu=$gpu lr=$lr beta=$beta mode=$mode yaml=$yaml log=$tlog"
  cd "$LF_ROOT"
  CUDA_VISIBLE_DEVICES="$gpu" DISABLE_VERSION_CHECK=1 WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_ENTITY="$WANDB_ENTITY" \
    "$LF" train "$yaml" >"$tlog" 2>&1
  local rc=$?
  log "TRAIN $arm finished rc=$rc"
  return "$rc"
}

write_manifest() {
  local out="$1"
  local lr beta arm adapter
  local rows="["
  local first=1
  for lr in "${LRS[@]}"; do
    for beta in "${BETAS[@]}"; do
      arm="$(arm_id "$lr" "$beta")"
      adapter="$(adapter_path "$lr" "$beta")"
      if [[ ! -f "$adapter/adapter_model.safetensors" ]]; then
        log "ERROR missing adapter $adapter"
        return 1
      fi
      if [[ $first -eq 1 ]]; then first=0; else rows+=","; fi
      rows+=$(printf '\n  {"id":"%s","adapter":"%s","resp":"rationale_json","lr":"%s","pref_beta":"%s"}' \
        "$arm" "$adapter" "$lr" "$beta")
    done
  done
  rows+=$'\n]'
  printf '%s\n' "$rows" >"$out"
  log "wrote manifest ($(grep -c '"id"' "$out") models) -> $out"
}

run_train_queue() {
  log "=== TRAIN queue: ${#LRS[@]} lrs × ${#BETAS[@]} betas = $((${#LRS[@]} * ${#BETAS[@]})) on ${NUM_GPUS} GPUs ==="
  declare -a PIDS=() GPUS_USED=() ARMS=()
  local lr beta gpu i

  for lr in "${LRS[@]}"; do
    for beta in "${BETAS[@]}"; do
      while true; do
        for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
          local busy=0
          for i in "${!PIDS[@]}"; do
            if [[ "${GPUS_USED[$i]}" == "$gpu" ]] && kill -0 "${PIDS[$i]}" 2>/dev/null; then
              busy=1
              break
            fi
          done
          if [[ $busy -eq 0 ]]; then
            train_one "$lr" "$beta" "$gpu" &
            PIDS+=($!)
            GPUS_USED+=("$gpu")
            ARMS+=("$(arm_id "$lr" "$beta")")
            sleep 2
            break 2
          fi
        done
        wait -n 2>/dev/null || true
        local new_pids=() new_gpus=() new_arms=()
        for i in "${!PIDS[@]}"; do
          if kill -0 "${PIDS[$i]}" 2>/dev/null; then
            new_pids+=("${PIDS[$i]}")
            new_gpus+=("${GPUS_USED[$i]}")
            new_arms+=("${ARMS[$i]}")
          else
            wait "${PIDS[$i]}" || log "WARN train failed arm=${ARMS[$i]} rc=$?"
          fi
        done
        PIDS=("${new_pids[@]:-}")
        GPUS_USED=("${new_gpus[@]:-}")
        ARMS=("${new_arms[@]:-}")
      done
    done
  done
  wait || true
  log "=== TRAIN queue done ==="
}

run_eval() {
  log "writing manifest -> $MANIFEST"
  write_manifest "$MANIFEST"
  log "=== EVAL algo40 from_manifest (rationale_json) ==="
  NUM_GPUS="$NUM_GPUS" JOBS_PER_GPU="$JOBS_PER_GPU" PACK_MIN_FREE_MB="$PACK_MIN_FREE_MB" \
    STAGE=from_manifest MANIFEST_JSON="$MANIFEST" HYBRID=H0 PROMPT_VARIANT=P5 \
    LLM_OPTIMIZERS=ptllm_tpe_qwen SEED="$SEED" EVAL_RESUME="$EVAL_RESUME" TIME_LIMIT=10800 \
    bash examples/benchmark/run_mltest20_algo40_eval.sh 2>&1 | tee -a "$PIPE"
  log "regen tables"
  python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm 2>&1 | tee -a "$PIPE"
  python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm --seed1-only 2>&1 | tee -a "$PIPE"
  log "=== EVAL done ==="
}

log "=== s1b_rg_orpo HP sweep START ts=$TS ==="
log "init=$INIT template=$TEMPLATE"
log "wandb=$WANDB_ENTITY/$WANDB_PROJECT"
log "grid: lr=${LRS[*]}  beta=${BETAS[*]}"

if [[ ! -f "$INIT/adapter_model.safetensors" ]]; then
  log "ERROR missing init $INIT"
  exit 1
fi
if [[ ! -f /root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/p5_s2_dpo_rationale_train.jsonl ]]; then
  log "ERROR missing rationale train jsonl"
  exit 1
fi

if [[ "$EVAL_ONLY" != "1" ]]; then
  run_train_queue
fi

if [[ "$TRAIN_ONLY" != "1" ]]; then
  run_eval
fi

log "=== s1b_rg_orpo HP sweep ALL DONE ==="
