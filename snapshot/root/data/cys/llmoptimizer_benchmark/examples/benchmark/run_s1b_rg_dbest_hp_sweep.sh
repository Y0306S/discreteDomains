#!/usr/bin/env bash
# RL HP sweep on s1b dpo_rationale_G using the same dbest + light16 setup as
# s1a_b_rw_dbest_ckpt16, then algo40 eval (P5/H0 ptllm_tpe_qwen).
#
# Locked:
#   init:   s1b_ep7p5_dpo_rationale_G
#   buffer: B_rw_dbest_light16
#   order:  light_16.json (dsseq, per_dataset)
#   ckpt:   16
#   grad_accum=8, max_grad_norm=1.0, seed=1
#
# Sweep grid (9 runs):
#   lr      ∈ {5e-7, 1e-6, 2e-6}
#   kl_beta ∈ {0.02, 0.05, 0.1}
#
# Usage:
#   bash examples/benchmark/run_s1b_rg_dbest_hp_sweep.sh
#   TRAIN_ONLY=1 bash ...   # skip eval
#   EVAL_ONLY=1 bash ...    # skip train, eval existing ckpts
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

INIT="${INIT:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_rationale_G}"
BUF="${BUF:-benchmark_data/rl_rollouts/B_rw_dbest_light16}"
ORDER="${ORDER:-benchmark_data/rl_task_selection/light_16.json}"
WEIGHT_ROOT="${WEIGHT_ROOT:-/root/data/cys/blockdata_archive/model-weights}"
LOGDIR="${LOGDIR:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
PYTHON_TRAIN="${PYTHON_TRAIN:-/root/miniconda3/envs/llamafactory_qwen35/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
PIPE="$LOGDIR/s1b_rg_dbest_hp_sweep_${TS}.log"
MANIFEST="${MANIFEST:-benchmark_data/rl_task_selection/s1b_rg_dbest_hp_sweep_manifest.json}"
WANDB_PROJECT="${WANDB_PROJECT:-llmoptimizer}"
WANDB_ENTITY="${WANDB_ENTITY:-cys36-nanyang-technological-university-singapore}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-15000}"
EVAL_RESUME="${EVAL_RESUME:-1}"
SEED="${SEED:-1}"
TRAIN_ONLY="${TRAIN_ONLY:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"

LRS=(5e-7 1e-6 2e-6)
KLS=(0.02 0.05 0.1)

mkdir -p "$LOGDIR"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

tag_lr() {
  case "$1" in
    5e-7|5.0e-7|0.0000005) echo "5e7" ;;
    1e-6|1.0e-6|0.000001) echo "1e6" ;;
    2e-6|2.0e-6|0.000002) echo "2e6" ;;
    *) echo "$1" | tr '.' 'p' | tr -d '-' ;;
  esac
}

tag_kl() {
  case "$1" in
    0.02) echo "02" ;;
    0.05) echo "05" ;;
    0.1|0.10) echo "10" ;;
    *) echo "$1" | tr -d '.' ;;
  esac
}

arm_id() {
  local lr="$1" kl="$2"
  echo "s1b_rg_dbest_lr$(tag_lr "$lr")_kl$(tag_kl "$kl")"
}

ckpt_path() {
  local arm="$1"
  echo "$WEIGHT_ROOT/llamafactory_qwen35_9b_lora_p5_${arm}/ckpt16"
}

train_one() {
  local lr="$1" kl="$2" gpu="$3"
  local arm mid ckpt_root final arm_log
  arm="$(arm_id "$lr" "$kl")"
  mid="${arm}_ckpt16"
  ckpt_root="$WEIGHT_ROOT/llamafactory_qwen35_9b_lora_p5_${arm}"
  final="$ckpt_root/final"
  arm_log="$LOGDIR/s1b_rg_train_${arm}_${TS}.log"
  mkdir -p "$ckpt_root"

  if [[ -f "$(ckpt_path "$arm")/adapter_model.safetensors" ]]; then
    log "SKIP train $mid (ckpt16 exists)"
    return 0
  fi

  log "TRAIN $mid gpu=$gpu lr=$lr kl=$kl log=$arm_log"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON_TRAIN" \
    examples/benchmark/train_p5_reinforce.py \
    --buffer-root "$BUF" \
    --init-adapter "$INIT" \
    --output-dir "$final" \
    --dataset-order-file "$ORDER" \
    --strict-dataset-order \
    --advantage-mode per_dataset \
    --dataset-sequential \
    --checkpoint-at 16 \
    --checkpoint-root "$ckpt_root" \
    --device cuda:0 \
    --lr "$lr" --kl-beta "$kl" --grad-accum 8 --max-grad-norm 1.0 --seed "$SEED" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    --wandb-run-name "$mid" \
    --wandb-group "s1b_rg_dbest_hp_sweep" \
    --wandb-tags "reinforce,s1b,dpo_rationale_G,dbest,light16,hp_sweep" \
    >"$arm_log" 2>&1
  local rc=$?
  log "TRAIN $mid finished rc=$rc"
  return "$rc"
}

write_manifest() {
  local out="$1"
  local lr kl arm mid adapter
  local rows="["
  local first=1
  for lr in "${LRS[@]}"; do
    for kl in "${KLS[@]}"; do
      arm="$(arm_id "$lr" "$kl")"
      mid="${arm}_ckpt16"
      adapter="$(ckpt_path "$arm")"
      if [[ ! -f "$adapter/adapter_model.safetensors" ]]; then
        log "ERROR missing adapter $adapter"
        return 1
      fi
      if [[ $first -eq 1 ]]; then first=0; else rows+=","; fi
      rows+=$(printf '\n  {"id":"%s","adapter":"%s","resp":"json_only","lr":"%s","kl_beta":"%s"}' \
        "$mid" "$adapter" "$lr" "$kl")
    done
  done
  rows+=$'\n]'
  printf '%s\n' "$rows" >"$out"
  log "wrote manifest ($(grep -c '"id"' "$out") models) -> $out"
}

run_train_queue() {
  log "=== TRAIN queue: ${#LRS[@]} lrs × ${#KLS[@]} kls = $((${#LRS[@]} * ${#KLS[@]})) on ${NUM_GPUS} GPUs ==="
  declare -a PIDS=() GPUS_USED=() ARMS=()
  local lr kl gpu i

  for lr in "${LRS[@]}"; do
    for kl in "${KLS[@]}"; do
      # find free GPU slot (one train per GPU)
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
            train_one "$lr" "$kl" "$gpu" &
            PIDS+=($!)
            GPUS_USED+=("$gpu")
            ARMS+=("$(arm_id "$lr" "$kl")")
            sleep 2
            break 2
          fi
        done
        # all GPUs busy — wait for any
        wait -n 2>/dev/null || true
        # prune dead pids
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
  log "=== EVAL algo40 from_manifest ==="
  NUM_GPUS="$NUM_GPUS" JOBS_PER_GPU="$JOBS_PER_GPU" PACK_MIN_FREE_MB="$PACK_MIN_FREE_MB" \
    STAGE=from_manifest MANIFEST_JSON="$MANIFEST" HYBRID=H0 PROMPT_VARIANT=P5 \
    LLM_OPTIMIZERS=ptllm_tpe_qwen SEED="$SEED" EVAL_RESUME="$EVAL_RESUME" TIME_LIMIT=10800 \
    bash examples/benchmark/run_mltest20_algo40_eval.sh 2>&1 | tee -a "$PIPE"
  log "regen tables"
  python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm 2>&1 | tee -a "$PIPE"
  python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm --seed1-only 2>&1 | tee -a "$PIPE"
  log "=== EVAL done ==="
}

log "=== s1b_rg_dbest HP sweep START ts=$TS ==="
log "init=$INIT buffer=$BUF order=$ORDER"
log "wandb=$WANDB_ENTITY/$WANDB_PROJECT group=s1b_rg_dbest_hp_sweep"

if [[ "$EVAL_ONLY" != "1" ]]; then
  [[ -f "$INIT/adapter_model.safetensors" ]] || { log "ERROR missing init $INIT"; exit 1; }
  [[ -d "$BUF" ]] || { log "ERROR missing buffer $BUF"; exit 1; }
  run_train_queue
fi

if [[ "$TRAIN_ONLY" != "1" ]]; then
  run_eval
fi

log "=== s1b_rg_dbest HP sweep ALL DONE ==="
#!/usr/bin/env bash
# RL HP sweep on s1b dpo_rationale_G using the same dbest + light16 setup as
# s1a_b_rw_dbest_ckpt16, then algo40 eval (P5/H0 ptllm_tpe_qwen).
#
# Locked:
#   init:   s1b_ep7p5_dpo_rationale_G
#   buffer: B_rw_dbest_light16
#   order:  light_16.json (dsseq, per_dataset)
#   ckpt:   16
#   grad_accum=8, max_grad_norm=1.0, seed=1
#
# Sweep grid (9 runs):
#   lr      ∈ {5e-7, 1e-6, 2e-6}
#   kl_beta ∈ {0.02, 0.05, 0.1}
#
# Usage:
#   bash examples/benchmark/run_s1b_rg_dbest_hp_sweep.sh
#   TRAIN_ONLY=1 bash ...   # skip eval
#   EVAL_ONLY=1 bash ...    # skip train, eval existing ckpts
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

INIT="${INIT:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_rationale_G}"
BUF="${BUF:-benchmark_data/rl_rollouts/B_rw_dbest_light16}"
ORDER="${ORDER:-benchmark_data/rl_task_selection/light_16.json}"
WEIGHT_ROOT="${WEIGHT_ROOT:-/root/data/cys/blockdata_archive/model-weights}"
LOGDIR="${LOGDIR:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
PYTHON_TRAIN="${PYTHON_TRAIN:-/root/miniconda3/envs/llamafactory_qwen35/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
PIPE="$LOGDIR/s1b_rg_dbest_hp_sweep_${TS}.log"
MANIFEST="${MANIFEST:-benchmark_data/rl_task_selection/s1b_rg_dbest_hp_sweep_manifest.json}"
WANDB_PROJECT="${WANDB_PROJECT:-llmoptimizer}"
WANDB_ENTITY="${WANDB_ENTITY:-cys36-nanyang-technological-university-singapore}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-15000}"
EVAL_RESUME="${EVAL_RESUME:-1}"
SEED="${SEED:-1}"
TRAIN_ONLY="${TRAIN_ONLY:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"

LRS=(5e-7 1e-6 2e-6)
KLS=(0.02 0.05 0.1)

mkdir -p "$LOGDIR"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

tag_lr() {
  case "$1" in
    5e-7|5.0e-7|0.0000005) echo "5e7" ;;
    1e-6|1.0e-6|0.000001) echo "1e6" ;;
    2e-6|2.0e-6|0.000002) echo "2e6" ;;
    *) echo "$1" | tr '.' 'p' | tr -d '-' ;;
  esac
}

tag_kl() {
  case "$1" in
    0.02) echo "02" ;;
    0.05) echo "05" ;;
    0.1|0.10) echo "10" ;;
    *) echo "$1" | tr -d '.' ;;
  esac
}

arm_id() {
  local lr="$1" kl="$2"
  echo "s1b_rg_dbest_lr$(tag_lr "$lr")_kl$(tag_kl "$kl")"
}

ckpt_path() {
  local arm="$1"
  echo "$WEIGHT_ROOT/llamafactory_qwen35_9b_lora_p5_${arm}/ckpt16"
}

train_one() {
  local lr="$1" kl="$2" gpu="$3"
  local arm mid ckpt_root final arm_log
  arm="$(arm_id "$lr" "$kl")"
  mid="${arm}_ckpt16"
  ckpt_root="$WEIGHT_ROOT/llamafactory_qwen35_9b_lora_p5_${arm}"
  final="$ckpt_root/final"
  arm_log="$LOGDIR/s1b_rg_train_${arm}_${TS}.log"
  mkdir -p "$ckpt_root"

  if [[ -f "$(ckpt_path "$arm")/adapter_model.safetensors" ]]; then
    log "SKIP train $mid (ckpt16 exists)"
    return 0
  fi

  log "TRAIN $mid gpu=$gpu lr=$lr kl=$kl log=$arm_log"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON_TRAIN" \
    examples/benchmark/train_p5_reinforce.py \
    --buffer-root "$BUF" \
    --init-adapter "$INIT" \
    --output-dir "$final" \
    --dataset-order-file "$ORDER" \
    --strict-dataset-order \
    --advantage-mode per_dataset \
    --dataset-sequential \
    --checkpoint-at 16 \
    --checkpoint-root "$ckpt_root" \
    --device cuda:0 \
    --lr "$lr" --kl-beta "$kl" --grad-accum 8 --max-grad-norm 1.0 --seed "$SEED" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    --wandb-run-name "$mid" \
    --wandb-group "s1b_rg_dbest_hp_sweep" \
    --wandb-tags "reinforce,s1b,dpo_rationale_G,dbest,light16,hp_sweep" \
    >"$arm_log" 2>&1
  local rc=$?
  log "TRAIN $mid finished rc=$rc"
  return "$rc"
}

write_manifest() {
  local out="$1"
  local lr kl arm mid adapter
  local rows="["
  local first=1
  for lr in "${LRS[@]}"; do
    for kl in "${KLS[@]}"; do
      arm="$(arm_id "$lr" "$kl")"
      mid="${arm}_ckpt16"
      adapter="$(ckpt_path "$arm")"
      if [[ ! -f "$adapter/adapter_model.safetensors" ]]; then
        log "ERROR missing adapter $adapter"
        return 1
      fi
      if [[ $first -eq 1 ]]; then first=0; else rows+=","; fi
      rows+=$(printf '\n  {"id":"%s","adapter":"%s","resp":"json_only","lr":"%s","kl_beta":"%s"}' \
        "$mid" "$adapter" "$lr" "$kl")
    done
  done
  rows+=$'\n]'
  printf '%s\n' "$rows" >"$out"
  log "wrote manifest ($(grep -c '"id"' "$out") models) -> $out"
}

run_train_queue() {
  log "=== TRAIN queue: ${#LRS[@]} lrs × ${#KLS[@]} kls = $((${#LRS[@]} * ${#KLS[@]})) on ${NUM_GPUS} GPUs ==="
  declare -a PIDS=() GPUS_USED=() ARMS=()
  local lr kl gpu i

  for lr in "${LRS[@]}"; do
    for kl in "${KLS[@]}"; do
      # find free GPU slot (one train per GPU)
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
            train_one "$lr" "$kl" "$gpu" &
            PIDS+=($!)
            GPUS_USED+=("$gpu")
            ARMS+=("$(arm_id "$lr" "$kl")")
            sleep 2
            break 2
          fi
        done
        # all GPUs busy — wait for any
        wait -n 2>/dev/null || true
        # prune dead pids
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
  log "=== EVAL algo40 from_manifest ==="
  NUM_GPUS="$NUM_GPUS" JOBS_PER_GPU="$JOBS_PER_GPU" PACK_MIN_FREE_MB="$PACK_MIN_FREE_MB" \
    STAGE=from_manifest MANIFEST_JSON="$MANIFEST" HYBRID=H0 PROMPT_VARIANT=P5 \
    LLM_OPTIMIZERS=ptllm_tpe_qwen SEED="$SEED" EVAL_RESUME="$EVAL_RESUME" TIME_LIMIT=10800 \
    bash examples/benchmark/run_mltest20_algo40_eval.sh 2>&1 | tee -a "$PIPE"
  log "regen tables"
  python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm 2>&1 | tee -a "$PIPE"
  python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm --seed1-only 2>&1 | tee -a "$PIPE"
  log "=== EVAL done ==="
}

log "=== s1b_rg_dbest HP sweep START ts=$TS ==="
log "init=$INIT buffer=$BUF order=$ORDER"
log "wandb=$WANDB_ENTITY/$WANDB_PROJECT group=s1b_rg_dbest_hp_sweep"

if [[ "$EVAL_ONLY" != "1" ]]; then
  [[ -f "$INIT/adapter_model.safetensors" ]] || { log "ERROR missing init $INIT"; exit 1; }
  [[ -d "$BUF" ]] || { log "ERROR missing buffer $BUF"; exit 1; }
  run_train_queue
fi

if [[ "$TRAIN_ONLY" != "1" ]]; then
  run_eval
fi

log "=== s1b_rg_dbest HP sweep ALL DONE ==="
#!/usr/bin/env bash
# RL HP sweep on s1b dpo_rationale_G using the same dbest + light16 setup as
# s1a_b_rw_dbest_ckpt16, then algo40 eval (P5/H0 ptllm_tpe_qwen).
#
# Locked:
#   init:   s1b_ep7p5_dpo_rationale_G
#   buffer: B_rw_dbest_light16
#   order:  light_16.json (dsseq, per_dataset)
#   ckpt:   16
#   grad_accum=8, max_grad_norm=1.0, seed=1
#
# Sweep grid (9 runs):
#   lr      ∈ {5e-7, 1e-6, 2e-6}
#   kl_beta ∈ {0.02, 0.05, 0.1}
#
# Usage:
#   bash examples/benchmark/run_s1b_rg_dbest_hp_sweep.sh
#   TRAIN_ONLY=1 bash ...   # skip eval
#   EVAL_ONLY=1 bash ...    # skip train, eval existing ckpts
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

INIT="${INIT:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_rationale_G}"
BUF="${BUF:-benchmark_data/rl_rollouts/B_rw_dbest_light16}"
ORDER="${ORDER:-benchmark_data/rl_task_selection/light_16.json}"
WEIGHT_ROOT="${WEIGHT_ROOT:-/root/data/cys/blockdata_archive/model-weights}"
LOGDIR="${LOGDIR:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
PYTHON_TRAIN="${PYTHON_TRAIN:-/root/miniconda3/envs/llamafactory_qwen35/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
PIPE="$LOGDIR/s1b_rg_dbest_hp_sweep_${TS}.log"
MANIFEST="${MANIFEST:-benchmark_data/rl_task_selection/s1b_rg_dbest_hp_sweep_manifest.json}"
WANDB_PROJECT="${WANDB_PROJECT:-llmoptimizer}"
WANDB_ENTITY="${WANDB_ENTITY:-cys36-nanyang-technological-university-singapore}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-15000}"
EVAL_RESUME="${EVAL_RESUME:-1}"
SEED="${SEED:-1}"
TRAIN_ONLY="${TRAIN_ONLY:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"

LRS=(5e-7 1e-6 2e-6)
KLS=(0.02 0.05 0.1)

mkdir -p "$LOGDIR"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

tag_lr() {
  case "$1" in
    5e-7|5.0e-7|0.0000005) echo "5e7" ;;
    1e-6|1.0e-6|0.000001) echo "1e6" ;;
    2e-6|2.0e-6|0.000002) echo "2e6" ;;
    *) echo "$1" | tr '.' 'p' | tr -d '-' ;;
  esac
}

tag_kl() {
  case "$1" in
    0.02) echo "02" ;;
    0.05) echo "05" ;;
    0.1|0.10) echo "10" ;;
    *) echo "$1" | tr -d '.' ;;
  esac
}

arm_id() {
  local lr="$1" kl="$2"
  echo "s1b_rg_dbest_lr$(tag_lr "$lr")_kl$(tag_kl "$kl")"
}

ckpt_path() {
  local arm="$1"
  echo "$WEIGHT_ROOT/llamafactory_qwen35_9b_lora_p5_${arm}/ckpt16"
}

train_one() {
  local lr="$1" kl="$2" gpu="$3"
  local arm mid ckpt_root final arm_log
  arm="$(arm_id "$lr" "$kl")"
  mid="${arm}_ckpt16"
  ckpt_root="$WEIGHT_ROOT/llamafactory_qwen35_9b_lora_p5_${arm}"
  final="$ckpt_root/final"
  arm_log="$LOGDIR/s1b_rg_train_${arm}_${TS}.log"
  mkdir -p "$ckpt_root"

  if [[ -f "$(ckpt_path "$arm")/adapter_model.safetensors" ]]; then
    log "SKIP train $mid (ckpt16 exists)"
    return 0
  fi

  log "TRAIN $mid gpu=$gpu lr=$lr kl=$kl log=$arm_log"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON_TRAIN" \
    examples/benchmark/train_p5_reinforce.py \
    --buffer-root "$BUF" \
    --init-adapter "$INIT" \
    --output-dir "$final" \
    --dataset-order-file "$ORDER" \
    --strict-dataset-order \
    --advantage-mode per_dataset \
    --dataset-sequential \
    --checkpoint-at 16 \
    --checkpoint-root "$ckpt_root" \
    --device cuda:0 \
    --lr "$lr" --kl-beta "$kl" --grad-accum 8 --max-grad-norm 1.0 --seed "$SEED" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    --wandb-run-name "$mid" \
    --wandb-group "s1b_rg_dbest_hp_sweep" \
    --wandb-tags "reinforce,s1b,dpo_rationale_G,dbest,light16,hp_sweep" \
    >"$arm_log" 2>&1
  local rc=$?
  log "TRAIN $mid finished rc=$rc"
  return "$rc"
}

write_manifest() {
  local out="$1"
  local lr kl arm mid adapter
  local rows="["
  local first=1
  for lr in "${LRS[@]}"; do
    for kl in "${KLS[@]}"; do
      arm="$(arm_id "$lr" "$kl")"
      mid="${arm}_ckpt16"
      adapter="$(ckpt_path "$arm")"
      if [[ ! -f "$adapter/adapter_model.safetensors" ]]; then
        log "ERROR missing adapter $adapter"
        return 1
      fi
      if [[ $first -eq 1 ]]; then first=0; else rows+=","; fi
      rows+=$(printf '\n  {"id":"%s","adapter":"%s","resp":"json_only","lr":"%s","kl_beta":"%s"}' \
        "$mid" "$adapter" "$lr" "$kl")
    done
  done
  rows+=$'\n]'
  printf '%s\n' "$rows" >"$out"
  log "wrote manifest ($(grep -c '"id"' "$out") models) -> $out"
}

run_train_queue() {
  log "=== TRAIN queue: ${#LRS[@]} lrs × ${#KLS[@]} kls = $((${#LRS[@]} * ${#KLS[@]})) on ${NUM_GPUS} GPUs ==="
  declare -a PIDS=() GPUS_USED=() ARMS=()
  local lr kl gpu i

  for lr in "${LRS[@]}"; do
    for kl in "${KLS[@]}"; do
      # find free GPU slot (one train per GPU)
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
            train_one "$lr" "$kl" "$gpu" &
            PIDS+=($!)
            GPUS_USED+=("$gpu")
            ARMS+=("$(arm_id "$lr" "$kl")")
            sleep 2
            break 2
          fi
        done
        # all GPUs busy — wait for any
        wait -n 2>/dev/null || true
        # prune dead pids
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
  log "=== EVAL algo40 from_manifest ==="
  NUM_GPUS="$NUM_GPUS" JOBS_PER_GPU="$JOBS_PER_GPU" PACK_MIN_FREE_MB="$PACK_MIN_FREE_MB" \
    STAGE=from_manifest MANIFEST_JSON="$MANIFEST" HYBRID=H0 PROMPT_VARIANT=P5 \
    LLM_OPTIMIZERS=ptllm_tpe_qwen SEED="$SEED" EVAL_RESUME="$EVAL_RESUME" TIME_LIMIT=10800 \
    bash examples/benchmark/run_mltest20_algo40_eval.sh 2>&1 | tee -a "$PIPE"
  log "regen tables"
  python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm 2>&1 | tee -a "$PIPE"
  python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm --seed1-only 2>&1 | tee -a "$PIPE"
  log "=== EVAL done ==="
}

log "=== s1b_rg_dbest HP sweep START ts=$TS ==="
log "init=$INIT buffer=$BUF order=$ORDER"
log "wandb=$WANDB_ENTITY/$WANDB_PROJECT group=s1b_rg_dbest_hp_sweep"

if [[ "$EVAL_ONLY" != "1" ]]; then
  [[ -f "$INIT/adapter_model.safetensors" ]] || { log "ERROR missing init $INIT"; exit 1; }
  [[ -d "$BUF" ]] || { log "ERROR missing buffer $BUF"; exit 1; }
  run_train_queue
fi

if [[ "$TRAIN_ONLY" != "1" ]]; then
  run_eval
fi

log "=== s1b_rg_dbest HP sweep ALL DONE ==="
#!/usr/bin/env bash
# RL HP sweep on s1b dpo_rationale_G using the same dbest + light16 setup as
# s1a_b_rw_dbest_ckpt16, then algo40 eval (P5/H0 ptllm_tpe_qwen).
#
# Locked:
#   init:   s1b_ep7p5_dpo_rationale_G
#   buffer: B_rw_dbest_light16
#   order:  light_16.json (dsseq, per_dataset)
#   ckpt:   16
#   grad_accum=8, max_grad_norm=1.0, seed=1
#
# Sweep grid (9 runs):
#   lr      ∈ {5e-7, 1e-6, 2e-6}
#   kl_beta ∈ {0.02, 0.05, 0.1}
#
# Usage:
#   bash examples/benchmark/run_s1b_rg_dbest_hp_sweep.sh
#   TRAIN_ONLY=1 bash ...   # skip eval
#   EVAL_ONLY=1 bash ...    # skip train, eval existing ckpts
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

INIT="${INIT:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_rationale_G}"
BUF="${BUF:-benchmark_data/rl_rollouts/B_rw_dbest_light16}"
ORDER="${ORDER:-benchmark_data/rl_task_selection/light_16.json}"
WEIGHT_ROOT="${WEIGHT_ROOT:-/root/data/cys/blockdata_archive/model-weights}"
LOGDIR="${LOGDIR:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
PYTHON_TRAIN="${PYTHON_TRAIN:-/root/miniconda3/envs/llamafactory_qwen35/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
PIPE="$LOGDIR/s1b_rg_dbest_hp_sweep_${TS}.log"
MANIFEST="${MANIFEST:-benchmark_data/rl_task_selection/s1b_rg_dbest_hp_sweep_manifest.json}"
WANDB_PROJECT="${WANDB_PROJECT:-llmoptimizer}"
WANDB_ENTITY="${WANDB_ENTITY:-cys36-nanyang-technological-university-singapore}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-15000}"
EVAL_RESUME="${EVAL_RESUME:-1}"
SEED="${SEED:-1}"
TRAIN_ONLY="${TRAIN_ONLY:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"

LRS=(5e-7 1e-6 2e-6)
KLS=(0.02 0.05 0.1)

mkdir -p "$LOGDIR"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

tag_lr() {
  case "$1" in
    5e-7|5.0e-7|0.0000005) echo "5e7" ;;
    1e-6|1.0e-6|0.000001) echo "1e6" ;;
    2e-6|2.0e-6|0.000002) echo "2e6" ;;
    *) echo "$1" | tr '.' 'p' | tr -d '-' ;;
  esac
}

tag_kl() {
  case "$1" in
    0.02) echo "02" ;;
    0.05) echo "05" ;;
    0.1|0.10) echo "10" ;;
    *) echo "$1" | tr -d '.' ;;
  esac
}

arm_id() {
  local lr="$1" kl="$2"
  echo "s1b_rg_dbest_lr$(tag_lr "$lr")_kl$(tag_kl "$kl")"
}

ckpt_path() {
  local arm="$1"
  echo "$WEIGHT_ROOT/llamafactory_qwen35_9b_lora_p5_${arm}/ckpt16"
}

train_one() {
  local lr="$1" kl="$2" gpu="$3"
  local arm mid ckpt_root final arm_log
  arm="$(arm_id "$lr" "$kl")"
  mid="${arm}_ckpt16"
  ckpt_root="$WEIGHT_ROOT/llamafactory_qwen35_9b_lora_p5_${arm}"
  final="$ckpt_root/final"
  arm_log="$LOGDIR/s1b_rg_train_${arm}_${TS}.log"
  mkdir -p "$ckpt_root"

  if [[ -f "$(ckpt_path "$arm")/adapter_model.safetensors" ]]; then
    log "SKIP train $mid (ckpt16 exists)"
    return 0
  fi

  log "TRAIN $mid gpu=$gpu lr=$lr kl=$kl log=$arm_log"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON_TRAIN" \
    examples/benchmark/train_p5_reinforce.py \
    --buffer-root "$BUF" \
    --init-adapter "$INIT" \
    --output-dir "$final" \
    --dataset-order-file "$ORDER" \
    --strict-dataset-order \
    --advantage-mode per_dataset \
    --dataset-sequential \
    --checkpoint-at 16 \
    --checkpoint-root "$ckpt_root" \
    --device cuda:0 \
    --lr "$lr" --kl-beta "$kl" --grad-accum 8 --max-grad-norm 1.0 --seed "$SEED" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    --wandb-run-name "$mid" \
    --wandb-group "s1b_rg_dbest_hp_sweep" \
    --wandb-tags "reinforce,s1b,dpo_rationale_G,dbest,light16,hp_sweep" \
    >"$arm_log" 2>&1
  local rc=$?
  log "TRAIN $mid finished rc=$rc"
  return "$rc"
}

write_manifest() {
  local out="$1"
  local lr kl arm mid adapter
  local rows="["
  local first=1
  for lr in "${LRS[@]}"; do
    for kl in "${KLS[@]}"; do
      arm="$(arm_id "$lr" "$kl")"
      mid="${arm}_ckpt16"
      adapter="$(ckpt_path "$arm")"
      if [[ ! -f "$adapter/adapter_model.safetensors" ]]; then
        log "ERROR missing adapter $adapter"
        return 1
      fi
      if [[ $first -eq 1 ]]; then first=0; else rows+=","; fi
      rows+=$(printf '\n  {"id":"%s","adapter":"%s","resp":"json_only","lr":"%s","kl_beta":"%s"}' \
        "$mid" "$adapter" "$lr" "$kl")
    done
  done
  rows+=$'\n]'
  printf '%s\n' "$rows" >"$out"
  log "wrote manifest ($(grep -c '"id"' "$out") models) -> $out"
}

run_train_queue() {
  log "=== TRAIN queue: ${#LRS[@]} lrs × ${#KLS[@]} kls = $((${#LRS[@]} * ${#KLS[@]})) on ${NUM_GPUS} GPUs ==="
  declare -a PIDS=() GPUS_USED=() ARMS=()
  local lr kl gpu i

  for lr in "${LRS[@]}"; do
    for kl in "${KLS[@]}"; do
      # find free GPU slot (one train per GPU)
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
            train_one "$lr" "$kl" "$gpu" &
            PIDS+=($!)
            GPUS_USED+=("$gpu")
            ARMS+=("$(arm_id "$lr" "$kl")")
            sleep 2
            break 2
          fi
        done
        # all GPUs busy — wait for any
        wait -n 2>/dev/null || true
        # prune dead pids
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
  log "=== EVAL algo40 from_manifest ==="
  NUM_GPUS="$NUM_GPUS" JOBS_PER_GPU="$JOBS_PER_GPU" PACK_MIN_FREE_MB="$PACK_MIN_FREE_MB" \
    STAGE=from_manifest MANIFEST_JSON="$MANIFEST" HYBRID=H0 PROMPT_VARIANT=P5 \
    LLM_OPTIMIZERS=ptllm_tpe_qwen SEED="$SEED" EVAL_RESUME="$EVAL_RESUME" TIME_LIMIT=10800 \
    bash examples/benchmark/run_mltest20_algo40_eval.sh 2>&1 | tee -a "$PIPE"
  log "regen tables"
  python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm 2>&1 | tee -a "$PIPE"
  python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm --seed1-only 2>&1 | tee -a "$PIPE"
  log "=== EVAL done ==="
}

log "=== s1b_rg_dbest HP sweep START ts=$TS ==="
log "init=$INIT buffer=$BUF order=$ORDER"
log "wandb=$WANDB_ENTITY/$WANDB_PROJECT group=s1b_rg_dbest_hp_sweep"

if [[ "$EVAL_ONLY" != "1" ]]; then
  [[ -f "$INIT/adapter_model.safetensors" ]] || { log "ERROR missing init $INIT"; exit 1; }
  [[ -d "$BUF" ]] || { log "ERROR missing buffer $BUF"; exit 1; }
  run_train_queue
fi

if [[ "$TRAIN_ONLY" != "1" ]]; then
  run_eval
fi

log "=== s1b_rg_dbest HP sweep ALL DONE ==="
