#!/usr/bin/env bash
# Local HP refine around drop20 top arms:
#   Track A: s2_G ORPO   around (lr~1e-5, β~0.3) on p5_s2_dpo
#   Track B: rationale ORPO around (lr~1e-5, β~0.1) on p5_s2_dpo_rationale
#   Track C: REINFORCE dbest around (lr~1e-6, kl~0.05) from s1a_ep7p5
#
# Usage:
#   PHASE=all bash examples/benchmark/run_local_refine_sweeps.sh
#   PHASE=A|B|C bash ...
#   TRAIN_ONLY=1 / EVAL_ONLY=1 supported
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

PHASE="${PHASE:-all}"
INIT_S1B="${INIT_S1B:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5}"
INIT_S1A="${INIT_S1A:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5}"
WEIGHT_ROOT="${WEIGHT_ROOT:-/root/data/cys/blockdata_archive/model-weights}"
LOGDIR="${LOGDIR:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
LF="${LLAMAFACTORY_CLI:-/root/miniconda3/envs/llamafactory_qwen35/bin/llamafactory-cli}"
LF_ROOT="${LF_ROOT:-/root/LLMOptimizer/LlamaFactory}"
PYTHON_TRAIN="${PYTHON_TRAIN:-/root/miniconda3/envs/llamafactory_qwen35/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
PIPE="$LOGDIR/local_refine_sweeps_${TS}.log"
WANDB_PROJECT="${WANDB_PROJECT:-llmoptimizer}"
WANDB_ENTITY="${WANDB_ENTITY:-cys36-nanyang-technological-university-singapore}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-15000}"
EVAL_RESUME="${EVAL_RESUME:-1}"
SEED="${SEED:-1}"
TRAIN_ONLY="${TRAIN_ONLY:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"
ORDER_L16="${ORDER_L16:-benchmark_data/rl_task_selection/light_16.json}"
BUF_DBEST="${BUF_DBEST:-benchmark_data/rl_rollouts/B_rw_dbest_light16}"

# shellcheck source=examples/benchmark/orpo_resume_helpers.sh
source "$REPO/examples/benchmark/orpo_resume_helpers.sh"

mkdir -p "$LOGDIR"
export PATH="/root/miniconda3/envs/llamafactory_qwen35/bin:$PATH"
export WANDB_PROJECT WANDB_ENTITY
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

# ---------- tagging ----------
tag_lr() {
  case "$1" in
    5e-7|5.0e-7) echo "5e7" ;;
    7.5e-7|7.50e-7) echo "7p5e7" ;;
    1e-6|1.0e-6) echo "1e6" ;;
    1.5e-6|1.50e-6) echo "1p5e6" ;;
    5e-6|5.0e-6) echo "5e6" ;;
    7.5e-6|7.50e-6) echo "7p5e6" ;;
    1e-5|1.0e-5) echo "1e5" ;;
    1.5e-5|1.50e-5) echo "1p5e5" ;;
    *) echo "$1" | tr -d '-' | tr '.' 'p' ;;
  esac
}
tag_beta() {
  case "$1" in
    0.07) echo "07" ;;
    0.1|0.10) echo "10" ;;
    0.15) echo "15" ;;
    0.2|0.20) echo "20" ;;
    0.25) echo "25" ;;
    0.3|0.30) echo "30" ;;
    0.35) echo "35" ;;
    0.4|0.40) echo "40" ;;
    *) echo "$1" | tr -d '.' ;;
  esac
}
tag_kl() {
  case "$1" in
    0.03) echo "03" ;;
    0.05) echo "05" ;;
    0.07) echo "07" ;;
    *) echo "$1" | tr -d '.' ;;
  esac
}
lr_sci() {
  case "$1" in
    5e-7) echo "5.0e-7" ;;
    7.5e-7) echo "7.5e-7" ;;
    1e-6) echo "1.0e-6" ;;
    1.5e-6) echo "1.5e-6" ;;
    5e-6) echo "5.0e-6" ;;
    7.5e-6) echo "7.5e-6" ;;
    1e-5) echo "1.0e-5" ;;
    1.5e-5) echo "1.5e-5" ;;
    *) echo "$1" ;;
  esac
}

# ---------- ORPO helpers ----------
orpo_write_yaml() {
  local prefix="$1" dataset="$2" lr="$3" beta="$4" cfg_dir="$5"
  local arm out_dir yaml
  arm="${prefix}_lr$(tag_lr "$lr")_b$(tag_beta "$beta")"
  out_dir="$WEIGHT_ROOT/llamafactory_qwen35_9b_lora_p5_${arm}"
  yaml="$cfg_dir/${arm}.yaml"
  mkdir -p "$cfg_dir"
  cat >"$yaml" <<EOF
### model
# Local refine ORPO: ${arm}
model_name_or_path: /root/eb-public/huggingface-models/Qwen/Qwen3.5-9B
adapter_name_or_path: ${INIT_S1B}
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
dataset: ${dataset}
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
learning_rate: $(lr_sci "$lr")
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
gradient_checkpointing: true
ddp_timeout: 180000000
resume_from_checkpoint: null
seed: 42
EOF
  echo "$yaml|$arm|$out_dir"
}

orpo_train_one() {
  local yaml="$1" arm="$2" out_dir="$3" gpu="$4"
  local tlog="$LOGDIR/local_refine_train_${arm}_${TS}.log" mode ckpt
  if [[ -f "$out_dir/adapter_model.safetensors" ]]; then
    log "SKIP train $arm (exists)"
    return 0
  fi
  mode="$(orpo_patch_yaml_resume "$yaml" "$out_dir")"
  if [[ "$mode" == "resume" ]]; then
    ckpt="$(orpo_latest_checkpoint "$out_dir")"
    log "RESUME train $arm from $ckpt"
  fi
  log "TRAIN $arm gpu=$gpu mode=$mode log=$tlog"
  cd "$LF_ROOT"
  CUDA_VISIBLE_DEVICES="$gpu" DISABLE_VERSION_CHECK=1 \
    WANDB_PROJECT="$WANDB_PROJECT" WANDB_ENTITY="$WANDB_ENTITY" \
    "$LF" train "$yaml" >"$tlog" 2>&1
  local rc=$?
  log "TRAIN $arm finished rc=$rc"
  return "$rc"
}

# Queue of "yaml|arm|outdir" lines
orpo_run_queue() {
  local -a jobs=("$@")
  log "=== ORPO queue: ${#jobs[@]} jobs on ${NUM_GPUS} GPUs ==="
  declare -a PIDS=() GPUS_USED=() ARMS=()
  local job gpu i yaml arm out_dir

  for job in "${jobs[@]}"; do
    IFS='|' read -r yaml arm out_dir <<<"$job"
    while true; do
      for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
        local busy=0
        for i in "${!PIDS[@]}"; do
          if [[ "${GPUS_USED[$i]}" == "$gpu" ]] && kill -0 "${PIDS[$i]}" 2>/dev/null; then
            busy=1; break
          fi
        done
        if [[ $busy -eq 0 ]]; then
          orpo_train_one "$yaml" "$arm" "$out_dir" "$gpu" &
          PIDS+=($!)
          GPUS_USED+=("$gpu")
          ARMS+=("$arm")
          sleep 2
          break 2
        fi
      done
      wait -n 2>/dev/null || true
      local new_pids=() new_gpus=() new_arms=()
      for i in "${!PIDS[@]}"; do
        if kill -0 "${PIDS[$i]}" 2>/dev/null; then
          new_pids+=("${PIDS[$i]}"); new_gpus+=("${GPUS_USED[$i]}"); new_arms+=("${ARMS[$i]}")
        else
          wait "${PIDS[$i]}" || log "WARN train failed arm=${ARMS[$i]} rc=$?"
        fi
      done
      PIDS=("${new_pids[@]:-}")
      GPUS_USED=("${new_gpus[@]:-}")
      ARMS=("${new_arms[@]:-}")
    done
  done
  wait || true
  log "=== ORPO queue done ==="
}

write_orpo_manifest() {
  local out="$1"; shift
  local -a arms=("$@")
  local arm adapter first=1 rows="["
  for arm in "${arms[@]}"; do
    adapter="$WEIGHT_ROOT/llamafactory_qwen35_9b_lora_p5_${arm}"
    if [[ ! -f "$adapter/adapter_model.safetensors" ]]; then
      log "ERROR missing $adapter"; return 1
    fi
    if [[ $first -eq 1 ]]; then first=0; else rows+=","; fi
    rows+=$(printf '\n  {"id":"%s","adapter":"%s","resp":"rationale_json"}' "$arm" "$adapter")
  done
  rows+=$'\n]'
  printf '%s\n' "$rows" >"$out"
  log "wrote manifest $(grep -c '"id"' "$out") -> $out"
}

eval_manifest() {
  local mf="$1"
  log "=== EVAL $mf ==="
  NUM_GPUS="$NUM_GPUS" JOBS_PER_GPU="$JOBS_PER_GPU" PACK_MIN_FREE_MB="$PACK_MIN_FREE_MB" \
    STAGE=from_manifest MANIFEST_JSON="$mf" HYBRID=H0 PROMPT_VARIANT=P5 \
    LLM_OPTIMIZERS=ptllm_tpe_qwen SEED="$SEED" EVAL_RESUME="$EVAL_RESUME" TIME_LIMIT=10800 \
    bash examples/benchmark/run_mltest20_algo40_eval.sh 2>&1 | tee -a "$PIPE"
  python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm 2>&1 | tee -a "$PIPE"
  python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm --seed1-only 2>&1 | tee -a "$PIPE"
  # re-annotate standard drop-20% seed1 table
  "$PYTHON_TRAIN" - <<'PY' 2>&1 | tee -a "$PIPE" || true
import csv, sys
from pathlib import Path
sys.path.insert(0, "examples/benchmark")
from plot_mltest20_algo40_avg_rank import method_meaning
avg = Path("benchmark_data/real_hpo_v2_mltest20_algo40/analysis/avg_rank")
fields = ["rank","display_name","method_meaning","variant","family","datasets",
          "avg_rank","avg_rank_cls","avg_rank_rgs","mean_cls_norm","mean_rgs_norm","mean_balanced_norm"]
for name in ["mean_norm_table_pooled_drop20pct_seed1.csv"]:
    p = avg/name
    if not p.exists(): continue
    rows=list(csv.DictReader(p.open()))
    with p.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows:
            r["method_meaning"]=method_meaning(r["variant"], r["display_name"])
            w.writerow({k:r.get(k,"") for k in fields})
    print(f"annotated {name} ({len(rows)})")
PY
  log "=== EVAL done ==="
}

# ---------- Track A ----------
run_track_A() {
  log "===== TRACK A: s2_G ORPO local ====="
  local cfg="/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/configs/ml/s1b_s2g_orpo_local"
  local lrs=(5e-6 7.5e-6 1e-5 1.5e-5)
  local betas=(0.25 0.3 0.35 0.4)
  local -a jobs=() arms=()
  local lr beta meta arm
  for lr in "${lrs[@]}"; do
    for beta in "${betas[@]}"; do
      # skip coarse duplicates already eval'd
      if { [[ "$lr" == "5e-6" ]] && [[ "$beta" == "0.3" ]]; } || \
         { [[ "$lr" == "1e-5" ]] && [[ "$beta" == "0.3" ]]; }; then
        arm="s1b_s2g_orpo_lr$(tag_lr "$lr")_b$(tag_beta "$beta")"
        log "SKIP schedule (already have) $arm"
        arms+=("$arm")
        continue
      fi
      meta="$(orpo_write_yaml s1b_s2g_orpo p5_s2_dpo "$lr" "$beta" "$cfg")"
      jobs+=("$meta")
      IFS='|' read -r _ arm _ <<<"$meta"
      arms+=("$arm")
    done
  done
  if [[ "$EVAL_ONLY" != "1" ]]; then
    orpo_run_queue "${jobs[@]}"
  fi
  if [[ "$TRAIN_ONLY" != "1" ]]; then
    local mf="benchmark_data/rl_task_selection/s1b_s2g_orpo_local_manifest.json"
    # eval only newly trained + keep champs? eval all arms in this local grid that exist
    local -a present=()
    for arm in "${arms[@]}"; do
      [[ -f "$WEIGHT_ROOT/llamafactory_qwen35_9b_lora_p5_${arm}/adapter_model.safetensors" ]] && present+=("$arm")
    done
    write_orpo_manifest "$mf" "${present[@]}"
    eval_manifest "$mf"
  fi
}

# ---------- Track B ----------
run_track_B() {
  log "===== TRACK B: rationale ORPO local ====="
  local cfg="/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/configs/ml/s1b_rg_orpo_local"
  local lrs=(7.5e-6 1e-5 1.5e-5)
  local betas=(0.07 0.1 0.15 0.2)
  local -a jobs=() arms=()
  local lr beta meta arm
  for lr in "${lrs[@]}"; do
    for beta in "${betas[@]}"; do
      if [[ "$lr" == "1e-5" ]] && [[ "$beta" == "0.1" ]]; then
        arm="s1b_rg_orpo_lr$(tag_lr "$lr")_b$(tag_beta "$beta")"
        log "SKIP schedule (already have) $arm"
        arms+=("$arm")
        continue
      fi
      meta="$(orpo_write_yaml s1b_rg_orpo p5_s2_dpo_rationale "$lr" "$beta" "$cfg")"
      jobs+=("$meta")
      IFS='|' read -r _ arm _ <<<"$meta"
      arms+=("$arm")
    done
  done
  if [[ "$EVAL_ONLY" != "1" ]]; then
    orpo_run_queue "${jobs[@]}"
  fi
  if [[ "$TRAIN_ONLY" != "1" ]]; then
    local mf="benchmark_data/rl_task_selection/s1b_rg_orpo_local_manifest.json"
    local -a present=()
    for arm in "${arms[@]}"; do
      [[ -f "$WEIGHT_ROOT/llamafactory_qwen35_9b_lora_p5_${arm}/adapter_model.safetensors" ]] && present+=("$arm")
    done
    write_orpo_manifest "$mf" "${present[@]}"
    eval_manifest "$mf"
  fi
}

# ---------- Track C REINFORCE ----------
rl_train_one() {
  local lr="$1" kl="$2" gpu="$3"
  local arm mid ckpt_root final tlog
  arm="s1a_b_rw_dbest_lr$(tag_lr "$lr")_kl$(tag_kl "$kl")"
  mid="${arm}_ckpt16"
  ckpt_root="$WEIGHT_ROOT/llamafactory_qwen35_9b_lora_p5_${arm}"
  final="$ckpt_root/final"
  tlog="$LOGDIR/local_refine_train_${arm}_${TS}.log"
  if [[ -f "$ckpt_root/ckpt16/adapter_model.safetensors" ]]; then
    log "SKIP train $mid (exists)"; return 0
  fi
  mkdir -p "$ckpt_root"
  log "TRAIN $mid gpu=$gpu lr=$lr kl=$kl"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON_TRAIN" \
    examples/benchmark/train_p5_reinforce.py \
    --buffer-root "$BUF_DBEST" \
    --init-adapter "$INIT_S1A" \
    --output-dir "$final" \
    --dataset-order-file "$ORDER_L16" \
    --strict-dataset-order \
    --advantage-mode per_dataset \
    --dataset-sequential \
    --checkpoint-at 16 \
    --checkpoint-root "$ckpt_root" \
    --device cuda:0 \
    --lr "$lr" --kl-beta "$kl" --grad-accum 8 --max-grad-norm 1.0 --seed 1 \
    --wandb-project "$WANDB_PROJECT" --wandb-entity "$WANDB_ENTITY" \
    --wandb-run-name "$mid" --wandb-group "s1a_dbest_local_refine" \
    --wandb-tags "reinforce,s1a,dbest,light16,local_refine" \
    >"$tlog" 2>&1
  local rc=$?
  log "TRAIN $mid finished rc=$rc"
  return "$rc"
}

run_track_C() {
  log "===== TRACK C: REINFORCE dbest local ====="
  local lrs=(5e-7 7.5e-7 1e-6 1.5e-6)
  local kls=(0.03 0.05 0.07)
  declare -a PIDS=() GPUS_USED=() ARMS=()
  local lr kl gpu i arm

  if [[ "$EVAL_ONLY" != "1" ]]; then
    [[ -d "$BUF_DBEST" ]] || { log "ERROR missing $BUF_DBEST"; return 1; }
    for lr in "${lrs[@]}"; do
      for kl in "${kls[@]}"; do
        arm="s1a_b_rw_dbest_lr$(tag_lr "$lr")_kl$(tag_kl "$kl")"
        while true; do
          for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
            local busy=0
            for i in "${!PIDS[@]}"; do
              if [[ "${GPUS_USED[$i]}" == "$gpu" ]] && kill -0 "${PIDS[$i]}" 2>/dev/null; then
                busy=1; break
              fi
            done
            if [[ $busy -eq 0 ]]; then
              rl_train_one "$lr" "$kl" "$gpu" &
              PIDS+=($!); GPUS_USED+=("$gpu"); ARMS+=("$arm")
              sleep 2; break 2
            fi
          done
          wait -n 2>/dev/null || true
          local new_pids=() new_gpus=() new_arms=()
          for i in "${!PIDS[@]}"; do
            if kill -0 "${PIDS[$i]}" 2>/dev/null; then
              new_pids+=("${PIDS[$i]}"); new_gpus+=("${GPUS_USED[$i]}"); new_arms+=("${ARMS[$i]}")
            else
              wait "${PIDS[$i]}" || log "WARN RL failed ${ARMS[$i]}"
            fi
          done
          PIDS=("${new_pids[@]:-}"); GPUS_USED=("${new_gpus[@]:-}"); ARMS=("${new_arms[@]:-}")
        done
      done
    done
    wait || true
    log "=== RL queue done ==="
  fi

  if [[ "$TRAIN_ONLY" != "1" ]]; then
    local mf="benchmark_data/rl_task_selection/s1a_dbest_local_manifest.json"
    local rows="[" first=1
    for lr in "${lrs[@]}"; do
      for kl in "${kls[@]}"; do
        arm="s1a_b_rw_dbest_lr$(tag_lr "$lr")_kl$(tag_kl "$kl")"
        mid="${arm}_ckpt16"
        adapter="$WEIGHT_ROOT/llamafactory_qwen35_9b_lora_p5_${arm}/ckpt16"
        if [[ ! -f "$adapter/adapter_model.safetensors" ]]; then
          log "ERROR missing $adapter"; return 1
        fi
        if [[ $first -eq 1 ]]; then first=0; else rows+=","; fi
        rows+=$(printf '\n  {"id":"%s","adapter":"%s","resp":"json_only"}' "$mid" "$adapter")
      done
    done
    # also include original champ for completeness in this eval? optional - skip, already on board
    rows+=$'\n]'
    printf '%s\n' "$rows" >"$mf"
    log "wrote $mf"
    eval_manifest "$mf"
  fi
}

log "=== local refine START phase=$PHASE ts=$TS ==="
case "$PHASE" in
  A) run_track_A ;;
  B) run_track_B ;;
  C) run_track_C ;;
  all)
    run_track_A
    run_track_B
    run_track_C
    ;;
  *) log "unknown PHASE=$PHASE (A|B|C|all)"; exit 1 ;;
esac
log "=== local refine ALL DONE phase=$PHASE ==="
