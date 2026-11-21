#!/usr/bin/env bash
# Hard16 RL: Setup-B rollouts → dbest buffer → REINFORCE ckpts 8/12/16 → tasks_40_hard eval.
#
# Locked to best s1b dbest hyperparams (s1b_rg_dbest_lr5e7_kl05):
#   init: lr=5e-7, kl=0.05, grad_accum=8, max_grad_norm=1.0, per_dataset+sequential
#   eval: P5plusA5T50X50RS10/HPUCB_arm ptllm_tpe_qwen on tasks_40_hard (override via EVAL_PROMPT/EVAL_HYBRID)
#
# Usage:
#   bash examples/benchmark/run_hard16_rl_dbest_ckpt81216.sh
#   SKIP_ROLLOUTS=1 bash ...     # buffer already collected
#   TRAIN_ONLY=1 bash ...
#   EVAL_ONLY=1 bash ...
#   DRY_RUN=1 bash ...           # validation only
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

INIT="${INIT:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_rationale_G}"
ROLLOUT_ADAPTER="${ROLLOUT_ADAPTER:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5}"
TASKS_JSON="${TASKS_JSON:-benchmark_data/rl_task_selection/hard_16.json}"
EVAL_TASKS_JSON="${EVAL_TASKS_JSON:-benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40_hard.json}"
SRC_ROLL="${SRC_ROLL:-benchmark_data/rl_rollouts/B_prefix10to100_n4_T1p0_hard16}"
BUF="${BUF:-benchmark_data/rl_rollouts/B_rw_dbest_hard16}"
ORDER="${ORDER:-benchmark_data/rl_task_selection/hard_16.json}"
ARMS_FILE="${ARMS_FILE:-benchmark_data/rl_task_selection/reward_sweep_arms.json}"
WEIGHT_ROOT="${WEIGHT_ROOT:-/root/data/cys/blockdata_archive/model-weights}"
ARM_ID="${ARM_ID:-s1b_rg_dbest_hard16}"
CKPT_ROOT="${CKPT_ROOT:-$WEIGHT_ROOT/llamafactory_qwen35_9b_lora_p5_${ARM_ID}}"
MANIFEST="${MANIFEST:-benchmark_data/rl_task_selection/${ARM_ID}_ckpt81216_manifest.json}"
EVAL_OUTPUT_BASE="${EVAL_OUTPUT_BASE:-benchmark_data/real_hpo_v2_mltest20_algo40}"
LOGDIR="${LOGDIR:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
PYTHON_TRAIN="${PYTHON_TRAIN:-/root/miniconda3/envs/llamafactory_qwen35/bin/python}"
PYTHON_EVAL="${PYTHON_EVAL:-/root/miniconda3/envs/openbox_qwen_eval/bin/python}"
EVAL_ENV="${EVAL_ENV:-/root/miniconda3/envs/openbox_qwen_eval}"
LR="${LR:-5e-7}"
KL="${KL:-0.05}"
CKPTS="${CKPTS:-8,12,16}"
SEED="${SEED:-1}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
EVAL_RESUME="${EVAL_RESUME:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-llmoptimizer}"
WANDB_ENTITY="${WANDB_ENTITY:-cys36-nanyang-technological-university-singapore}"
SKIP_ROLLOUTS="${SKIP_ROLLOUTS:-0}"
TRAIN_ONLY="${TRAIN_ONLY:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"
EVAL_PROMPT="${EVAL_PROMPT:-P5plusA5T50X50RS10}"
EVAL_HYBRID="${EVAL_HYBRID:-HPUCB_arm}"
DRY_RUN="${DRY_RUN:-0}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
PIPE="$LOGDIR/${ARM_ID}_pipeline_${TS}.log"

mkdir -p "$LOGDIR"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

ensure_hard16() {
  if [[ ! -f "$TASKS_JSON" ]]; then
    log "generating $TASKS_JSON"
    python3 examples/benchmark/select_hard16_from_hard40.py --out "$TASKS_JSON"
  fi
  python3 - <<PY
import json
from pathlib import Path
tasks = json.loads(Path("$TASKS_JSON").read_text())
assert len(tasks) == 16, len(tasks)
print("ok hard16 tasks=", len(tasks))
PY
}

validate_prereqs() {
  log "=== validate prereqs ==="
  ensure_hard16
  [[ -f "$INIT/adapter_model.safetensors" ]] || { log "ERROR missing init $INIT"; exit 1; }
  [[ -f "$ROLLOUT_ADAPTER/adapter_model.safetensors" ]] || { log "ERROR missing rollout adapter $ROLLOUT_ADAPTER"; exit 1; }
  [[ -f "$ARMS_FILE" ]] || { log "ERROR missing arms file $ARMS_FILE"; exit 1; }
  [[ -f "$EVAL_TASKS_JSON" ]] || { log "ERROR missing eval tasks $EVAL_TASKS_JSON"; exit 1; }
  python3 examples/benchmark/select_hard16_from_hard40.py --dry-run
  log "prereqs ok init=$INIT lr=$LR kl=$KL ckpts=$CKPTS"
}

count_rollout_records() {
  python3 - <<PY
from pathlib import Path
root = Path("$SRC_ROLL")
n = sum(sum(1 for ln in p.open() if ln.strip()) for p in root.rglob("rollouts_*.jsonl")) if root.exists() else 0
print(n)
PY
}

run_rollouts() {
  log "=== Setup-B rollouts hard16 -> $SRC_ROLL ==="
  TASKS_JSON="$TASKS_JSON" OUT_ROOT="$SRC_ROLL" ADAPTER="$ROLLOUT_ADAPTER" \
    GPU_IDS="${GPU_IDS:-0,1,2,3}" JOBS_PER_GPU="${ROLLOUT_JOBS_PER_GPU:-1}" \
    bash examples/benchmark/run_setupB_prefix_rollouts.sh >>"$PIPE" 2>&1
}

reweight_dbest() {
  log "=== reweight dbest -> $BUF ==="
  python3 examples/benchmark/reweight_rl_buffer.py \
    --src-root "$SRC_ROLL" \
    --dst-root "$BUF" \
    --arm dbest \
    --arms-file "$ARMS_FILE" \
    --task-order-file "$ORDER" \
    --overwrite
}

train_ckpts() {
  local gpu="${TRAIN_GPU:-0}"
  local final="$CKPT_ROOT/final"
  local arm_log="$LOGDIR/${ARM_ID}_train_${TS}.log"
  mkdir -p "$CKPT_ROOT"

  if [[ -f "$CKPT_ROOT/ckpt16/adapter_model.safetensors" ]]; then
    log "SKIP train (ckpt16 exists at $CKPT_ROOT/ckpt16)"
    return 0
  fi

  log "=== TRAIN $ARM_ID gpu=$gpu lr=$LR kl=$KL ckpts=$CKPTS log=$arm_log ==="
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON_TRAIN" \
    examples/benchmark/train_p5_reinforce.py \
    --buffer-root "$BUF" \
    --init-adapter "$INIT" \
    --output-dir "$final" \
    --dataset-order-file "$ORDER" \
    --strict-dataset-order \
    --advantage-mode per_dataset \
    --dataset-sequential \
    --checkpoint-at "$CKPTS" \
    --checkpoint-root "$CKPT_ROOT" \
    --device cuda:0 \
    --lr "$LR" --kl-beta "$KL" --grad-accum 8 --max-grad-norm 1.0 --seed "$SEED" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    --wandb-run-name "${ARM_ID}_ckpt81216" \
    --wandb-group "$ARM_ID" \
    --wandb-tags "reinforce,s1b,dpo_rationale_G,dbest,hard16,ckpt81216" \
    >"$arm_log" 2>&1
  log "TRAIN finished rc=$?"
}

write_manifest() {
  python3 - <<PY
import json
from pathlib import Path
ckpts = [int(x) for x in "$CKPTS".split(",") if x.strip()]
root = Path("$CKPT_ROOT")
rows = []
for c in ckpts:
    mid = f"${ARM_ID}_ckpt{c}"
    adapter = root / f"ckpt{c}"
    if not (adapter / "adapter_model.safetensors").is_file():
        raise SystemExit(f"missing ckpt adapter: {adapter}")
    rows.append({"id": mid, "adapter": str(adapter), "resp": "json_only", "lr": "$LR", "kl_beta": "$KL"})
Path("$MANIFEST").write_text(json.dumps(rows, indent=2) + "\n")
print(f"wrote {len(rows)} models -> $MANIFEST")
PY
  log "manifest -> $MANIFEST"
}

run_eval() {
  write_manifest
  log "=== EVAL tasks_40_hard from_manifest ($EVAL_PROMPT/$EVAL_HYBRID) base=$EVAL_OUTPUT_BASE ==="
  TASKS_JSON="$EVAL_TASKS_JSON" OUTPUT_ROOT="$EVAL_OUTPUT_BASE" \
    NUM_GPUS="$NUM_GPUS" JOBS_PER_GPU="$JOBS_PER_GPU" PACK_MIN_FREE_MB="$PACK_MIN_FREE_MB" \
    EVAL_GPU_MAP="${EVAL_GPU_MAP:-}" \
    STAGE=from_manifest MANIFEST_JSON="$MANIFEST" HYBRID="$EVAL_HYBRID" PROMPT_VARIANT="$EVAL_PROMPT" \
    LLM_OPTIMIZERS=ptllm_tpe_qwen SEED="$SEED" EVAL_RESUME="$EVAL_RESUME" TIME_LIMIT=10800 \
    bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1
  HARD40_ANALYSIS="${EVAL_OUTPUT_BASE%/}/analysis/tasks_40_hard/avg_rank"
  log "regen tables (tasks_40_hard -> $HARD40_ANALYSIS)"
  python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py \
    --tasks-json "$EVAL_TASKS_JSON" \
    --out-dir "$HARD40_ANALYSIS" \
    --rgs-pooled-trial-norm \
    >>"$PIPE" 2>&1 || true
  python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py \
    --tasks-json "$EVAL_TASKS_JSON" \
    --out-dir "$HARD40_ANALYSIS" \
    --rgs-pooled-trial-norm --seed1-only \
    >>"$PIPE" 2>&1 || true
  log "=== EVAL done ==="
}

log "=== $ARM_ID pipeline START ts=$TS ==="

if [[ "$DRY_RUN" == "1" ]]; then
  validate_prereqs
  n=$(count_rollout_records)
  log "DRY_RUN ok (rollout records=$n under $SRC_ROLL)"
  exit 0
fi

validate_prereqs

if [[ "$EVAL_ONLY" != "1" ]]; then
  if [[ "$SKIP_ROLLOUTS" != "1" ]]; then
    n=$(count_rollout_records)
    need=640
    if (( n < need )); then
      run_rollouts
    else
      log "SKIP rollouts ($n records >= $need)"
    fi
  fi
  reweight_dbest
  train_ckpts
fi

if [[ "$TRAIN_ONLY" != "1" ]]; then
  run_eval
fi

log "=== $ARM_ID pipeline ALL DONE ==="
echo "$PIPE"
