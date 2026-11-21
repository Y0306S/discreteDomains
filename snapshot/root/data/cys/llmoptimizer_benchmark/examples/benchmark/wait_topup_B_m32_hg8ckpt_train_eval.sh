#!/usr/bin/env bash
# 1) Top up kick/xgboost + Allstate/catboost BO histories to 100 with rationale_G P5/H0 tpe
# 2) Setup B rollouts for those 2 tasks into medium32 buffer
# 3) One dsseq train (mid@8=hg8, final=m32) + algo40 eval both + tables
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
cd "$REPO"
LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
EVAL_ENV="${EVAL_ENV:-/root/miniconda3/envs/openbox_qwen_eval}"
LF_ENV="${LF_ENV:-/root/miniconda3/envs/llamafactory_qwen35}"
PY_EVAL="${PY_EVAL:-$EVAL_ENV/bin/python}"
PY_TRAIN="${PY_TRAIN:-$LF_ENV/bin/python}"
ADAPTER_G="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_rationale_G"
INIT="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5"
BUF="${BUF:-benchmark_data/rl_rollouts/B_prefix10to100_n4_T1p0_medium32}"
ORDER="${ORDER:-benchmark_data/rl_task_selection/medium_32_train_order_hg8_first.json}"
OUT_MID="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_b_prefix_dsseq_hg8"
OUT_FINAL="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_b_prefix_dsseq_m32"
BO_ROOT="${BO_ROOT:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark/history_data/history_data}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/topup_B_m32_hg8ckpt_train_eval_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

count_bo() {
  local path="$1"
  "$PY_EVAL" - <<PY
import json
from pathlib import Path
p=Path("$path")
d=json.loads(p.read_text())
print(len([o for o in d.get("observations") or [] if int(o.get("trial_state",-1))==0]))
PY
}

count_records() {
  "$PY_TRAIN" - <<PY
from pathlib import Path
root=Path("$BUF")
n=0
if root.exists():
  for p in root.rglob("rollouts_*.jsonl"):
    n += sum(1 for ln in p.open() if ln.strip())
print(n)
PY
}

log "=== stage1: top up BO histories with rationale_G ==="
KICK_BO="$BO_ROOT/cls/bal_acc/kick/xgboost.json"
ALL_BO="$BO_ROOT/rgs/mse/Allstate_Claims_Severity/catboost.json"
log "kick obs=$(count_bo "$KICK_BO") Allstate obs=$(count_bo "$ALL_BO")"

# Parallel top-ups on GPU 0 and 1
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PY_EVAL" examples/benchmark/topup_bo_history_llm.py \
  --task-type cls --dataset kick --algorithm xgboost --metric bal_acc --target-n 100 \
  --qwen-adapter-path "$ADAPTER_G" --llm-response-format rationale_json \
  --prompt-variant P5 --hybrid-variant H0 \
  --output-dir "benchmark_data/bo_topup_runs/kick_xgboost_${TS}" \
  >>"$PIPE" 2>&1 &
pid_k=$!
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 "$PY_EVAL" examples/benchmark/topup_bo_history_llm.py \
  --task-type rgs --dataset Allstate_Claims_Severity --algorithm catboost --metric mse --target-n 100 \
  --qwen-adapter-path "$ADAPTER_G" --llm-response-format rationale_json \
  --prompt-variant P5 --hybrid-variant H0 \
  --output-dir "benchmark_data/bo_topup_runs/Allstate_catboost_${TS}" \
  >>"$PIPE" 2>&1 &
pid_a=$!
fail=0
wait "$pid_k" || fail=1
wait "$pid_a" || fail=1
if (( fail )); then
  log "ERROR top-up failed"
  exit 1
fi
log "after topup kick=$(count_bo "$KICK_BO") Allstate=$(count_bo "$ALL_BO")"

log "=== stage2: Setup B rollouts for kick + Allstate ==="
# remove partial wrong-algo dirs; use original pairs
rm -rf "$BUF/cls_kick_catboost" "$BUF/rgs_Allstate_Claims_Severity_lightgbm" || true
# restore order file algos to originals
"$PY_TRAIN" - <<'PY'
import json
from pathlib import Path
p=Path("benchmark_data/rl_task_selection/medium_32_train_order_hg8_first.json")
order=json.loads(p.read_text())
for t in order:
  if t["dataset"]=="kick":
    t["algorithm"]="xgboost"
  if t["dataset"]=="Allstate_Claims_Severity":
    t["algorithm"]="catboost"
p.write_text(json.dumps(order, indent=2)+"\n")
print("order restored to kick/xgboost + Allstate/catboost")
PY

TASKS_JSON=benchmark_data/rl_task_selection/medium_32_bo_topup_pair.json
"$PY_TRAIN" - <<'PY'
import json
from pathlib import Path
Path("benchmark_data/rl_task_selection/medium_32_bo_topup_pair.json").write_text(json.dumps([
  {"task_type":"cls","dataset":"kick","algorithm":"xgboost","metric":"bal_acc"},
  {"task_type":"rgs","dataset":"Allstate_Claims_Severity","algorithm":"catboost","metric":"mse"},
], indent=2)+"\n")
PY

TASKS_JSON="$TASKS_JSON" OUT_ROOT="$BUF" GPU_IDS=0,1 JOBS_PER_GPU=1 PACK_MIN_FREE_MB=22000 \
  bash examples/benchmark/run_setupB_prefix_rollouts.sh >>"$PIPE" 2>&1

n=$(count_records)
log "buffer records after B pair=$n (need >= 1280)"
if (( n < 1280 )); then
  log "ERROR buffer still short"
  exit 1
fi

log "=== stage3: train once mid@8 + final m32 ==="
export PATH="$LF_ENV/bin:$PATH"
CUDA_VISIBLE_DEVICES="${TRAIN_GPU:-0}" PYTHONUNBUFFERED=1 "$PY_TRAIN" examples/benchmark/train_p5_reinforce.py \
  --buffer-root "$BUF" \
  --init-adapter "$INIT" \
  --output-dir "$OUT_FINAL" \
  --mid-checkpoint-dir "$OUT_MID" \
  --mid-checkpoint-after 8 \
  --dataset-order-file "$ORDER" \
  --advantage-mode per_dataset \
  --dataset-sequential \
  --lr 1e-6 --kl-beta 0.05 --grad-accum 8 --seed 1 \
  >>"$PIPE" 2>&1
log "train done"

log "=== stage4: algo40 eval both ==="
NUM_GPUS="${EVAL_NUM_GPUS:-4}" JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-3}" PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-15000}" \
  STAGE=s1a_b_prefix_dsseq_hg8_m32 HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen \
  SEED=1 EVAL_RESUME=1 TIME_LIMIT=10800 \
  bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1

log "=== tables ==="
python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm >>"$PIPE" 2>&1 || true
python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm --seed1-only >>"$PIPE" 2>&1 || true
log "=== ALL COMPLETE ==="
echo "$PIPE"
