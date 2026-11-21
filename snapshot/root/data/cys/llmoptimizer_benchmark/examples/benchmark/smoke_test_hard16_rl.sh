#!/usr/bin/env bash
# Smoke test hard16 RL pipeline: selection, BO histories, 1-task rollout, reweight, train load.
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

EVAL_ENV="${EVAL_ENV:-/root/miniconda3/envs/openbox_qwen_eval}"
PY="${PY:-$EVAL_ENV/bin/python}"
LF_ENV="${LF_ENV:-/root/miniconda3/envs/llamafactory_qwen35}"
PY_TRAIN="${PY_TRAIN:-$LF_ENV/bin/python}"
TS="$(date +%Y%m%d_%H%M%S)"
SMOKE_ROOT="benchmark_data/rl_rollouts/_smoke_hard16_${TS}"
TASKS_JSON="benchmark_data/rl_task_selection/hard_16.json"
INIT="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_rationale_G"
ROLLOUT_ADAPTER="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5"

log() { echo "[$(date -Is)] $*"; }

log "=== smoke 1/5: select hard16 + dry-run pipeline ==="
python3 examples/benchmark/select_hard16_from_hard40.py --out "$TASKS_JSON"
DRY_RUN=1 bash examples/benchmark/run_hard16_rl_dbest_ckpt81216.sh

log "=== smoke 2/5: bash syntax ==="
bash -n examples/benchmark/run_hard16_rl_dbest_ckpt81216.sh
bash -n examples/benchmark/wait_evals_then_hard16_rl.sh
bash -n examples/benchmark/run_setupB_prefix_rollouts.sh

log "=== smoke 3/5: single-task rollout (prefix 10, n=1) ==="
mkdir -p "$SMOKE_ROOT"
first="$("$PY" - <<PY
import json
from pathlib import Path
t = json.loads(Path("$TASKS_JSON").read_text())[0]
print(f"{t['task_type']}|{t['dataset']}|{t['algorithm']}")
PY
)"
IFS='|' read -r tt ds algo <<<"$first"
metric="bal_acc"
[[ "$tt" == "rgs" ]] && metric="mse"
outdir="$SMOKE_ROOT/${tt}_${ds}_${algo}"
CUDA_VISIBLE_DEVICES="${SMOKE_GPU:-0}" "$PY" examples/benchmark/run_p5_rl_bo_prefix_b.py \
  --task-type "$tt" --dataset "$ds" --algorithm "$algo" --metric "$metric" \
  --prefixes 10 --n-samples 1 \
  --seed 1 --time-limit 3600 --output-dir "$outdir" \
  --prompt-variant P5 --qwen-backend hf \
  --qwen-model-path /root/eb-public/huggingface-models/Qwen/Qwen3.5-9B \
  --qwen-adapter-path "$ROLLOUT_ADAPTER" \
  --llm-response-format json_only --llm-max-tokens 768 --llm-temperature 1.0
n_rec="$("$PY" - <<PY
from pathlib import Path
p = list(Path("$outdir").glob("rollouts_*.jsonl"))
print(sum(1 for ln in p[0].open() if ln.strip()) if p else 0)
PY
)"
log "rollout records=$n_rec (expect >=1)"
(( n_rec >= 1 )) || { log "FAIL rollout"; exit 1; }

log "=== smoke 4/5: reweight dbest on smoke buffer ==="
SMOKE_BUF="$SMOKE_ROOT/B_rw_dbest_smoke"
python3 examples/benchmark/reweight_rl_buffer.py \
  --src-root "$SMOKE_ROOT" --dst-root "$SMOKE_BUF" --arm dbest \
  --arms-file benchmark_data/rl_task_selection/reward_sweep_arms.json \
  --task-order-file "$TASKS_JSON" --overwrite

log "=== smoke 5/5: train script loads buffer (1 record sanity) ==="
"$PY_TRAIN" - <<PY
import json
from pathlib import Path
from train_p5_reinforce import load_records
recs = load_records(Path("$SMOKE_BUF"))
assert recs, "no records"
print("ok load_records n=", len(recs), "reward=", recs[0].get("reward"))
PY

log "=== smoke test PASSED ==="
log "smoke artifacts: $SMOKE_ROOT (safe to delete)"
