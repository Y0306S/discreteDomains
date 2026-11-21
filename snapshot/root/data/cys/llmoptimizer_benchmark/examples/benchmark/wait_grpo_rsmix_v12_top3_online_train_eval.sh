#!/usr/bin/env bash
# Online GRPO on top-3 v12 adapters: collect → update per task (batch=12 jobs).
#
# Online adaptive only: collect with adaptive 3+1, GRPO update per dataset block (batch=12).
#
# Top-3 inits (same as Phase A):
#   v4 ckpt32, v4 ckpt16, v3 ckpt8
#
# Usage:
#   bash examples/benchmark/wait_grpo_rsmix_v12_top3_online_train_eval.sh
#   SKIP_TRAIN=1 SKIP_EVAL=1 bash ...
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
cd "$REPO"

LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
LF_ENV="${LF_ENV:-/root/miniconda3/envs/llamafactory_qwen35}"
EVAL_ENV="${EVAL_ENV:-/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval}"
PY_TRAIN="${PY_TRAIN:-$LF_ENV/bin/python}"
PY_EVAL="${PY_EVAL:-$EVAL_ENV/bin/python}"
[[ -x "$PY_TRAIN" ]] || PY_TRAIN="/root/data/cys/blockdata_archive/miniconda3/envs/llamafactory_qwen35/bin/python"
[[ -x "$PY_EVAL" ]] || PY_EVAL="/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python"

ORDER="${ORDER:-benchmark_data/rl_task_selection/picked_32_pareto_train_order.json}"
TRAIN_MEMBER_POLICY="${TRAIN_MEMBER_POLICY:-adaptive}"
BATCH_SIZE="${BATCH_SIZE:-12}"
# Online starts from a peak adapter: 1e-6 x 32 noisy per-task steps moved CLS off the peak.
LR="${LR:-3e-7}"
KL_BETA="${KL_BETA:-0.10}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
TRAIN_GPUS="${TRAIN_GPUS:-0 1 2}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

OUTPUT_BASE="${OUTPUT_BASE:-benchmark_data/real_hpo_v2_mltest20_algo40_p5plus_grpo_rsmix_v12_top3_online}"
MANIFEST="${MANIFEST:-benchmark_data/rl_task_selection/b_prefix_grpo_rsmix_v12_top3_online_manifest.json}"

TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/grpo_rsmix_v12_top3_online_${TS}.log"
mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

# ver|ckpt|init_adapter|rs_version|best_eval_prompt
declare -a TOP3=(
  "v4|32|/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_v4_pareto32_v12/ckpt32|v4|P5plusA5T50X50RS5f05v4"
  "v4|16|/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_v4_pareto32_v12/ckpt16|v4|P5plusA5T50X50RS3f05v4"
  "v3|8|/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_v3_pareto32_v12/ckpt8|v3|P5plusA5T50X50RS5f05"
)

SUFFIX="online_adaptive_b12"

log "=== online GRPO top-3 adaptive batch_size=$BATCH_SIZE lr=$LR kl=$KL_BETA max_seq_len=$MAX_SEQ_LEN ==="
log "SKIP_TRAIN=$SKIP_TRAIN SKIP_EVAL=$SKIP_EVAL pipe=$PIPE"

if [[ "$SKIP_TRAIN" != "1" ]]; then
  export PATH="$LF_ENV/bin:$PATH"
  read -ra TGPU_ARR <<<"$TRAIN_GPUS"
  declare -a PIDS=()
  declare -a LABELS=()
  i=0
  for spec in "${TOP3[@]}"; do
    IFS='|' read -r ver ckpt init rs _prompt <<<"$spec"
    out="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_${ver}_pareto32_v12_ckpt${ckpt}_${SUFFIX}"
    gpu="${TGPU_ARR[$((i % ${#TGPU_ARR[@]}))]}"
    i=$((i + 1))
    if [[ ! -f "$init/adapter_model.safetensors" ]]; then
      log "ERROR missing init $init"
      exit 1
    fi
    rm -rf "$out"
    log "=== online train $ver ckpt$ckpt adaptive GPU=$gpu init=$init out=$out ==="
    (
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY_TRAIN" examples/benchmark/train_p5_online_grpo.py \
        --init-adapter "$init" \
        --ref-adapter "$init" \
        --output-dir "$out" \
        --checkpoint-root "$out" \
        --checkpoint-at 8,16,32 \
        --dataset-order-file "$ORDER" \
        --rs-version "$rs" \
        --train-member-policy "$TRAIN_MEMBER_POLICY" \
        --batch-size "$BATCH_SIZE" \
        --py-eval "$PY_EVAL" \
        --max-seq-len "$MAX_SEQ_LEN" \
        --lr "$LR" --kl-beta "$KL_BETA" --seed 1
    ) >>"$PIPE" 2>&1 &
    PIDS+=("$!")
    LABELS+=("${ver}_ckpt${ckpt}_${SUFFIX}")
  done
  fail=0
  for j in "${!PIDS[@]}"; do
    if wait "${PIDS[$j]}"; then
      log "online train OK ${LABELS[$j]}"
    else
      log "ERROR online train FAILED ${LABELS[$j]}"
      fail=1
    fi
  done
  (( fail == 0 )) || exit 1
else
  log "SKIP_TRAIN=1"
fi

log "=== write manifest -> $MANIFEST ==="
"$PY_TRAIN" - <<'PY' | tee -a "$PIPE"
import json
from pathlib import Path

top3 = [
    ("v4", 32, "v4", "P5plusA5T50X50RS5f05v4"),
    ("v4", 16, "v4", "P5plusA5T50X50RS3f05v4"),
    ("v3", 8, "v3", "P5plusA5T50X50RS5f05"),
]
suffix = "online_adaptive_b12"

rows = []
for ver, ckpt, rs, prompt in top3:
    adapter = Path(
        f"/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_{ver}_pareto32_v12_ckpt{ckpt}_{suffix}"
    )
    mid = f"s1b_attr_t2_b_prefix_grpo_rsmix_{ver}_pareto32_v12_ckpt{ckpt}_{suffix}"
    if not (adapter / "adapter_model.safetensors").exists():
        raise SystemExit(f"missing {adapter}")
    rows.append({
        "id": mid,
        "adapter": str(adapter),
        "resp": "json_only",
        "rs_version": ver,
        "ckpt": ckpt,
        "recipe": suffix,
        "init": f"grpo_rsmix_{ver}_ckpt{ckpt}",
        "tasks": "picked_32_pareto",
        "schedule": "rs_mix_v12_online_b12",
        "train_member_policy": "adaptive",
        "best_prompt": prompt,
    })
mf = Path("benchmark_data/rl_task_selection/b_prefix_grpo_rsmix_v12_top3_online_manifest.json")
mf.write_text(json.dumps(rows, indent=2) + "\n")
print(f"wrote {len(rows)} -> {mf}")
PY

if [[ "$SKIP_EVAL" == "1" ]]; then
  log "SKIP_EVAL=1 manifest=$MANIFEST pipe=$PIPE"
  exit 0
fi

run_model_eval() {
  local model_id="$1" prompt="$2"
  local mf="benchmark_data/rl_task_selection/b_prefix_grpo_rsmix_v12_top3_online_${model_id}_${prompt}.json"
  "$PY_TRAIN" - <<PY
import json
from pathlib import Path
rows = [r for r in json.loads(Path("$MANIFEST").read_text()) if r.get("id") == "$model_id"]
Path("$mf").write_text(json.dumps(rows, indent=2) + "\n")
print(f"wrote {len(rows)} -> $mf")
if len(rows) != 1:
    raise SystemExit(1)
PY
  log "=== eval online $model_id prompt=$prompt ==="
  TASKS_JSON=benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40.json \
    OUTPUT_ROOT="$OUTPUT_BASE" NUM_GPUS="${EVAL_NUM_GPUS:-4}" JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-2}" \
    PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-15000}" \
    STAGE=from_manifest MANIFEST_JSON="$mf" \
    HYBRID=HPUCB_arm PROMPT_VARIANT="$prompt" LLM_OPTIMIZERS=ptllm_tpe_qwen \
    SEED=1 EVAL_RESUME=1 TIME_LIMIT=10800 \
    PYTHON_LLM="$PY_EVAL" PYTHON_CLASS="$PY_EVAL" \
    bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1
}

while IFS= read -r line; do
  id=$(python3 -c "import json,sys; r=json.loads(sys.stdin.read()); print(r['id'])" <<<"$line")
  prompt=$(python3 -c "import json,sys; r=json.loads(sys.stdin.read()); print(r['best_prompt'])" <<<"$line")
  run_model_eval "$id" "$prompt"
done < <("$PY_TRAIN" - <<'PY'
import json
from pathlib import Path
for r in json.loads(Path("benchmark_data/rl_task_selection/b_prefix_grpo_rsmix_v12_top3_online_manifest.json").read_text()):
    print(json.dumps(r))
PY
)

log "=== rebuild tables ==="
"$PY_EVAL" examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm --seed1-only >>"$PIPE" 2>&1 || true

log "=== online top-3 DONE ==="
log "manifest=$MANIFEST output_base=$OUTPUT_BASE pipe=$PIPE"
