#!/usr/bin/env bash
# Phase A: strict-first-4 GRPO continue-train on top-3 v12 adapters, eval RS3+RS5 only.
#
# Top-3 (by RS sweep):
#   v4 ckpt32  (#1 RS5f05v4)
#   v4 ckpt16  (best RS3 among v4)
#   v3 ckpt8   (strong RS5)
#
# Usage:
#   bash examples/benchmark/wait_grpo_rsmix_v12_top3_strict4_train_eval.sh
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
MANIFEST="${MANIFEST:-benchmark_data/rl_task_selection/b_prefix_grpo_rsmix_v12_top3_strict4_manifest.json}"
OUTPUT_BASE="${OUTPUT_BASE:-benchmark_data/real_hpo_v2_mltest20_algo40_p5plus_grpo_rsmix_v12_top3_strict4}"
LR="${LR:-1e-6}"
KL_BETA="${KL_BETA:-0.05}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
TRAIN_GPUS="${TRAIN_GPUS:-0 1 2}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/grpo_rsmix_v12_top3_strict4_${TS}.log"
mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

declare -a SPECS=(
  "v4|32|/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_v4_pareto32_v12/ckpt32|benchmark_data/rl_rollouts/B_rsmix_v4_n4to10_T1p0_pareto32_v12|/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_v4_pareto32_v12_ckpt32_strict4_r1"
  "v4|16|/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_v4_pareto32_v12/ckpt16|benchmark_data/rl_rollouts/B_rsmix_v4_n4to10_T1p0_pareto32_v12|/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_v4_pareto32_v12_ckpt16_strict4_r1"
  "v3|8|/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_v3_pareto32_v12/ckpt8|benchmark_data/rl_rollouts/B_rsmix_v3_n4to10_T1p0_pareto32_v12|/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_v3_pareto32_v12_ckpt8_strict4_r1"
)

log "=== Phase A: top-3 strict-first-4 GRPO continue-train + RS3/RS5 eval ==="

if [[ "$SKIP_TRAIN" != "1" ]]; then
  export PATH="$LF_ENV/bin:$PATH"
  read -ra TGPU_ARR <<<"$TRAIN_GPUS"
  declare -a PIDS=()
  i=0
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r ver ckpt init buf out <<<"$spec"
    gpu="${TGPU_ARR[$((i % ${#TGPU_ARR[@]}))]}"
    i=$((i + 1))
    if [[ ! -f "$init/adapter_model.safetensors" ]]; then
      log "ERROR missing init $init"
      exit 1
    fi
    log "=== train $ver ckpt$ckpt strict4 GPU=$gpu init=$init buf=$buf out=$out ==="
    (
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY_TRAIN" examples/benchmark/train_p5_reinforce.py \
        --buffer-root "$buf" \
        --strict-first-n 4 \
        --init-adapter "$init" \
        --output-dir "$out" \
        --dataset-order-file "$ORDER" \
        --strict-dataset-order \
        --advantage-mode grpo_group \
        --dataset-sequential \
        --lr "$LR" --kl-beta "$KL_BETA" --grad-accum "$GRAD_ACCUM" --seed 1
    ) >>"$PIPE" 2>&1 &
    PIDS+=("$!")
  done
  fail=0
  for pid in "${PIDS[@]}"; do
    if wait "$pid"; then
      log "train OK pid=$pid"
    else
      log "ERROR train FAILED pid=$pid"
      fail=1
    fi
  done
  if (( fail )); then
    exit 1
  fi
else
  log "SKIP_TRAIN=1"
fi

log "=== write manifest -> $MANIFEST ==="
"$PY_TRAIN" - <<'PY' | tee -a "$PIPE"
import json
from pathlib import Path

specs = [
    ("v4", 32, "s1b_attr_t2_b_prefix_grpo_rsmix_v4_pareto32_v12_ckpt32_strict4_r1",
     "/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_v4_pareto32_v12_ckpt32_strict4_r1"),
    ("v4", 16, "s1b_attr_t2_b_prefix_grpo_rsmix_v4_pareto32_v12_ckpt16_strict4_r1",
     "/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_v4_pareto32_v12_ckpt16_strict4_r1"),
    ("v3", 8, "s1b_attr_t2_b_prefix_grpo_rsmix_v3_pareto32_v12_ckpt8_strict4_r1",
     "/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_v3_pareto32_v12_ckpt8_strict4_r1"),
]
rows = []
for ver, ckpt, mid, adapter in specs:
    p = Path(adapter)
    if not (p / "adapter_model.safetensors").exists():
        raise SystemExit(f"missing adapter {p}")
    rows.append({
        "id": mid,
        "adapter": str(p),
        "resp": "json_only",
        "rs_version": ver,
        "ckpt": ckpt,
        "recipe": "b_prefix_grpo_strict4_r1",
        "init": f"grpo_rsmix_{ver}_ckpt{ckpt}",
        "tasks": "picked_32_pareto",
        "schedule": "rs_mix_v12_strict4_top3",
    })
mf = Path("benchmark_data/rl_task_selection/b_prefix_grpo_rsmix_v12_top3_strict4_manifest.json")
mf.write_text(json.dumps(rows, indent=2) + "\n")
print(f"wrote {len(rows)} -> {mf}")
PY

if [[ "$SKIP_EVAL" == "1" ]]; then
  log "SKIP_EVAL=1 manifest=$MANIFEST pipe=$PIPE"
  exit 0
fi

run_model_eval() {
  local model_id="$1" prompt="$2"
  local mf="benchmark_data/rl_task_selection/b_prefix_grpo_rsmix_v12_top3_strict4_manifest_${model_id}_${prompt}.json"
  "$PY_TRAIN" - <<PY
import json
from pathlib import Path
rows = [r for r in json.loads(Path("$MANIFEST").read_text()) if r.get("id") == "$model_id"]
Path("$mf").write_text(json.dumps(rows, indent=2) + "\n")
print(f"wrote {len(rows)} -> $mf")
if len(rows) != 1:
    raise SystemExit(1)
PY
  log "=== eval strict4 $model_id best prompt=$prompt (${mf}) ==="
  TASKS_JSON=benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40.json \
    OUTPUT_ROOT="$OUTPUT_BASE" NUM_GPUS="${EVAL_NUM_GPUS:-4}" JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-2}" \
    PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-15000}" \
    STAGE=from_manifest MANIFEST_JSON="$mf" \
    HYBRID=HPUCB_arm PROMPT_VARIANT="$prompt" LLM_OPTIMIZERS=ptllm_tpe_qwen \
    SEED=1 EVAL_RESUME=1 TIME_LIMIT=10800 \
    PYTHON_LLM="$PY_EVAL" PYTHON_CLASS="$PY_EVAL" \
    bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1
}

# Each unit on its version-matched BEST RS prompt only (from v12 sweep)
run_model_eval s1b_attr_t2_b_prefix_grpo_rsmix_v4_pareto32_v12_ckpt32_strict4_r1 P5plusA5T50X50RS5f05v4
run_model_eval s1b_attr_t2_b_prefix_grpo_rsmix_v4_pareto32_v12_ckpt16_strict4_r1 P5plusA5T50X50RS3f05v4
run_model_eval s1b_attr_t2_b_prefix_grpo_rsmix_v3_pareto32_v12_ckpt8_strict4_r1 P5plusA5T50X50RS5f05

log "=== rebuild tables ==="
"$PY_EVAL" examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm --seed1-only >>"$PIPE" 2>&1 || true

log "=== Phase A DONE ==="
log "manifest=$MANIFEST output_base=$OUTPUT_BASE pipe=$PIPE"
