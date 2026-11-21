#!/usr/bin/env bash
# Re-eval RS-mix GRPO adapters with version-matched P5+ RS10f05 protocol.
#
# Three batches (one PROMPT_VARIANT per manifest run):
#   v3 -> P5plusA5T50X50RS10f03
#   v4 -> P5plusA5T50X50RS10f05v4
#   v5 -> P5plusA5T50X50RS10f05v5
# Planner: ptllm_tpe_qwen, hybrid: HPUCB_arm
#
# Usage:
#   bash examples/benchmark/wait_eval_grpo_rsmix_v12_p5plus_rs10f05.sh
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

FULL_MANIFEST="${FULL_MANIFEST:-benchmark_data/rl_task_selection/b_prefix_grpo_rsmix_v12_pareto32_ckpt81632_manifest.json}"
OUTPUT_BASE="${OUTPUT_BASE:-benchmark_data/real_hpo_v2_mltest20_algo40_p5plus_grpo_rsmix_v12}"
LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
PY_EVAL="${PY_EVAL:-/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python}"
PYTHON_CLASS="${PYTHON_CLASS:-/root/data/cys/blockdata_archive/miniconda3/envs/openbox_eval/bin/python}"
[[ -x "$PY_EVAL" ]] || PY_EVAL="/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python"
[[ -x "$PYTHON_CLASS" ]] || PYTHON_CLASS="/root/data/cys/blockdata_archive/miniconda3/envs/openbox_eval/bin/python"

NUM_GPUS="${EVAL_NUM_GPUS:-4}"
JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-2}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-15000}"
SEED="${SEED:-1}"
EVAL_RESUME="${EVAL_RESUME:-1}"
TIME_LIMIT="${TIME_LIMIT:-10800}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
PIPE="$LOG_ROOT/eval_grpo_rsmix_v12_p5plus_rs10f05_${TS}.log"
mkdir -p "$LOG_ROOT" "$(dirname "$OUTPUT_BASE")"

log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

write_version_manifest() {
  local ver="$1" out="$2"
  "$PYTHON_CLASS" - <<PY
import json
from pathlib import Path

ver = "$ver"
rows = [r for r in json.loads(Path("$FULL_MANIFEST").read_text()) if r.get("rs_version") == ver]
missing = [r for r in rows if not (Path(r["adapter"]) / "adapter_model.safetensors").exists()]
Path("$out").write_text(json.dumps(rows, indent=2) + "\n")
print(f"wrote {len(rows)} -> $out")
if missing:
    for r in missing:
        print("MISSING", r["id"], r["adapter"])
    raise SystemExit(1)
PY
}

run_batch() {
  local ver="$1" prompt="$2"
  local mf="benchmark_data/rl_task_selection/b_prefix_grpo_rsmix_v12_p5plus_rs10f05_manifest_${ver}.json"
  write_version_manifest "$ver" "$mf"
  log "=== eval batch $ver prompt=$prompt hybrid=HPUCB_arm (${mf}) ==="
  if [[ "$SKIP_EVAL" == "1" ]]; then
    return 0
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    TASKS_JSON=benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40.json \
      OUTPUT_ROOT="$OUTPUT_BASE" NUM_GPUS=1 JOBS_PER_GPU=1 \
      STAGE=from_manifest MANIFEST_JSON="$mf" \
      HYBRID=HPUCB_arm PROMPT_VARIANT="$prompt" LLM_OPTIMIZERS=ptllm_tpe_qwen \
      SEED="$SEED" EVAL_RESUME="$EVAL_RESUME" TIME_LIMIT=300 N_TRIALS=6 \
      PYTHON_LLM="$PY_EVAL" PYTHON_CLASS="$PYTHON_CLASS" \
      bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1
    return 0
  fi
  TASKS_JSON=benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40.json \
    OUTPUT_ROOT="$OUTPUT_BASE" NUM_GPUS="$NUM_GPUS" JOBS_PER_GPU="$JOBS_PER_GPU" \
    PACK_MIN_FREE_MB="$PACK_MIN_FREE_MB" \
    STAGE=from_manifest MANIFEST_JSON="$mf" \
    HYBRID=HPUCB_arm PROMPT_VARIANT="$prompt" LLM_OPTIMIZERS=ptllm_tpe_qwen \
    SEED="$SEED" EVAL_RESUME="$EVAL_RESUME" TIME_LIMIT="$TIME_LIMIT" \
    PYTHON_LLM="$PY_EVAL" PYTHON_CLASS="$PYTHON_CLASS" \
    bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1
}

log "=== RS-mix GRPO P5+ RS10f05 eval START (9 adapters) ==="
log "output_base=$OUTPUT_BASE full_manifest=$FULL_MANIFEST gpus=$NUM_GPUS jobs_per_gpu=$JOBS_PER_GPU python=$PY_EVAL"

run_batch v3 P5plusA5T50X50RS10f03
run_batch v4 P5plusA5T50X50RS10f05v4
run_batch v5 P5plusA5T50X50RS10f05v5

if [[ "$SKIP_EVAL" != "1" && "$DRY_RUN" != "1" ]]; then
  log "=== rebuild drop20pct tables ==="
  "$PY_EVAL" examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm >>"$PIPE" 2>&1 || true
  "$PY_EVAL" examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm --seed1-only >>"$PIPE" 2>&1 || true
fi

log "=== RS-mix GRPO P5+ RS10f05 eval DONE ==="
log "pipe=$PIPE"
echo "$PIPE"
