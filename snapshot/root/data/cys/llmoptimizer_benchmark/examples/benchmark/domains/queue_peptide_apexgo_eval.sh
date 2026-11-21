#!/usr/bin/env bash
# Wait for peptide TPE 500-trial runs to finish, then re-score with APEXGo objective.
set -euo pipefail

ROOT="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
PY="/root/data/cys/blockdata_archive/miniconda3/envs/hpo_benchmark/bin/python"
SCRIPT="$ROOT/examples/benchmark/domains/rescore_peptide_apexgo.py"
LOG_DIR="$ROOT/benchmark_data/domain_protein/apexgo_eval/logs"
mkdir -p "$LOG_DIR"

TARGET=1000
CHECKPOINT=100
MIN_OBS=100
POLL_SEC=300

echo "[queue] waiting for >=${TARGET} peptide histories with >=500 observations..."
while true; do
  done500=$("$PY" - <<'PY'
import json
from pathlib import Path
from mindware.domains.paths import protein_bo80_bolt_root
root = protein_bo80_bolt_root()
n = 0
for h in root.glob("pep_*/tpe/seed1/history.json"):
    try:
        obs = json.loads(h.read_text()).get("observations") or []
    except Exception:
        continue
    if len(obs) >= 500:
        n += 1
print(n)
PY
)
  running=$(pgrep -cf 'run_bolt_bo_suite.*protein' || true)
  echo "[queue] done500=${done500}/${TARGET} running_shards=${running}"
  if [[ "$done500" -ge "$TARGET" && "$running" -eq 0 ]]; then
    break
  fi
  sleep "$POLL_SEC"
done

echo "[queue] launching APEXGo re-score (checkpoint=${CHECKPOINT}) on 4 GPU shards..."
export CUDA_VISIBLE_DEVICES=0
nohup "$PY" "$SCRIPT" --checkpoint "$CHECKPOINT" --min-obs "$MIN_OBS" \
  --shard-index 0 --n-shards 4 --summary-name summary_shard0.json \
  >"$LOG_DIR/apexgo_ckpt${CHECKPOINT}_s0.log" 2>&1 &
nohup "$PY" "$SCRIPT" --checkpoint "$CHECKPOINT" --min-obs "$MIN_OBS" \
  --shard-index 1 --n-shards 4 --summary-name summary_shard1.json \
  >"$LOG_DIR/apexgo_ckpt${CHECKPOINT}_s1.log" 2>&1 &
nohup "$PY" "$SCRIPT" --checkpoint "$CHECKPOINT" --min-obs "$MIN_OBS" \
  --shard-index 2 --n-shards 4 --summary-name summary_shard2.json \
  >"$LOG_DIR/apexgo_ckpt${CHECKPOINT}_s2.log" 2>&1 &
nohup "$PY" "$SCRIPT" --checkpoint "$CHECKPOINT" --min-obs "$MIN_OBS" \
  --shard-index 3 --n-shards 4 --summary-name summary_shard3.json \
  >"$LOG_DIR/apexgo_ckpt${CHECKPOINT}_s3.log" 2>&1 &

echo "[queue] also scoring checkpoint 500 once histories are ready..."
nohup "$PY" "$SCRIPT" --checkpoint 500 --min-obs 500 --shard-index 0 --n-shards 4 \
  --out-dir "$ROOT/benchmark_data/domain_protein/apexgo_eval/ckpt500" \
  --summary-name summary_shard0.json >"$LOG_DIR/apexgo_ckpt500_s0.log" 2>&1 &
nohup "$PY" "$SCRIPT" --checkpoint 500 --min-obs 500 --shard-index 1 --n-shards 4 \
  --out-dir "$ROOT/benchmark_data/domain_protein/apexgo_eval/ckpt500" \
  --summary-name summary_shard1.json >"$LOG_DIR/apexgo_ckpt500_s1.log" 2>&1 &
nohup "$PY" "$SCRIPT" --checkpoint 500 --min-obs 500 --shard-index 2 --n-shards 4 \
  --out-dir "$ROOT/benchmark_data/domain_protein/apexgo_eval/ckpt500" \
  --summary-name summary_shard2.json >"$LOG_DIR/apexgo_ckpt500_s2.log" 2>&1 &
nohup "$PY" "$SCRIPT" --checkpoint 500 --min-obs 500 --shard-index 3 --n-shards 4 \
  --out-dir "$ROOT/benchmark_data/domain_protein/apexgo_eval/ckpt500" \
  --summary-name summary_shard3.json >"$LOG_DIR/apexgo_ckpt500_s3.log" 2>&1 &

echo "[queue] APEXGo eval jobs launched. Logs: $LOG_DIR"
