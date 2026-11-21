#!/usr/bin/env bash
# Wait for Qwen P5+ sweep to finish, then repair + complete GPT RC/RS retrieve bake-off.
#
# The 10 "partial" RC/RS slots are not missing trials — they have 80 rows but the
# wrong algorithm in the traj (shared dataset/variant dir across algo40 tasks).
# This script removes mismatched CASH runs and re-runs only those jobs.
#
#   bash examples/benchmark/run_p5plus_retrieve_h2_finish_after_qwen.sh
#   QWEN_PID=1010557 bash ...
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

LOGDIR="${LOGDIR:-benchmark_data/real_hpo_v2_mltest20_algo40/logs}"
HANDOVER_LOG="$LOGDIR/retrieve_h2_finish_after_qwen.log"
QWEN_PID="${QWEN_PID:-$(cat "$LOGDIR/qwen_p5plus_attr_t2_phase2.pid" 2>/dev/null || true)}"
PARALLEL="${PARALLEL:-4}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

log() { echo "[$(date -Is)] $*" | tee -a "$HANDOVER_LOG"; }

if [[ -z "$QWEN_PID" ]]; then
  log "WARN: no QWEN_PID — checking if qwen launcher is running"
  QWEN_PID="$(pgrep -f 'run_p5plus_qwen_attr_t2_eval.sh' | head -1 || true)"
fi

if [[ -n "$QWEN_PID" ]] && kill -0 "$QWEN_PID" 2>/dev/null; then
  log "waiting for Qwen sweep pid=$QWEN_PID"
  while kill -0 "$QWEN_PID" 2>/dev/null; do
    sleep 60
  done
  log "Qwen sweep pid=$QWEN_PID finished"
else
  log "Qwen sweep not running (pid=${QWEN_PID:-none}) — proceeding immediately"
fi

log "scanning RC/RS jobs needing repair"
MANIFEST="$LOGDIR/retrieve_h2_repair_manifest_${TS}.txt"
PYTHON="${PYTHON:-/root/miniconda3/envs/openbox_eval/bin/python}"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" - "$MANIFEST" <<'PY' | tee -a "$HANDOVER_LOG"
import csv, json, shutil, sys
from collections import Counter
from pathlib import Path

REPO = Path("/root/data/cys/blockdata_archive/llmoptimizer_benchmark")
ROOT = REPO / "benchmark_data/real_hpo_v2_mltest20_algo40"
tasks = json.loads((ROOT / "tasks_40.json").read_text())
PROMPTS = ["P5plusA5T50X50RC", "P5plusA5T50X50RS"]
manifest_path = Path(sys.argv[1])

def traj_mode_algo(traj: Path) -> str | None:
    rows = list(csv.DictReader(traj.open(newline="", encoding="utf-8")))
    algos = [str(r.get("algorithm") or "").strip() for r in rows if str(r.get("algorithm") or "").strip()]
    if not algos:
        return None
    return Counter(algos).most_common(1)[0][0]

needs = []
for t in tasks:
    tt, ds, algo = t["task_type"], t["dataset"], t["algorithm"]
    for prompt in PROMPTS:
        variant = f"ptllm_tpe__{prompt}__H2"
        opt_dir = ROOT / tt / ds / variant
        if not opt_dir.is_dir():
            needs.append((prompt, tt, ds, algo, "missing_dir"))
            continue
        bad_runs = []
        good = False
        for run in sorted(opt_dir.glob("CASH-*")):
            trajs = sorted(run.glob("*_traj.csv"))
            if not trajs:
                bad_runs.append(run)
                continue
            mode = traj_mode_algo(trajs[-1])
            n = sum(1 for _ in csv.DictReader(trajs[-1].open())) 
            if mode == algo and n >= 80:
                good = True
            else:
                bad_runs.append(run)
        if good:
            continue
        for run in bad_runs:
            shutil.rmtree(run, ignore_errors=True)
        needs.append((prompt, tt, ds, algo, f"cleaned_{len(bad_runs)}_runs"))

print(f"repair_jobs={len(needs)}")
for row in needs:
    prompt, tt, ds, algo, why = row
    print(f"NEED|{prompt}|{tt}|{ds}|{algo}|{why}")
manifest_path.write_text(
    "\n".join(f"{p}|{tt}|{ds}|{a}" for p, tt, ds, a, _ in needs) + ("\n" if needs else ""),
    encoding="utf-8",
)
PY

if [[ ! -f "$MANIFEST" ]] || [[ ! -s "$MANIFEST" ]]; then
  log "nothing to repair — RC/RS already complete"
  exit 0
fi

NJOBS=$(wc -l <"$MANIFEST")
log "launching $NJOBS repair jobs parallel=$PARALLEL"
RETRIEVE_LOG="$LOGDIR/retrieve_h2_finish_${TS}.log"
export TS PARALLEL
bash examples/benchmark/run_p5plus_retrieve_h2_repair_from_manifest.sh "$MANIFEST" \
  >>"$RETRIEVE_LOG" 2>&1

log "=== retrieve H2 finish done — see $RETRIEVE_LOG ==="
