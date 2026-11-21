#!/bin/bash
# Quick progress snapshot for real_hpo_v2 runs.
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
BASE="${BASE_DIR:-$REPO/benchmark_data/real_hpo_v2}"
N_TRIALS="${N_TRIALS:-80}"

export PYTHONPATH="$REPO/examples/benchmark:${PYTHONPATH:-}"
PYTHON="${PYTHON:-/root/miniconda3/envs/openbox_eval/bin/python}"

echo "=== HPO v2 progress ($(date)) ==="
echo "base=$BASE  target_trials=$N_TRIALS"
echo ""

pgrep -af "run_option2_v2|run_toy_categorical" 2>/dev/null | grep -v pgrep | head -3 || echo "(no active runner)"
echo ""

LOG=$(ls -t "$BASE"/logs/baselines_*.log "$BASE"/logs/*.log 2>/dev/null | head -1 || true)
if [[ -n "${LOG:-}" ]]; then
  echo "Latest log: $LOG"
  tail -4 "$LOG"
  echo ""
fi

"$PYTHON" - <<'PY' "$BASE" "$N_TRIALS"
import csv, sys
from pathlib import Path

base = Path(sys.argv[1])
n_trials = int(sys.argv[2])
complete, partial = [], []

for task_type in ("cls", "rgs"):
    root = base / task_type
    if not root.is_dir():
        continue
    for ds in sorted(root.iterdir()):
        if not ds.is_dir():
            continue
        for opt in sorted(ds.iterdir()):
            if not opt.is_dir():
                continue
            trajs = sorted(opt.glob("CASH-*/*_traj.csv"))
            if not trajs:
                continue
            rows = list(csv.DictReader(trajs[-1].open()))
            n = len(rows)
            key = f"{task_type}/{ds.name}/{opt.name}"
            if n >= n_trials:
                complete.append(key)
            else:
                wall = float(rows[-1].get("wall_elapsed_sec") or 0) if rows else 0
                partial.append((key, n, wall))

print(f"Complete runs: {len(complete)}/{len(complete)+len(partial)} started")
if partial:
    print("In progress:")
    for key, n, w in sorted(partial, key=lambda x: -x[2])[:8]:
        print(f"  {key}: {n}/{n_trials} trials ({w/60:.1f} min)")
    if len(partial) > 8:
        print(f"  ... +{len(partial)-8} more partial")
if complete:
    print("Recently finished:")
    for key in complete[-5:]:
        print(f"  ✓ {key}")
PY
