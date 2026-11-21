#!/bin/bash
# Conservative Stage 2 launch: wait for Stage 1 pilot (ptllm) to finish, pick best prompt, run hybrid sweep.
#
# Stage 1 input:  5 pilot datasets × 8 prompts × ptllm × 80 trials
# Stage 2 output: 5 pilot datasets × 8 hybrids (H0–H7) × ptllm_tpe × winning prompt × 80 trials
#
# Usage:
#   bash examples/benchmark/launch_stage2_conservative.sh
#   bash examples/benchmark/launch_stage2_conservative.sh --dry-run
#   bash examples/benchmark/launch_stage2_conservative.sh --wait-until 01:00
#   bash examples/benchmark/launch_stage2_conservative.sh --wait-until 01:00 --dry-run
#   bash examples/benchmark/launch_stage2_conservative.sh --test-schedule 45
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

PYTHON="${PYTHON:-/root/miniconda3/envs/openbox_eval/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark_data/real_hpo_v2}"
N_TRIALS="${N_TRIALS:-80}"
LOG_DIR="${OUTPUT_DIR}/logs/stage2_pilot_$(date +%Y%m%d_%H%M%S)"
PROMPTS=(P0 P1 P2 P4 P5 P6 P7 P8)
PILOT_SPECS=("cls yeast" "cls adult" "cls covertype" "rgs boston" "rgs elevators")
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-6}"
FALLBACK_PROMPT="${FALLBACK_PROMPT:-}"
DRY_RUN=0
SCHEDULE_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1; SCHEDULE_ARGS+=(--dry-run) ;;
    --test-schedule)
      # handled below with next arg
      ;;
  esac
done

schedule_at() {
  local target="$1"
  local script="$REPO/examples/benchmark/launch_stage2_conservative.sh"
  local extra="${SCHEDULE_ARGS[*]:-}"
  local cmd="bash ${script}${extra:+ ${extra}}"
  if command -v at >/dev/null 2>&1; then
    echo "Scheduling Stage 2 launcher at $target"
    echo "  command: $cmd"
    echo "$cmd" | at "$target"
    atq || true
    exit 0
  fi
  local now target_epoch wait_sec sched_log
  now=$(date +%s)
  target_epoch=$(date -d "today ${target}" +%s 2>/dev/null || true)
  if [[ -z "${target_epoch:-}" ]] || (( target_epoch <= now )); then
    target_epoch=$(date -d "tomorrow ${target}" +%s)
  fi
  wait_sec=$(( target_epoch - now ))
  sched_log="${OUTPUT_DIR}/logs/stage2_schedule_$(date +%Y%m%d_%H%M%S).log"
  echo "Scheduling Stage 2 launcher at ${target} (sleep ${wait_sec}s) via nohup"
  echo "  command: $cmd"
  echo "  log: $sched_log"
  nohup bash -c "sleep ${wait_sec} && ${cmd}" > "$sched_log" 2>&1 &
  echo "  scheduler_pid=$!"
  exit 0
}

schedule_in_seconds() {
  local delay="${1:-45}"
  local script="$REPO/examples/benchmark/launch_stage2_conservative.sh"
  local extra="${SCHEDULE_ARGS[*]:-}"
  local cmd="bash ${script}${extra:+ ${extra}}"
  local sched_log="${OUTPUT_DIR}/logs/stage2_schedule_test_$(date +%Y%m%d_%H%M%S).log"
  local fire_at
  fire_at=$(date -d "+${delay} seconds" -Is 2>/dev/null || date -Is)
  echo "TEST SCHEDULE: fire in ${delay}s (~${fire_at})"
  echo "  command: $cmd"
  echo "  log: $sched_log"
  nohup bash -c "sleep ${delay} && ${cmd}" > "$sched_log" 2>&1 &
  echo "  scheduler_pid=$!"
  echo "  verify: tail -f $sched_log"
  exit 0
}

if [[ "${1:-}" == "--test-schedule" ]]; then
  schedule_in_seconds "${2:-45}"
fi

if [[ "${1:-}" == "--at" || "${1:-}" == "--wait-until" ]]; then
  schedule_at "${2:-01:00}"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  LOG_DIR="${OUTPUT_DIR}/logs/stage2_dryrun_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/launch.log"
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "[$(date -Is)] Stage 2 conservative launcher start"
echo "  output=$OUTPUT_DIR"
echo "  max_wait=${MAX_WAIT_HOURS}h"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "  mode=DRY_RUN (no wait loop, will not launch Stage 2)"
fi

stage1_complete() {
  "$PYTHON" - <<'PY' "$OUTPUT_DIR" "$N_TRIALS"
import csv, sys
from pathlib import Path

base = Path(sys.argv[1])
n = int(sys.argv[2])
prompts = ["P0", "P1", "P2", "P4", "P5", "P6", "P7", "P8"]
pilot = [
    ("cls", "yeast"), ("cls", "adult"), ("cls", "covertype"),
    ("rgs", "boston"), ("rgs", "elevators"),
]

def trials(path: Path) -> int:
    tr = sorted(path.glob("CASH-*/*_traj.csv"))
    return sum(1 for _ in csv.DictReader(tr[-1].open())) if tr else 0

ok = True
for tt, ds in pilot:
    for p in prompts:
        opt = base / tt / ds / f"ptllm__{p}"
        t = trials(opt) if opt.is_dir() else 0
        if t < n:
            print(f"INCOMPLETE {tt}/{ds}/ptllm__{p}: {t}/{n}")
            ok = False
if ok:
    print("STAGE1_READY")
sys.exit(0 if ok else 1)
PY
}

deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
if ! stage1_complete; then
  if [[ "$DRY_RUN" == "1" ]]; then
    echo ""
    echo "[$(date -Is)] DRY_RUN: Stage 1 not complete — would wait up to ${MAX_WAIT_HOURS}h, then abort if still incomplete"
  else
    while ! stage1_complete; do
      if (( $(date +%s) >= deadline )); then
        echo "[$(date -Is)] ERROR: Stage 1 pilot not complete after ${MAX_WAIT_HOURS}h — not launching Stage 2"
        exit 1
      fi
      echo "[$(date -Is)] Stage 1 not ready; sleeping 10m"
      sleep 600
    done
  fi
else
  echo "[$(date -Is)] Stage 1 pilot not yet complete"
fi

echo ""
echo "[$(date -Is)] Picking best prompt (ptllm, avg_rank on 5 pilot datasets)"
PICK_LOG="$LOG_DIR/pick_winner.log"
"$PYTHON" examples/benchmark/pick_pilot_winner.py \
  --base-dir "$OUTPUT_DIR" --mode prompt --top 5 | tee "$PICK_LOG"

PROMPT="$("$PYTHON" - <<PY "$OUTPUT_DIR" "$PYTHON"
import re, subprocess, sys
from pathlib import Path

base = Path(sys.argv[1])
py = sys.argv[2]
out = subprocess.check_output(
    [py, "examples/benchmark/pick_pilot_winner.py", "--base-dir", str(base), "--mode", "prompt", "--top", "1"],
    text=True,
    cwd="/root/data/cys/blockdata_archive/llmoptimizer_benchmark",
)
m = re.search(r"^\s+(P\d+):", out, re.M)
print(m.group(1) if m else "")
PY
)"

if [[ -z "$PROMPT" ]]; then
  if [[ -n "$FALLBACK_PROMPT" ]]; then
    PROMPT="$FALLBACK_PROMPT"
    echo "[$(date -Is)] WARN: pick_winner failed; using FALLBACK_PROMPT=$PROMPT"
  else
    echo "[$(date -Is)] ERROR: could not determine winning prompt" >&2
    exit 1
  fi
fi

echo ""
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[$(date -Is)] DRY_RUN: would launch Stage 2 pilot"
  echo "  prompt=$PROMPT"
  echo "  optimizer=ptllm_tpe"
  echo "  hybrids=H0 H1 H2 H3 H4 H5 H6 H7"
  echo "  scope=pilot (5 datasets)"
  echo "  runs=5 datasets × 8 hybrids = 40 runs × ${N_TRIALS} trials"
  echo ""
  echo "Command:"
  echo "  nohup env STAGE=stage2_pilot PROMPT=$PROMPT OUTPUT_DIR=$OUTPUT_DIR \\"
  echo "    bash examples/benchmark/run_option2_v2.sh > $LOG_DIR/stage2_pilot.log 2>&1 &"
  echo "[$(date -Is)] DRY_RUN done (nothing launched)"
  exit 0
fi

echo "[$(date -Is)] Launching Stage 2 pilot: ptllm_tpe × prompt=$PROMPT × hybrids H0–H7 × 5 pilot datasets"
nohup env STAGE=stage2_pilot PROMPT="$PROMPT" OUTPUT_DIR="$OUTPUT_DIR" \
  bash examples/benchmark/run_option2_v2.sh \
  > "$LOG_DIR/stage2_pilot.log" 2>&1 &
echo "  stage2_pid=$!"
echo "  monitor: tail -f $LOG_DIR/stage2_pilot.log"
echo "[$(date -Is)] done"
