#!/usr/bin/env bash
# After the D1/D2/D3/O1/O4 train+eval pipeline finishes:
#   1) resume incomplete conservative arms (D1/D2/D3/O1/O4)
#   2) resume older s1a_s2_C gaps (colleges + jasmine were never run)
# EVAL_RESUME=1 skips already-complete 80-trial runs.
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
PIPE_PAT="${PIPE_PAT:-wait_s1b_rationale_then_s1a_s2_D1D2_D3O1O4_train_eval.sh}"
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-4}"
EVAL_JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-3}"
TIME_LIMIT="${TIME_LIMIT:-10800}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/s1a_s2_resume_incomplete_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

wait_pipeline() {
  log "WAIT until no process matches: $PIPE_PAT"
  while pgrep -f "$PIPE_PAT" >/dev/null 2>&1; do
    sleep 60
  done
  log "WAIT clear — primary pipeline finished"
}

scan_incomplete() {
  cd "$REPO"
  python3 - <<'PY'
import json
from pathlib import Path

REPO = Path(".")
TASKS = json.loads((REPO / "benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40.json").read_text())
VARIANT = "ptllm_tpe_qwen__P5__H0"
models = [
    "s1a_s2_D1",
    "s1a_s2_D2",
    "s1a_s2_D3",
    "s1a_s2_O1",
    "s1a_s2_O4",
    "s1a_s2_C",  # older first-round DPO; missing colleges + jasmine
    "s1a_s2_G",
]

def best_n(root, tt, ds):
    d = root / tt / ds / VARIANT
    if not d.is_dir():
        return 0
    nmax = 0
    for run in d.glob("CASH-*"):
        trajs = sorted(run.glob("*_traj.csv"))
        if not trajs:
            continue
        n = sum(1 for _ in open(trajs[-1])) - 1
        nmax = max(nmax, n)
    return nmax

total_bad = 0
for m in models:
    root = REPO / f"benchmark_data/real_hpo_v2_mltest20_algo40_p5_{m}"
    bad = []
    for t in TASKS:
        n = best_n(root, t["task_type"], t["dataset"]) if root.is_dir() else 0
        if n < 80:
            bad.append(f"{t['task_type']}/{t['dataset']}/{t['algorithm']}={n}")
    print(f"{m}: complete={40 - len(bad)}/40 incomplete={len(bad)}")
    for b in bad:
        print(f"  {b}")
    total_bad += len(bad)
print(f"TOTAL_INCOMPLETE={total_bad}")
PY
}

run_stage() {
  local stage="$1"
  log "EVAL resume STAGE=$stage TIME_LIMIT=${TIME_LIMIT}s NUM_GPUS=$EVAL_NUM_GPUS"
  cd "$REPO"
  NUM_GPUS="$EVAL_NUM_GPUS" JOBS_PER_GPU="$EVAL_JOBS_PER_GPU" \
    STAGE="$stage" HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen \
    EVAL_RESUME=1 TIME_LIMIT="$TIME_LIMIT" \
    bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1
}

log "=== resume incomplete s1a_s2 evals (conservative + older C) ==="
log "Note: historical log '(77/80)' was queue progress (job 77 of 80), not a 77-trial traj."
log "Real older gap: s1a_s2_C missing rgs/colleges + cls/jasmine."
wait_pipeline

log "Pre-resume coverage scan:"
scan_incomplete | tee -a "$PIPE"

run_stage s1a_s2_conservative
run_stage s1a_s2_c

log "EVAL s1b rationale DPO C+G (rationale_json, P5/H0)"
run_stage s1b_dpo_rationale_cg

log "Post-resume coverage scan:"
scan_incomplete | tee -a "$PIPE"
log "=== resume + rationale eval complete ==="
#!/usr/bin/env bash
# After the D1/D2/D3/O1/O4 train+eval pipeline finishes:
#   1) resume incomplete conservative arms (D1/D2/D3/O1/O4)
#   2) resume older s1a_s2_C gaps (colleges + jasmine were never run)
# EVAL_RESUME=1 skips already-complete 80-trial runs.
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
PIPE_PAT="${PIPE_PAT:-wait_s1b_rationale_then_s1a_s2_D1D2_D3O1O4_train_eval.sh}"
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-4}"
EVAL_JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-3}"
TIME_LIMIT="${TIME_LIMIT:-10800}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/s1a_s2_resume_incomplete_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

wait_pipeline() {
  log "WAIT until no process matches: $PIPE_PAT"
  while pgrep -f "$PIPE_PAT" >/dev/null 2>&1; do
    sleep 60
  done
  log "WAIT clear — primary pipeline finished"
}

scan_incomplete() {
  cd "$REPO"
  python3 - <<'PY'
import json
from pathlib import Path

REPO = Path(".")
TASKS = json.loads((REPO / "benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40.json").read_text())
VARIANT = "ptllm_tpe_qwen__P5__H0"
models = [
    "s1a_s2_D1",
    "s1a_s2_D2",
    "s1a_s2_D3",
    "s1a_s2_O1",
    "s1a_s2_O4",
    "s1a_s2_C",  # older first-round DPO; missing colleges + jasmine
    "s1a_s2_G",
]

def best_n(root, tt, ds):
    d = root / tt / ds / VARIANT
    if not d.is_dir():
        return 0
    nmax = 0
    for run in d.glob("CASH-*"):
        trajs = sorted(run.glob("*_traj.csv"))
        if not trajs:
            continue
        n = sum(1 for _ in open(trajs[-1])) - 1
        nmax = max(nmax, n)
    return nmax

total_bad = 0
for m in models:
    root = REPO / f"benchmark_data/real_hpo_v2_mltest20_algo40_p5_{m}"
    bad = []
    for t in TASKS:
        n = best_n(root, t["task_type"], t["dataset"]) if root.is_dir() else 0
        if n < 80:
            bad.append(f"{t['task_type']}/{t['dataset']}/{t['algorithm']}={n}")
    print(f"{m}: complete={40 - len(bad)}/40 incomplete={len(bad)}")
    for b in bad:
        print(f"  {b}")
    total_bad += len(bad)
print(f"TOTAL_INCOMPLETE={total_bad}")
PY
}

run_stage() {
  local stage="$1"
  log "EVAL resume STAGE=$stage TIME_LIMIT=${TIME_LIMIT}s NUM_GPUS=$EVAL_NUM_GPUS"
  cd "$REPO"
  NUM_GPUS="$EVAL_NUM_GPUS" JOBS_PER_GPU="$EVAL_JOBS_PER_GPU" \
    STAGE="$stage" HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen \
    EVAL_RESUME=1 TIME_LIMIT="$TIME_LIMIT" \
    bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1
}

log "=== resume incomplete s1a_s2 evals (conservative + older C) ==="
log "Note: historical log '(77/80)' was queue progress (job 77 of 80), not a 77-trial traj."
log "Real older gap: s1a_s2_C missing rgs/colleges + cls/jasmine."
wait_pipeline

log "Pre-resume coverage scan:"
scan_incomplete | tee -a "$PIPE"

run_stage s1a_s2_conservative
run_stage s1a_s2_c

log "EVAL s1b rationale DPO C+G (rationale_json, P5/H0)"
run_stage s1b_dpo_rationale_cg

log "Post-resume coverage scan:"
scan_incomplete | tee -a "$PIPE"
log "=== resume + rationale eval complete ==="
#!/usr/bin/env bash
# After the D1/D2/D3/O1/O4 train+eval pipeline finishes:
#   1) resume incomplete conservative arms (D1/D2/D3/O1/O4)
#   2) resume older s1a_s2_C gaps (colleges + jasmine were never run)
# EVAL_RESUME=1 skips already-complete 80-trial runs.
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
PIPE_PAT="${PIPE_PAT:-wait_s1b_rationale_then_s1a_s2_D1D2_D3O1O4_train_eval.sh}"
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-4}"
EVAL_JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-3}"
TIME_LIMIT="${TIME_LIMIT:-10800}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/s1a_s2_resume_incomplete_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

wait_pipeline() {
  log "WAIT until no process matches: $PIPE_PAT"
  while pgrep -f "$PIPE_PAT" >/dev/null 2>&1; do
    sleep 60
  done
  log "WAIT clear — primary pipeline finished"
}

scan_incomplete() {
  cd "$REPO"
  python3 - <<'PY'
import json
from pathlib import Path

REPO = Path(".")
TASKS = json.loads((REPO / "benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40.json").read_text())
VARIANT = "ptllm_tpe_qwen__P5__H0"
models = [
    "s1a_s2_D1",
    "s1a_s2_D2",
    "s1a_s2_D3",
    "s1a_s2_O1",
    "s1a_s2_O4",
    "s1a_s2_C",  # older first-round DPO; missing colleges + jasmine
    "s1a_s2_G",
]

def best_n(root, tt, ds):
    d = root / tt / ds / VARIANT
    if not d.is_dir():
        return 0
    nmax = 0
    for run in d.glob("CASH-*"):
        trajs = sorted(run.glob("*_traj.csv"))
        if not trajs:
            continue
        n = sum(1 for _ in open(trajs[-1])) - 1
        nmax = max(nmax, n)
    return nmax

total_bad = 0
for m in models:
    root = REPO / f"benchmark_data/real_hpo_v2_mltest20_algo40_p5_{m}"
    bad = []
    for t in TASKS:
        n = best_n(root, t["task_type"], t["dataset"]) if root.is_dir() else 0
        if n < 80:
            bad.append(f"{t['task_type']}/{t['dataset']}/{t['algorithm']}={n}")
    print(f"{m}: complete={40 - len(bad)}/40 incomplete={len(bad)}")
    for b in bad:
        print(f"  {b}")
    total_bad += len(bad)
print(f"TOTAL_INCOMPLETE={total_bad}")
PY
}

run_stage() {
  local stage="$1"
  log "EVAL resume STAGE=$stage TIME_LIMIT=${TIME_LIMIT}s NUM_GPUS=$EVAL_NUM_GPUS"
  cd "$REPO"
  NUM_GPUS="$EVAL_NUM_GPUS" JOBS_PER_GPU="$EVAL_JOBS_PER_GPU" \
    STAGE="$stage" HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen \
    EVAL_RESUME=1 TIME_LIMIT="$TIME_LIMIT" \
    bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1
}

log "=== resume incomplete s1a_s2 evals (conservative + older C) ==="
log "Note: historical log '(77/80)' was queue progress (job 77 of 80), not a 77-trial traj."
log "Real older gap: s1a_s2_C missing rgs/colleges + cls/jasmine."
wait_pipeline

log "Pre-resume coverage scan:"
scan_incomplete | tee -a "$PIPE"

run_stage s1a_s2_conservative
run_stage s1a_s2_c

log "EVAL s1b rationale DPO C+G (rationale_json, P5/H0)"
run_stage s1b_dpo_rationale_cg

log "Post-resume coverage scan:"
scan_incomplete | tee -a "$PIPE"
log "=== resume + rationale eval complete ==="
#!/usr/bin/env bash
# After the D1/D2/D3/O1/O4 train+eval pipeline finishes:
#   1) resume incomplete conservative arms (D1/D2/D3/O1/O4)
#   2) resume older s1a_s2_C gaps (colleges + jasmine were never run)
# EVAL_RESUME=1 skips already-complete 80-trial runs.
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
PIPE_PAT="${PIPE_PAT:-wait_s1b_rationale_then_s1a_s2_D1D2_D3O1O4_train_eval.sh}"
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-4}"
EVAL_JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-3}"
TIME_LIMIT="${TIME_LIMIT:-10800}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/s1a_s2_resume_incomplete_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

wait_pipeline() {
  log "WAIT until no process matches: $PIPE_PAT"
  while pgrep -f "$PIPE_PAT" >/dev/null 2>&1; do
    sleep 60
  done
  log "WAIT clear — primary pipeline finished"
}

scan_incomplete() {
  cd "$REPO"
  python3 - <<'PY'
import json
from pathlib import Path

REPO = Path(".")
TASKS = json.loads((REPO / "benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40.json").read_text())
VARIANT = "ptllm_tpe_qwen__P5__H0"
models = [
    "s1a_s2_D1",
    "s1a_s2_D2",
    "s1a_s2_D3",
    "s1a_s2_O1",
    "s1a_s2_O4",
    "s1a_s2_C",  # older first-round DPO; missing colleges + jasmine
    "s1a_s2_G",
]

def best_n(root, tt, ds):
    d = root / tt / ds / VARIANT
    if not d.is_dir():
        return 0
    nmax = 0
    for run in d.glob("CASH-*"):
        trajs = sorted(run.glob("*_traj.csv"))
        if not trajs:
            continue
        n = sum(1 for _ in open(trajs[-1])) - 1
        nmax = max(nmax, n)
    return nmax

total_bad = 0
for m in models:
    root = REPO / f"benchmark_data/real_hpo_v2_mltest20_algo40_p5_{m}"
    bad = []
    for t in TASKS:
        n = best_n(root, t["task_type"], t["dataset"]) if root.is_dir() else 0
        if n < 80:
            bad.append(f"{t['task_type']}/{t['dataset']}/{t['algorithm']}={n}")
    print(f"{m}: complete={40 - len(bad)}/40 incomplete={len(bad)}")
    for b in bad:
        print(f"  {b}")
    total_bad += len(bad)
print(f"TOTAL_INCOMPLETE={total_bad}")
PY
}

run_stage() {
  local stage="$1"
  log "EVAL resume STAGE=$stage TIME_LIMIT=${TIME_LIMIT}s NUM_GPUS=$EVAL_NUM_GPUS"
  cd "$REPO"
  NUM_GPUS="$EVAL_NUM_GPUS" JOBS_PER_GPU="$EVAL_JOBS_PER_GPU" \
    STAGE="$stage" HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen \
    EVAL_RESUME=1 TIME_LIMIT="$TIME_LIMIT" \
    bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1
}

log "=== resume incomplete s1a_s2 evals (conservative + older C) ==="
log "Note: historical log '(77/80)' was queue progress (job 77 of 80), not a 77-trial traj."
log "Real older gap: s1a_s2_C missing rgs/colleges + cls/jasmine."
wait_pipeline

log "Pre-resume coverage scan:"
scan_incomplete | tee -a "$PIPE"

run_stage s1a_s2_conservative
run_stage s1a_s2_c

log "EVAL s1b rationale DPO C+G (rationale_json, P5/H0)"
run_stage s1b_dpo_rationale_cg

log "Post-resume coverage scan:"
scan_incomplete | tee -a "$PIPE"
log "=== resume + rationale eval complete ==="
