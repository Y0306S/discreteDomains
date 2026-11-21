#!/usr/bin/env bash
# After the resume+rationale waiter finishes, run s1a_ep7p5_r3 (same adapter, SEED=3).
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
WAIT_PAT="${WAIT_PAT:-wait_then_resume_s1a_s2_conservative_incomplete.sh}"
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-4}"
EVAL_JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-3}"
TIME_LIMIT="${TIME_LIMIT:-10800}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/s1a_ep7p5_tpe_r3_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

log "=== wait then eval s1a_ep7p5_r3 (SEED=3) ==="
log "WAIT until no process matches: $WAIT_PAT"
while pgrep -f "$WAIT_PAT" >/dev/null 2>&1; do
  sleep 60
done
log "WAIT clear — starting STAGE=s1a_ep7p5_tpe_r3 SEED=3"

cd "$REPO"
NUM_GPUS="$EVAL_NUM_GPUS" JOBS_PER_GPU="$EVAL_JOBS_PER_GPU" \
  STAGE=s1a_ep7p5_tpe_r3 HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen \
  SEED=3 TIME_LIMIT="$TIME_LIMIT" \
  bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1

log "=== s1a_ep7p5_r3 eval complete ==="
#!/usr/bin/env bash
# After the resume+rationale waiter finishes, run s1a_ep7p5_r3 (same adapter, SEED=3).
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
WAIT_PAT="${WAIT_PAT:-wait_then_resume_s1a_s2_conservative_incomplete.sh}"
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-4}"
EVAL_JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-3}"
TIME_LIMIT="${TIME_LIMIT:-10800}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/s1a_ep7p5_tpe_r3_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

log "=== wait then eval s1a_ep7p5_r3 (SEED=3) ==="
log "WAIT until no process matches: $WAIT_PAT"
while pgrep -f "$WAIT_PAT" >/dev/null 2>&1; do
  sleep 60
done
log "WAIT clear — starting STAGE=s1a_ep7p5_tpe_r3 SEED=3"

cd "$REPO"
NUM_GPUS="$EVAL_NUM_GPUS" JOBS_PER_GPU="$EVAL_JOBS_PER_GPU" \
  STAGE=s1a_ep7p5_tpe_r3 HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen \
  SEED=3 TIME_LIMIT="$TIME_LIMIT" \
  bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1

log "=== s1a_ep7p5_r3 eval complete ==="
#!/usr/bin/env bash
# After the resume+rationale waiter finishes, run s1a_ep7p5_r3 (same adapter, SEED=3).
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
WAIT_PAT="${WAIT_PAT:-wait_then_resume_s1a_s2_conservative_incomplete.sh}"
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-4}"
EVAL_JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-3}"
TIME_LIMIT="${TIME_LIMIT:-10800}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/s1a_ep7p5_tpe_r3_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

log "=== wait then eval s1a_ep7p5_r3 (SEED=3) ==="
log "WAIT until no process matches: $WAIT_PAT"
while pgrep -f "$WAIT_PAT" >/dev/null 2>&1; do
  sleep 60
done
log "WAIT clear — starting STAGE=s1a_ep7p5_tpe_r3 SEED=3"

cd "$REPO"
NUM_GPUS="$EVAL_NUM_GPUS" JOBS_PER_GPU="$EVAL_JOBS_PER_GPU" \
  STAGE=s1a_ep7p5_tpe_r3 HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen \
  SEED=3 TIME_LIMIT="$TIME_LIMIT" \
  bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1

log "=== s1a_ep7p5_r3 eval complete ==="
#!/usr/bin/env bash
# After the resume+rationale waiter finishes, run s1a_ep7p5_r3 (same adapter, SEED=3).
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
WAIT_PAT="${WAIT_PAT:-wait_then_resume_s1a_s2_conservative_incomplete.sh}"
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-4}"
EVAL_JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-3}"
TIME_LIMIT="${TIME_LIMIT:-10800}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/s1a_ep7p5_tpe_r3_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

log "=== wait then eval s1a_ep7p5_r3 (SEED=3) ==="
log "WAIT until no process matches: $WAIT_PAT"
while pgrep -f "$WAIT_PAT" >/dev/null 2>&1; do
  sleep 60
done
log "WAIT clear — starting STAGE=s1a_ep7p5_tpe_r3 SEED=3"

cd "$REPO"
NUM_GPUS="$EVAL_NUM_GPUS" JOBS_PER_GPU="$EVAL_JOBS_PER_GPU" \
  STAGE=s1a_ep7p5_tpe_r3 HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen \
  SEED=3 TIME_LIMIT="$TIME_LIMIT" \
  bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1

log "=== s1a_ep7p5_r3 eval complete ==="
