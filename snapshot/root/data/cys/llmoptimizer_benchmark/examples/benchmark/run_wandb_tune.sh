#!/usr/bin/env bash
# Create a W&B sweep and optionally launch agent(s) for llmoptimizer REINFORCE tuning.
#
# Usage:
#   bash examples/benchmark/run_wandb_tune.sh create
#   bash examples/benchmark/run_wandb_tune.sh agent <sweep_id>
#   bash examples/benchmark/run_wandb_tune.sh run          # create + 1 agent
#   COUNT=4 bash examples/benchmark/run_wandb_tune.sh run  # create + 4 agents
#
# Env overrides:
#   WANDB_PROJECT=llmoptimizer
#   WANDB_ENTITY=cys36-nanyang-technological-university-singapore
#   CUDA_DEVICE=cuda:0
#   INIT_ADAPTER=/path/to/s1a_ep7p5
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

WANDB_PROJECT="${WANDB_PROJECT:-llmoptimizer}"
WANDB_ENTITY="${WANDB_ENTITY:-cys36-nanyang-technological-university-singapore}"
SWEEP_YAML="${SWEEP_YAML:-examples/benchmark/wandb/sweep_reinforce_hpo.yaml}"
PYTHON="${PYTHON_TRAIN:-/root/miniconda3/envs/llamafactory_qwen35/bin/python}"
COUNT="${COUNT:-1}"
CMD="${1:-run}"

log() { echo "[$(date -Is)] $*"; }

create_sweep() {
  log "Creating sweep in project=$WANDB_PROJECT entity=$WANDB_ENTITY"
  "$PYTHON" -m wandb sweep "$SWEEP_YAML" --project "$WANDB_PROJECT" --entity "$WANDB_ENTITY"
}

run_agent() {
  local sweep_id="$1"
  local target="${WANDB_ENTITY}/${WANDB_PROJECT}/${sweep_id}"
  log "Starting wandb agent -> $target (CUDA_DEVICE=${CUDA_DEVICE:-cuda:0})"
  CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}" "$PYTHON" -m wandb agent "$target"
}

case "$CMD" in
  create)
    create_sweep
    ;;
  agent)
    if [[ -z "${2:-}" ]]; then
      echo "Usage: $0 agent <sweep_id>" >&2
      exit 1
    fi
    run_agent "$2"
    ;;
  run)
    out="$(create_sweep 2>&1 | tee /dev/stderr)"
    sweep_id="$(echo "$out" | sed -n 's/.* sweep with ID: \([^ ]*\).*/\1/p' | tail -1)"
    if [[ -z "$sweep_id" ]]; then
      echo "Could not parse sweep id from wandb output" >&2
      exit 1
    fi
    log "Parsed sweep_id=$sweep_id; launching $COUNT agent(s)"
    for ((i = 0; i < COUNT; i++)); do
      run_agent "$sweep_id" &
    done
    wait
    ;;
  *)
    echo "Unknown command: $CMD (use create | agent | run)" >&2
    exit 1
    ;;
esac
