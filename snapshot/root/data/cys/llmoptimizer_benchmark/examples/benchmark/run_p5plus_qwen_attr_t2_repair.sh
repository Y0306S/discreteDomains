#!/usr/bin/env bash
# Repair Qwen s1b_attr_t2 P5+ algo40 slots that have 80 trials of the WRONG algorithm
# (shared dataset/variant dir collision). Does NOT re-run the full sweep.
#
# For each needed (config, dataset, algorithm):
#   1) quarantine sibling-algo CASH runs so --resume does not skip
#   2) re-run only that job
#   3) restore quarantined siblings
#
#   bash examples/benchmark/run_p5plus_qwen_attr_t2_repair.sh
#   NUM_GPUS=4 JOBS_PER_GPU=3 bash ...
#   DRY_RUN=1 bash ...
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

TASKS_JSON="${TASKS_JSON:-benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40.json}"
SUMMARY_CSV="${SUMMARY_CSV:-benchmark_data/real_hpo_v2_mltest20_algo40/analysis/p5plus/gpt_p5_p5plus_summary.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_data/real_hpo_v2_mltest20_algo40_p5_s1b_attr_t2_p5plus}"
LOGDIR="${LOGDIR:-benchmark_data/real_hpo_v2_mltest20_algo40/logs}"
PYTHON_LLM="${PYTHON_LLM:-/root/miniconda3/envs/openbox_qwen_eval/bin/python}"
PYTHON_SCAN="${PYTHON_SCAN:-/root/miniconda3/envs/openbox_eval/bin/python}"
BASE_MODEL="${BASE_MODEL:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
ADAPTER_PATH="${ADAPTER_PATH:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-7200}"
PER_RUN_TIME_LIMIT="${PER_RUN_TIME_LIMIT:-1800}"
DATASET_RETRIES="${DATASET_RETRIES:-3}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
POLL_SEC="${POLL_SEC:-15}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_PURE_PTLLM="${SKIP_PURE_PTLLM:-1}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="$LOGDIR/qwen_p5plus_attr_t2_repair_${TS}.log"
MANIFEST="$LOGDIR/qwen_attr_t2_repair_manifest_${TS}.txt"
PIDFILE="$LOGDIR/qwen_p5plus_attr_t2_repair.pid"

mkdir -p "$LOGDIR" "$OUTPUT_ROOT"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"
echo $$ >"$PIDFILE"

log() { echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"; }

log "=== Qwen attr_t2 P5+ wrong-algo repair ==="
log "output=$OUTPUT_ROOT gpus=$NUM_GPUS jobs_per_gpu=$JOBS_PER_GPU"

log "scanning repair slots → $MANIFEST"
"$PYTHON_SCAN" - "$MANIFEST" "$OUTPUT_ROOT" "$TASKS_JSON" "$SUMMARY_CSV" "$SKIP_PURE_PTLLM" <<'PY' | tee -a "$MASTER_LOG"
import csv, sys
from collections import Counter
from pathlib import Path

manifest_path = Path(sys.argv[1])
root = Path(sys.argv[2])
tasks = __import__("json").loads(Path(sys.argv[3]).read_text())
summary_csv = Path(sys.argv[4])
skip_pure = sys.argv[5] == "1"

def parse_gpt_variant(v: str):
    parts = v.split("__")
    opt = parts[0]
    prompt = parts[1] if len(parts) > 1 else ""
    hybrid = parts[2] if len(parts) > 2 else ""
    return opt, prompt, hybrid

def traj_mode(traj: Path):
    rows = list(csv.DictReader(traj.open(newline="", encoding="utf-8")))
    algos = [str(r.get("algorithm") or "").strip() for r in rows if str(r.get("algorithm") or "").strip()]
    if not algos:
        return None, len(rows)
    return Counter(algos).most_common(1)[0][0], len(rows)

wanted = []
for row in csv.DictReader(summary_csv.open()):
    gv = row["variant"].strip()
    if skip_pure and gv.startswith("ptllm__") and not gv.startswith("ptllm_tpe"):
        continue
    opt, prompt, hybrid = parse_gpt_variant(gv)
    q_opt = opt.replace("ptllm_tpe", "ptllm_tpe_qwen", 1) if opt.startswith("ptllm_tpe") else opt.replace("ptllm", "ptllm_qwen", 1)
    disk = f"{q_opt}__{prompt}" + (f"__{hybrid}" if hybrid else "")
    wanted.append((q_opt, prompt, hybrid, disk, gv))

needs = []
for q_opt, prompt, hybrid, disk, gv in wanted:
    for t in tasks:
        tt, ds, algo = t["task_type"], t["dataset"], t["algorithm"]
        opt_dir = root / tt / ds / disk
        good = False
        if opt_dir.is_dir():
            for run in sorted(opt_dir.glob("CASH-*")):
                trajs = sorted(run.glob("*_traj.csv"))
                if not trajs:
                    continue
                mode, n = traj_mode(trajs[-1])
                if mode == algo and n >= 80:
                    good = True
                    break
        if good:
            continue
        needs.append(f"{q_opt}|{prompt}|{hybrid}|{tt}|{ds}|{algo}")

manifest_path.write_text("\n".join(needs) + ("\n" if needs else ""), encoding="utf-8")
print(f"repair_jobs={len(needs)}")
by = Counter(x.split("|")[0] + "__" + x.split("|")[1] + (("__" + x.split("|")[2]) if x.split("|")[2] else "") for x in needs)
for k, n in by.most_common(12):
    print(f"  {n:3d}  {k}")
if len(by) > 12:
    print(f"  ... {len(by)} configs total")
PY

if [[ ! -s "$MANIFEST" ]]; then
  log "nothing to repair — all TPE arms algo-correct"
  exit 0
fi

NJOBS=$(grep -cve '^$' "$MANIFEST" || true)
log "launching $NJOBS repair jobs"

if [[ "$DRY_RUN" == "1" ]]; then
  log "DRY_RUN: manifest only at $MANIFEST"
  exit 0
fi

gpu_free_mb() {
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}
is_alive() { [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null; }

quarantine_siblings() {
  local opt_dir="$1" algo="$2" qdir="$3"
  mkdir -p "$qdir"
  [[ -d "$opt_dir" ]] || return 0
  local run trajs mode
  for run in "$opt_dir"/CASH-*; do
    [[ -d "$run" ]] || continue
    trajs=("$run"/*_traj.csv)
    if [[ ! -e "${trajs[0]:-}" ]]; then
      mv "$run" "$qdir/"
      continue
    fi
    mode="$("$PYTHON_SCAN" - "$run" <<'PY'
import csv, sys
from collections import Counter
from pathlib import Path
run = Path(sys.argv[1])
trajs = sorted(run.glob("*_traj.csv"))
if not trajs:
    print("")
    raise SystemExit
rows = list(csv.DictReader(trajs[-1].open(newline="", encoding="utf-8")))
algos = [str(r.get("algorithm") or "").strip() for r in rows if str(r.get("algorithm") or "").strip()]
print(Counter(algos).most_common(1)[0][0] if algos else "")
PY
)"
    if [[ "$mode" != "$algo" ]]; then
      mv "$run" "$qdir/"
    fi
  done
}

restore_quarantine() {
  local opt_dir="$1" qdir="$2"
  [[ -d "$qdir" ]] || return 0
  mkdir -p "$opt_dir"
  local item
  for item in "$qdir"/*; do
    [[ -e "$item" ]] || continue
    mv "$item" "$opt_dir/"
  done
  rmdir "$qdir" 2>/dev/null || true
}

run_one_repair() {
  local opt_base="$1" prompt="$2" hybrid="$3" tt="$4" ds="$5" algo="$6" gpu="$7"
  local tag="${opt_base}__${prompt}"
  [[ -n "$hybrid" ]] && tag="${tag}__${hybrid}"
  local opt_dir="$OUTPUT_ROOT/$tt/$ds/$tag"
  local qdir="$OUTPUT_ROOT/$tt/$ds/.quarantine_repair_${algo}_${TS}"
  local logf="$LOGDIR/qwen_attr_t2_repair_${tag}_${ds}_${algo}_${TS}.log"
  local runner

  log "  gpu=$gpu repair $tag $tt/$ds/$algo"
  quarantine_siblings "$opt_dir" "$algo" "$qdir" >>"$logf" 2>&1 || true

  runner=( "$PYTHON_LLM" examples/benchmark/run_toy_categorical_optimizers.py
    --algorithm "$algo"
    --n-trials "$N_TRIALS" --seed "$SEED" --skip-test-eval
    --llm-backends qwen-sft6 --qwen-backend hf
    --qwen-model-path "$BASE_MODEL"
    --qwen-adapter-path "$ADAPTER_PATH"
    --llm-max-tokens 768
    --llm-response-format rationale_json
    --output-dir "$OUTPUT_ROOT"
    --all-datasets --include-datasets "$ds"
    --time-limit "$TIME_LIMIT"
    --per-run-time-limit "$PER_RUN_TIME_LIMIT"
    --dataset-retries "$DATASET_RETRIES"
    --optimizer "$opt_base"
    --prompt-variant "$prompt"
    --resume
  )
  if [[ -n "$hybrid" ]]; then
    runner+=( --hybrid-variant "$hybrid" )
  fi

  set +e
  CUDA_VISIBLE_DEVICES="$gpu" "${runner[@]}" >>"$logf" 2>&1
  local rc=$?
  set -e
  restore_quarantine "$opt_dir" "$qdir" >>"$logf" 2>&1 || true
  if [[ $rc -ne 0 ]]; then
    log "  WARN repair rc=$rc $tag $ds/$algo (see $logf)"
  fi
  return 0
}

mapfile -t JOBS < "$MANIFEST"
declare -A SLOT_PID=()
declare -A SLOT_KEY=()  # serialize same dataset+variant
queue_idx=0
total=${#JOBS[@]}

log "repair queue: $total jobs"

while [[ $queue_idx -lt $total ]] || {
    any=0
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
      for ((slot=0; slot<JOBS_PER_GPU; slot++)); do
        is_alive "${SLOT_PID[${gpu}_${slot}]:-}" && any=1
      done
    done
    [[ $any -eq 1 ]]
  }; do
  for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    for ((slot=0; slot<JOBS_PER_GPU; slot++)); do
      pid="${SLOT_PID[${gpu}_${slot}]:-}"
      if is_alive "$pid"; then continue; fi
      SLOT_PID[${gpu}_${slot}]=
      SLOT_KEY[${gpu}_${slot}]=
      [[ $queue_idx -ge $total ]] && continue

      count=0
      for ((s=0; s<JOBS_PER_GPU; s++)); do
        is_alive "${SLOT_PID[${gpu}_${s}]:-}" && count=$((count+1))
      done
      [[ $count -lt $JOBS_PER_GPU ]] || continue
      if [[ $count -ge 1 ]]; then
        free=$(gpu_free_mb "$gpu")
        [[ "${free:-0}" -ge $PACK_MIN_FREE_MB ]] || continue
      fi

      # pick next job whose (tt,ds,tag) is not already running
      found=-1
      for ((j=queue_idx; j<total; j++)); do
        IFS='|' read -r opt_base prompt hybrid tt ds algo <<<"${JOBS[$j]}"
        tag="${opt_base}__${prompt}"
        [[ -n "$hybrid" ]] && tag="${tag}__${hybrid}"
        key="$tt|$ds|$tag"
        busy=0
        for ((g=0; g<NUM_GPUS; g++)); do
          for ((s2=0; s2<JOBS_PER_GPU; s2++)); do
            if is_alive "${SLOT_PID[${g}_${s2}]:-}" && [[ "${SLOT_KEY[${g}_${s2}]:-}" == "$key" ]]; then
              busy=1
            fi
          done
        done
        [[ $busy -eq 0 ]] || continue
        found=$j
        break
      done
      [[ $found -ge 0 ]] || continue

      # swap found into queue_idx position
      if [[ $found -ne $queue_idx ]]; then
        tmp="${JOBS[$queue_idx]}"
        JOBS[$queue_idx]="${JOBS[$found]}"
        JOBS[$found]="$tmp"
      fi

      IFS='|' read -r opt_base prompt hybrid tt ds algo <<<"${JOBS[$queue_idx]}"
      tag="${opt_base}__${prompt}"
      [[ -n "$hybrid" ]] && tag="${tag}__${hybrid}"
      key="$tt|$ds|$tag"
      queue_idx=$((queue_idx+1))
      log "dispatch repair $queue_idx/$total gpu=$gpu $tag $ds/$algo"
      ( run_one_repair "$opt_base" "$prompt" "$hybrid" "$tt" "$ds" "$algo" "$gpu" ) &
      SLOT_PID[${gpu}_${slot}]=$!
      SLOT_KEY[${gpu}_${slot}]="$key"
    done
  done
  sleep "$POLL_SEC"
done
wait

log "=== Qwen attr_t2 P5+ repair done ==="
log "manifest=$MANIFEST master_log=$MASTER_LOG"

# quick coverage report
"$PYTHON_SCAN" - "$OUTPUT_ROOT" "$TASKS_JSON" "$SUMMARY_CSV" <<'PY' | tee -a "$MASTER_LOG"
import csv, json, sys
from collections import Counter
from pathlib import Path
from plot_mltest20_algo40_avg_rank import find_traj, load_tasks

root = Path(sys.argv[1])
tasks = load_tasks(Path(sys.argv[2]))
hist = Counter()
short = []
for row in csv.DictReader(Path(sys.argv[3]).open()):
    gv = row["variant"].strip()
    if gv.startswith("ptllm__") and not gv.startswith("ptllm_tpe"):
        continue
    disk = gv.replace("ptllm_tpe__", "ptllm_tpe_qwen__", 1).replace("ptllm__", "ptllm_qwen__", 1)
    n = sum(1 for t in tasks if find_traj(root, t["task_type"], t["dataset"], disk, 80, algorithm=t["algorithm"]))
    hist[n] += 1
    if n < 40:
        short.append((n, disk))
print("algo-correct coverage after repair:")
for k in sorted(hist, reverse=True):
    print(f"  {k}/40: {hist[k]} configs")
if short:
    print("still short:")
    for n, v in sorted(short)[:20]:
        print(f"  {n}/40  {v}")
PY
#!/usr/bin/env bash
# Repair Qwen s1b_attr_t2 P5+ algo40 slots that have 80 trials of the WRONG algorithm
# (shared dataset/variant dir collision). Does NOT re-run the full sweep.
#
# For each needed (config, dataset, algorithm):
#   1) quarantine sibling-algo CASH runs so --resume does not skip
#   2) re-run only that job
#   3) restore quarantined siblings
#
#   bash examples/benchmark/run_p5plus_qwen_attr_t2_repair.sh
#   NUM_GPUS=4 JOBS_PER_GPU=3 bash ...
#   DRY_RUN=1 bash ...
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

TASKS_JSON="${TASKS_JSON:-benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40.json}"
SUMMARY_CSV="${SUMMARY_CSV:-benchmark_data/real_hpo_v2_mltest20_algo40/analysis/p5plus/gpt_p5_p5plus_summary.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_data/real_hpo_v2_mltest20_algo40_p5_s1b_attr_t2_p5plus}"
LOGDIR="${LOGDIR:-benchmark_data/real_hpo_v2_mltest20_algo40/logs}"
PYTHON_LLM="${PYTHON_LLM:-/root/miniconda3/envs/openbox_qwen_eval/bin/python}"
PYTHON_SCAN="${PYTHON_SCAN:-/root/miniconda3/envs/openbox_eval/bin/python}"
BASE_MODEL="${BASE_MODEL:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
ADAPTER_PATH="${ADAPTER_PATH:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-7200}"
PER_RUN_TIME_LIMIT="${PER_RUN_TIME_LIMIT:-1800}"
DATASET_RETRIES="${DATASET_RETRIES:-3}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
POLL_SEC="${POLL_SEC:-15}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_PURE_PTLLM="${SKIP_PURE_PTLLM:-1}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="$LOGDIR/qwen_p5plus_attr_t2_repair_${TS}.log"
MANIFEST="$LOGDIR/qwen_attr_t2_repair_manifest_${TS}.txt"
PIDFILE="$LOGDIR/qwen_p5plus_attr_t2_repair.pid"

mkdir -p "$LOGDIR" "$OUTPUT_ROOT"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"
echo $$ >"$PIDFILE"

log() { echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"; }

log "=== Qwen attr_t2 P5+ wrong-algo repair ==="
log "output=$OUTPUT_ROOT gpus=$NUM_GPUS jobs_per_gpu=$JOBS_PER_GPU"

log "scanning repair slots → $MANIFEST"
"$PYTHON_SCAN" - "$MANIFEST" "$OUTPUT_ROOT" "$TASKS_JSON" "$SUMMARY_CSV" "$SKIP_PURE_PTLLM" <<'PY' | tee -a "$MASTER_LOG"
import csv, sys
from collections import Counter
from pathlib import Path

manifest_path = Path(sys.argv[1])
root = Path(sys.argv[2])
tasks = __import__("json").loads(Path(sys.argv[3]).read_text())
summary_csv = Path(sys.argv[4])
skip_pure = sys.argv[5] == "1"

def parse_gpt_variant(v: str):
    parts = v.split("__")
    opt = parts[0]
    prompt = parts[1] if len(parts) > 1 else ""
    hybrid = parts[2] if len(parts) > 2 else ""
    return opt, prompt, hybrid

def traj_mode(traj: Path):
    rows = list(csv.DictReader(traj.open(newline="", encoding="utf-8")))
    algos = [str(r.get("algorithm") or "").strip() for r in rows if str(r.get("algorithm") or "").strip()]
    if not algos:
        return None, len(rows)
    return Counter(algos).most_common(1)[0][0], len(rows)

wanted = []
for row in csv.DictReader(summary_csv.open()):
    gv = row["variant"].strip()
    if skip_pure and gv.startswith("ptllm__") and not gv.startswith("ptllm_tpe"):
        continue
    opt, prompt, hybrid = parse_gpt_variant(gv)
    q_opt = opt.replace("ptllm_tpe", "ptllm_tpe_qwen", 1) if opt.startswith("ptllm_tpe") else opt.replace("ptllm", "ptllm_qwen", 1)
    disk = f"{q_opt}__{prompt}" + (f"__{hybrid}" if hybrid else "")
    wanted.append((q_opt, prompt, hybrid, disk, gv))

needs = []
for q_opt, prompt, hybrid, disk, gv in wanted:
    for t in tasks:
        tt, ds, algo = t["task_type"], t["dataset"], t["algorithm"]
        opt_dir = root / tt / ds / disk
        good = False
        if opt_dir.is_dir():
            for run in sorted(opt_dir.glob("CASH-*")):
                trajs = sorted(run.glob("*_traj.csv"))
                if not trajs:
                    continue
                mode, n = traj_mode(trajs[-1])
                if mode == algo and n >= 80:
                    good = True
                    break
        if good:
            continue
        needs.append(f"{q_opt}|{prompt}|{hybrid}|{tt}|{ds}|{algo}")

manifest_path.write_text("\n".join(needs) + ("\n" if needs else ""), encoding="utf-8")
print(f"repair_jobs={len(needs)}")
by = Counter(x.split("|")[0] + "__" + x.split("|")[1] + (("__" + x.split("|")[2]) if x.split("|")[2] else "") for x in needs)
for k, n in by.most_common(12):
    print(f"  {n:3d}  {k}")
if len(by) > 12:
    print(f"  ... {len(by)} configs total")
PY

if [[ ! -s "$MANIFEST" ]]; then
  log "nothing to repair — all TPE arms algo-correct"
  exit 0
fi

NJOBS=$(grep -cve '^$' "$MANIFEST" || true)
log "launching $NJOBS repair jobs"

if [[ "$DRY_RUN" == "1" ]]; then
  log "DRY_RUN: manifest only at $MANIFEST"
  exit 0
fi

gpu_free_mb() {
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}
is_alive() { [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null; }

quarantine_siblings() {
  local opt_dir="$1" algo="$2" qdir="$3"
  mkdir -p "$qdir"
  [[ -d "$opt_dir" ]] || return 0
  local run trajs mode
  for run in "$opt_dir"/CASH-*; do
    [[ -d "$run" ]] || continue
    trajs=("$run"/*_traj.csv)
    if [[ ! -e "${trajs[0]:-}" ]]; then
      mv "$run" "$qdir/"
      continue
    fi
    mode="$("$PYTHON_SCAN" - "$run" <<'PY'
import csv, sys
from collections import Counter
from pathlib import Path
run = Path(sys.argv[1])
trajs = sorted(run.glob("*_traj.csv"))
if not trajs:
    print("")
    raise SystemExit
rows = list(csv.DictReader(trajs[-1].open(newline="", encoding="utf-8")))
algos = [str(r.get("algorithm") or "").strip() for r in rows if str(r.get("algorithm") or "").strip()]
print(Counter(algos).most_common(1)[0][0] if algos else "")
PY
)"
    if [[ "$mode" != "$algo" ]]; then
      mv "$run" "$qdir/"
    fi
  done
}

restore_quarantine() {
  local opt_dir="$1" qdir="$2"
  [[ -d "$qdir" ]] || return 0
  mkdir -p "$opt_dir"
  local item
  for item in "$qdir"/*; do
    [[ -e "$item" ]] || continue
    mv "$item" "$opt_dir/"
  done
  rmdir "$qdir" 2>/dev/null || true
}

run_one_repair() {
  local opt_base="$1" prompt="$2" hybrid="$3" tt="$4" ds="$5" algo="$6" gpu="$7"
  local tag="${opt_base}__${prompt}"
  [[ -n "$hybrid" ]] && tag="${tag}__${hybrid}"
  local opt_dir="$OUTPUT_ROOT/$tt/$ds/$tag"
  local qdir="$OUTPUT_ROOT/$tt/$ds/.quarantine_repair_${algo}_${TS}"
  local logf="$LOGDIR/qwen_attr_t2_repair_${tag}_${ds}_${algo}_${TS}.log"
  local runner

  log "  gpu=$gpu repair $tag $tt/$ds/$algo"
  quarantine_siblings "$opt_dir" "$algo" "$qdir" >>"$logf" 2>&1 || true

  runner=( "$PYTHON_LLM" examples/benchmark/run_toy_categorical_optimizers.py
    --algorithm "$algo"
    --n-trials "$N_TRIALS" --seed "$SEED" --skip-test-eval
    --llm-backends qwen-sft6 --qwen-backend hf
    --qwen-model-path "$BASE_MODEL"
    --qwen-adapter-path "$ADAPTER_PATH"
    --llm-max-tokens 768
    --llm-response-format rationale_json
    --output-dir "$OUTPUT_ROOT"
    --all-datasets --include-datasets "$ds"
    --time-limit "$TIME_LIMIT"
    --per-run-time-limit "$PER_RUN_TIME_LIMIT"
    --dataset-retries "$DATASET_RETRIES"
    --optimizer "$opt_base"
    --prompt-variant "$prompt"
    --resume
  )
  if [[ -n "$hybrid" ]]; then
    runner+=( --hybrid-variant "$hybrid" )
  fi

  set +e
  CUDA_VISIBLE_DEVICES="$gpu" "${runner[@]}" >>"$logf" 2>&1
  local rc=$?
  set -e
  restore_quarantine "$opt_dir" "$qdir" >>"$logf" 2>&1 || true
  if [[ $rc -ne 0 ]]; then
    log "  WARN repair rc=$rc $tag $ds/$algo (see $logf)"
  fi
  return 0
}

mapfile -t JOBS < "$MANIFEST"
declare -A SLOT_PID=()
declare -A SLOT_KEY=()  # serialize same dataset+variant
queue_idx=0
total=${#JOBS[@]}

log "repair queue: $total jobs"

while [[ $queue_idx -lt $total ]] || {
    any=0
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
      for ((slot=0; slot<JOBS_PER_GPU; slot++)); do
        is_alive "${SLOT_PID[${gpu}_${slot}]:-}" && any=1
      done
    done
    [[ $any -eq 1 ]]
  }; do
  for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    for ((slot=0; slot<JOBS_PER_GPU; slot++)); do
      pid="${SLOT_PID[${gpu}_${slot}]:-}"
      if is_alive "$pid"; then continue; fi
      SLOT_PID[${gpu}_${slot}]=
      SLOT_KEY[${gpu}_${slot}]=
      [[ $queue_idx -ge $total ]] && continue

      count=0
      for ((s=0; s<JOBS_PER_GPU; s++)); do
        is_alive "${SLOT_PID[${gpu}_${s}]:-}" && count=$((count+1))
      done
      [[ $count -lt $JOBS_PER_GPU ]] || continue
      if [[ $count -ge 1 ]]; then
        free=$(gpu_free_mb "$gpu")
        [[ "${free:-0}" -ge $PACK_MIN_FREE_MB ]] || continue
      fi

      # pick next job whose (tt,ds,tag) is not already running
      found=-1
      for ((j=queue_idx; j<total; j++)); do
        IFS='|' read -r opt_base prompt hybrid tt ds algo <<<"${JOBS[$j]}"
        tag="${opt_base}__${prompt}"
        [[ -n "$hybrid" ]] && tag="${tag}__${hybrid}"
        key="$tt|$ds|$tag"
        busy=0
        for ((g=0; g<NUM_GPUS; g++)); do
          for ((s2=0; s2<JOBS_PER_GPU; s2++)); do
            if is_alive "${SLOT_PID[${g}_${s2}]:-}" && [[ "${SLOT_KEY[${g}_${s2}]:-}" == "$key" ]]; then
              busy=1
            fi
          done
        done
        [[ $busy -eq 0 ]] || continue
        found=$j
        break
      done
      [[ $found -ge 0 ]] || continue

      # swap found into queue_idx position
      if [[ $found -ne $queue_idx ]]; then
        tmp="${JOBS[$queue_idx]}"
        JOBS[$queue_idx]="${JOBS[$found]}"
        JOBS[$found]="$tmp"
      fi

      IFS='|' read -r opt_base prompt hybrid tt ds algo <<<"${JOBS[$queue_idx]}"
      tag="${opt_base}__${prompt}"
      [[ -n "$hybrid" ]] && tag="${tag}__${hybrid}"
      key="$tt|$ds|$tag"
      queue_idx=$((queue_idx+1))
      log "dispatch repair $queue_idx/$total gpu=$gpu $tag $ds/$algo"
      ( run_one_repair "$opt_base" "$prompt" "$hybrid" "$tt" "$ds" "$algo" "$gpu" ) &
      SLOT_PID[${gpu}_${slot}]=$!
      SLOT_KEY[${gpu}_${slot}]="$key"
    done
  done
  sleep "$POLL_SEC"
done
wait

log "=== Qwen attr_t2 P5+ repair done ==="
log "manifest=$MANIFEST master_log=$MASTER_LOG"

# quick coverage report
"$PYTHON_SCAN" - "$OUTPUT_ROOT" "$TASKS_JSON" "$SUMMARY_CSV" <<'PY' | tee -a "$MASTER_LOG"
import csv, json, sys
from collections import Counter
from pathlib import Path
from plot_mltest20_algo40_avg_rank import find_traj, load_tasks

root = Path(sys.argv[1])
tasks = load_tasks(Path(sys.argv[2]))
hist = Counter()
short = []
for row in csv.DictReader(Path(sys.argv[3]).open()):
    gv = row["variant"].strip()
    if gv.startswith("ptllm__") and not gv.startswith("ptllm_tpe"):
        continue
    disk = gv.replace("ptllm_tpe__", "ptllm_tpe_qwen__", 1).replace("ptllm__", "ptllm_qwen__", 1)
    n = sum(1 for t in tasks if find_traj(root, t["task_type"], t["dataset"], disk, 80, algorithm=t["algorithm"]))
    hist[n] += 1
    if n < 40:
        short.append((n, disk))
print("algo-correct coverage after repair:")
for k in sorted(hist, reverse=True):
    print(f"  {k}/40: {hist[k]} configs")
if short:
    print("still short:")
    for n, v in sorted(short)[:20]:
        print(f"  {n}/40  {v}")
PY
#!/usr/bin/env bash
# Repair Qwen s1b_attr_t2 P5+ algo40 slots that have 80 trials of the WRONG algorithm
# (shared dataset/variant dir collision). Does NOT re-run the full sweep.
#
# For each needed (config, dataset, algorithm):
#   1) quarantine sibling-algo CASH runs so --resume does not skip
#   2) re-run only that job
#   3) restore quarantined siblings
#
#   bash examples/benchmark/run_p5plus_qwen_attr_t2_repair.sh
#   NUM_GPUS=4 JOBS_PER_GPU=3 bash ...
#   DRY_RUN=1 bash ...
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

TASKS_JSON="${TASKS_JSON:-benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40.json}"
SUMMARY_CSV="${SUMMARY_CSV:-benchmark_data/real_hpo_v2_mltest20_algo40/analysis/p5plus/gpt_p5_p5plus_summary.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_data/real_hpo_v2_mltest20_algo40_p5_s1b_attr_t2_p5plus}"
LOGDIR="${LOGDIR:-benchmark_data/real_hpo_v2_mltest20_algo40/logs}"
PYTHON_LLM="${PYTHON_LLM:-/root/miniconda3/envs/openbox_qwen_eval/bin/python}"
PYTHON_SCAN="${PYTHON_SCAN:-/root/miniconda3/envs/openbox_eval/bin/python}"
BASE_MODEL="${BASE_MODEL:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
ADAPTER_PATH="${ADAPTER_PATH:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-7200}"
PER_RUN_TIME_LIMIT="${PER_RUN_TIME_LIMIT:-1800}"
DATASET_RETRIES="${DATASET_RETRIES:-3}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
POLL_SEC="${POLL_SEC:-15}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_PURE_PTLLM="${SKIP_PURE_PTLLM:-1}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="$LOGDIR/qwen_p5plus_attr_t2_repair_${TS}.log"
MANIFEST="$LOGDIR/qwen_attr_t2_repair_manifest_${TS}.txt"
PIDFILE="$LOGDIR/qwen_p5plus_attr_t2_repair.pid"

mkdir -p "$LOGDIR" "$OUTPUT_ROOT"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"
echo $$ >"$PIDFILE"

log() { echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"; }

log "=== Qwen attr_t2 P5+ wrong-algo repair ==="
log "output=$OUTPUT_ROOT gpus=$NUM_GPUS jobs_per_gpu=$JOBS_PER_GPU"

log "scanning repair slots → $MANIFEST"
"$PYTHON_SCAN" - "$MANIFEST" "$OUTPUT_ROOT" "$TASKS_JSON" "$SUMMARY_CSV" "$SKIP_PURE_PTLLM" <<'PY' | tee -a "$MASTER_LOG"
import csv, sys
from collections import Counter
from pathlib import Path

manifest_path = Path(sys.argv[1])
root = Path(sys.argv[2])
tasks = __import__("json").loads(Path(sys.argv[3]).read_text())
summary_csv = Path(sys.argv[4])
skip_pure = sys.argv[5] == "1"

def parse_gpt_variant(v: str):
    parts = v.split("__")
    opt = parts[0]
    prompt = parts[1] if len(parts) > 1 else ""
    hybrid = parts[2] if len(parts) > 2 else ""
    return opt, prompt, hybrid

def traj_mode(traj: Path):
    rows = list(csv.DictReader(traj.open(newline="", encoding="utf-8")))
    algos = [str(r.get("algorithm") or "").strip() for r in rows if str(r.get("algorithm") or "").strip()]
    if not algos:
        return None, len(rows)
    return Counter(algos).most_common(1)[0][0], len(rows)

wanted = []
for row in csv.DictReader(summary_csv.open()):
    gv = row["variant"].strip()
    if skip_pure and gv.startswith("ptllm__") and not gv.startswith("ptllm_tpe"):
        continue
    opt, prompt, hybrid = parse_gpt_variant(gv)
    q_opt = opt.replace("ptllm_tpe", "ptllm_tpe_qwen", 1) if opt.startswith("ptllm_tpe") else opt.replace("ptllm", "ptllm_qwen", 1)
    disk = f"{q_opt}__{prompt}" + (f"__{hybrid}" if hybrid else "")
    wanted.append((q_opt, prompt, hybrid, disk, gv))

needs = []
for q_opt, prompt, hybrid, disk, gv in wanted:
    for t in tasks:
        tt, ds, algo = t["task_type"], t["dataset"], t["algorithm"]
        opt_dir = root / tt / ds / disk
        good = False
        if opt_dir.is_dir():
            for run in sorted(opt_dir.glob("CASH-*")):
                trajs = sorted(run.glob("*_traj.csv"))
                if not trajs:
                    continue
                mode, n = traj_mode(trajs[-1])
                if mode == algo and n >= 80:
                    good = True
                    break
        if good:
            continue
        needs.append(f"{q_opt}|{prompt}|{hybrid}|{tt}|{ds}|{algo}")

manifest_path.write_text("\n".join(needs) + ("\n" if needs else ""), encoding="utf-8")
print(f"repair_jobs={len(needs)}")
by = Counter(x.split("|")[0] + "__" + x.split("|")[1] + (("__" + x.split("|")[2]) if x.split("|")[2] else "") for x in needs)
for k, n in by.most_common(12):
    print(f"  {n:3d}  {k}")
if len(by) > 12:
    print(f"  ... {len(by)} configs total")
PY

if [[ ! -s "$MANIFEST" ]]; then
  log "nothing to repair — all TPE arms algo-correct"
  exit 0
fi

NJOBS=$(grep -cve '^$' "$MANIFEST" || true)
log "launching $NJOBS repair jobs"

if [[ "$DRY_RUN" == "1" ]]; then
  log "DRY_RUN: manifest only at $MANIFEST"
  exit 0
fi

gpu_free_mb() {
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}
is_alive() { [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null; }

quarantine_siblings() {
  local opt_dir="$1" algo="$2" qdir="$3"
  mkdir -p "$qdir"
  [[ -d "$opt_dir" ]] || return 0
  local run trajs mode
  for run in "$opt_dir"/CASH-*; do
    [[ -d "$run" ]] || continue
    trajs=("$run"/*_traj.csv)
    if [[ ! -e "${trajs[0]:-}" ]]; then
      mv "$run" "$qdir/"
      continue
    fi
    mode="$("$PYTHON_SCAN" - "$run" <<'PY'
import csv, sys
from collections import Counter
from pathlib import Path
run = Path(sys.argv[1])
trajs = sorted(run.glob("*_traj.csv"))
if not trajs:
    print("")
    raise SystemExit
rows = list(csv.DictReader(trajs[-1].open(newline="", encoding="utf-8")))
algos = [str(r.get("algorithm") or "").strip() for r in rows if str(r.get("algorithm") or "").strip()]
print(Counter(algos).most_common(1)[0][0] if algos else "")
PY
)"
    if [[ "$mode" != "$algo" ]]; then
      mv "$run" "$qdir/"
    fi
  done
}

restore_quarantine() {
  local opt_dir="$1" qdir="$2"
  [[ -d "$qdir" ]] || return 0
  mkdir -p "$opt_dir"
  local item
  for item in "$qdir"/*; do
    [[ -e "$item" ]] || continue
    mv "$item" "$opt_dir/"
  done
  rmdir "$qdir" 2>/dev/null || true
}

run_one_repair() {
  local opt_base="$1" prompt="$2" hybrid="$3" tt="$4" ds="$5" algo="$6" gpu="$7"
  local tag="${opt_base}__${prompt}"
  [[ -n "$hybrid" ]] && tag="${tag}__${hybrid}"
  local opt_dir="$OUTPUT_ROOT/$tt/$ds/$tag"
  local qdir="$OUTPUT_ROOT/$tt/$ds/.quarantine_repair_${algo}_${TS}"
  local logf="$LOGDIR/qwen_attr_t2_repair_${tag}_${ds}_${algo}_${TS}.log"
  local runner

  log "  gpu=$gpu repair $tag $tt/$ds/$algo"
  quarantine_siblings "$opt_dir" "$algo" "$qdir" >>"$logf" 2>&1 || true

  runner=( "$PYTHON_LLM" examples/benchmark/run_toy_categorical_optimizers.py
    --algorithm "$algo"
    --n-trials "$N_TRIALS" --seed "$SEED" --skip-test-eval
    --llm-backends qwen-sft6 --qwen-backend hf
    --qwen-model-path "$BASE_MODEL"
    --qwen-adapter-path "$ADAPTER_PATH"
    --llm-max-tokens 768
    --llm-response-format rationale_json
    --output-dir "$OUTPUT_ROOT"
    --all-datasets --include-datasets "$ds"
    --time-limit "$TIME_LIMIT"
    --per-run-time-limit "$PER_RUN_TIME_LIMIT"
    --dataset-retries "$DATASET_RETRIES"
    --optimizer "$opt_base"
    --prompt-variant "$prompt"
    --resume
  )
  if [[ -n "$hybrid" ]]; then
    runner+=( --hybrid-variant "$hybrid" )
  fi

  set +e
  CUDA_VISIBLE_DEVICES="$gpu" "${runner[@]}" >>"$logf" 2>&1
  local rc=$?
  set -e
  restore_quarantine "$opt_dir" "$qdir" >>"$logf" 2>&1 || true
  if [[ $rc -ne 0 ]]; then
    log "  WARN repair rc=$rc $tag $ds/$algo (see $logf)"
  fi
  return 0
}

mapfile -t JOBS < "$MANIFEST"
declare -A SLOT_PID=()
declare -A SLOT_KEY=()  # serialize same dataset+variant
queue_idx=0
total=${#JOBS[@]}

log "repair queue: $total jobs"

while [[ $queue_idx -lt $total ]] || {
    any=0
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
      for ((slot=0; slot<JOBS_PER_GPU; slot++)); do
        is_alive "${SLOT_PID[${gpu}_${slot}]:-}" && any=1
      done
    done
    [[ $any -eq 1 ]]
  }; do
  for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    for ((slot=0; slot<JOBS_PER_GPU; slot++)); do
      pid="${SLOT_PID[${gpu}_${slot}]:-}"
      if is_alive "$pid"; then continue; fi
      SLOT_PID[${gpu}_${slot}]=
      SLOT_KEY[${gpu}_${slot}]=
      [[ $queue_idx -ge $total ]] && continue

      count=0
      for ((s=0; s<JOBS_PER_GPU; s++)); do
        is_alive "${SLOT_PID[${gpu}_${s}]:-}" && count=$((count+1))
      done
      [[ $count -lt $JOBS_PER_GPU ]] || continue
      if [[ $count -ge 1 ]]; then
        free=$(gpu_free_mb "$gpu")
        [[ "${free:-0}" -ge $PACK_MIN_FREE_MB ]] || continue
      fi

      # pick next job whose (tt,ds,tag) is not already running
      found=-1
      for ((j=queue_idx; j<total; j++)); do
        IFS='|' read -r opt_base prompt hybrid tt ds algo <<<"${JOBS[$j]}"
        tag="${opt_base}__${prompt}"
        [[ -n "$hybrid" ]] && tag="${tag}__${hybrid}"
        key="$tt|$ds|$tag"
        busy=0
        for ((g=0; g<NUM_GPUS; g++)); do
          for ((s2=0; s2<JOBS_PER_GPU; s2++)); do
            if is_alive "${SLOT_PID[${g}_${s2}]:-}" && [[ "${SLOT_KEY[${g}_${s2}]:-}" == "$key" ]]; then
              busy=1
            fi
          done
        done
        [[ $busy -eq 0 ]] || continue
        found=$j
        break
      done
      [[ $found -ge 0 ]] || continue

      # swap found into queue_idx position
      if [[ $found -ne $queue_idx ]]; then
        tmp="${JOBS[$queue_idx]}"
        JOBS[$queue_idx]="${JOBS[$found]}"
        JOBS[$found]="$tmp"
      fi

      IFS='|' read -r opt_base prompt hybrid tt ds algo <<<"${JOBS[$queue_idx]}"
      tag="${opt_base}__${prompt}"
      [[ -n "$hybrid" ]] && tag="${tag}__${hybrid}"
      key="$tt|$ds|$tag"
      queue_idx=$((queue_idx+1))
      log "dispatch repair $queue_idx/$total gpu=$gpu $tag $ds/$algo"
      ( run_one_repair "$opt_base" "$prompt" "$hybrid" "$tt" "$ds" "$algo" "$gpu" ) &
      SLOT_PID[${gpu}_${slot}]=$!
      SLOT_KEY[${gpu}_${slot}]="$key"
    done
  done
  sleep "$POLL_SEC"
done
wait

log "=== Qwen attr_t2 P5+ repair done ==="
log "manifest=$MANIFEST master_log=$MASTER_LOG"

# quick coverage report
"$PYTHON_SCAN" - "$OUTPUT_ROOT" "$TASKS_JSON" "$SUMMARY_CSV" <<'PY' | tee -a "$MASTER_LOG"
import csv, json, sys
from collections import Counter
from pathlib import Path
from plot_mltest20_algo40_avg_rank import find_traj, load_tasks

root = Path(sys.argv[1])
tasks = load_tasks(Path(sys.argv[2]))
hist = Counter()
short = []
for row in csv.DictReader(Path(sys.argv[3]).open()):
    gv = row["variant"].strip()
    if gv.startswith("ptllm__") and not gv.startswith("ptllm_tpe"):
        continue
    disk = gv.replace("ptllm_tpe__", "ptllm_tpe_qwen__", 1).replace("ptllm__", "ptllm_qwen__", 1)
    n = sum(1 for t in tasks if find_traj(root, t["task_type"], t["dataset"], disk, 80, algorithm=t["algorithm"]))
    hist[n] += 1
    if n < 40:
        short.append((n, disk))
print("algo-correct coverage after repair:")
for k in sorted(hist, reverse=True):
    print(f"  {k}/40: {hist[k]} configs")
if short:
    print("still short:")
    for n, v in sorted(short)[:20]:
        print(f"  {n}/40  {v}")
PY
#!/usr/bin/env bash
# Repair Qwen s1b_attr_t2 P5+ algo40 slots that have 80 trials of the WRONG algorithm
# (shared dataset/variant dir collision). Does NOT re-run the full sweep.
#
# For each needed (config, dataset, algorithm):
#   1) quarantine sibling-algo CASH runs so --resume does not skip
#   2) re-run only that job
#   3) restore quarantined siblings
#
#   bash examples/benchmark/run_p5plus_qwen_attr_t2_repair.sh
#   NUM_GPUS=4 JOBS_PER_GPU=3 bash ...
#   DRY_RUN=1 bash ...
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

TASKS_JSON="${TASKS_JSON:-benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40.json}"
SUMMARY_CSV="${SUMMARY_CSV:-benchmark_data/real_hpo_v2_mltest20_algo40/analysis/p5plus/gpt_p5_p5plus_summary.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_data/real_hpo_v2_mltest20_algo40_p5_s1b_attr_t2_p5plus}"
LOGDIR="${LOGDIR:-benchmark_data/real_hpo_v2_mltest20_algo40/logs}"
PYTHON_LLM="${PYTHON_LLM:-/root/miniconda3/envs/openbox_qwen_eval/bin/python}"
PYTHON_SCAN="${PYTHON_SCAN:-/root/miniconda3/envs/openbox_eval/bin/python}"
BASE_MODEL="${BASE_MODEL:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
ADAPTER_PATH="${ADAPTER_PATH:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-7200}"
PER_RUN_TIME_LIMIT="${PER_RUN_TIME_LIMIT:-1800}"
DATASET_RETRIES="${DATASET_RETRIES:-3}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
POLL_SEC="${POLL_SEC:-15}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_PURE_PTLLM="${SKIP_PURE_PTLLM:-1}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="$LOGDIR/qwen_p5plus_attr_t2_repair_${TS}.log"
MANIFEST="$LOGDIR/qwen_attr_t2_repair_manifest_${TS}.txt"
PIDFILE="$LOGDIR/qwen_p5plus_attr_t2_repair.pid"

mkdir -p "$LOGDIR" "$OUTPUT_ROOT"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"
echo $$ >"$PIDFILE"

log() { echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"; }

log "=== Qwen attr_t2 P5+ wrong-algo repair ==="
log "output=$OUTPUT_ROOT gpus=$NUM_GPUS jobs_per_gpu=$JOBS_PER_GPU"

log "scanning repair slots → $MANIFEST"
"$PYTHON_SCAN" - "$MANIFEST" "$OUTPUT_ROOT" "$TASKS_JSON" "$SUMMARY_CSV" "$SKIP_PURE_PTLLM" <<'PY' | tee -a "$MASTER_LOG"
import csv, sys
from collections import Counter
from pathlib import Path

manifest_path = Path(sys.argv[1])
root = Path(sys.argv[2])
tasks = __import__("json").loads(Path(sys.argv[3]).read_text())
summary_csv = Path(sys.argv[4])
skip_pure = sys.argv[5] == "1"

def parse_gpt_variant(v: str):
    parts = v.split("__")
    opt = parts[0]
    prompt = parts[1] if len(parts) > 1 else ""
    hybrid = parts[2] if len(parts) > 2 else ""
    return opt, prompt, hybrid

def traj_mode(traj: Path):
    rows = list(csv.DictReader(traj.open(newline="", encoding="utf-8")))
    algos = [str(r.get("algorithm") or "").strip() for r in rows if str(r.get("algorithm") or "").strip()]
    if not algos:
        return None, len(rows)
    return Counter(algos).most_common(1)[0][0], len(rows)

wanted = []
for row in csv.DictReader(summary_csv.open()):
    gv = row["variant"].strip()
    if skip_pure and gv.startswith("ptllm__") and not gv.startswith("ptllm_tpe"):
        continue
    opt, prompt, hybrid = parse_gpt_variant(gv)
    q_opt = opt.replace("ptllm_tpe", "ptllm_tpe_qwen", 1) if opt.startswith("ptllm_tpe") else opt.replace("ptllm", "ptllm_qwen", 1)
    disk = f"{q_opt}__{prompt}" + (f"__{hybrid}" if hybrid else "")
    wanted.append((q_opt, prompt, hybrid, disk, gv))

needs = []
for q_opt, prompt, hybrid, disk, gv in wanted:
    for t in tasks:
        tt, ds, algo = t["task_type"], t["dataset"], t["algorithm"]
        opt_dir = root / tt / ds / disk
        good = False
        if opt_dir.is_dir():
            for run in sorted(opt_dir.glob("CASH-*")):
                trajs = sorted(run.glob("*_traj.csv"))
                if not trajs:
                    continue
                mode, n = traj_mode(trajs[-1])
                if mode == algo and n >= 80:
                    good = True
                    break
        if good:
            continue
        needs.append(f"{q_opt}|{prompt}|{hybrid}|{tt}|{ds}|{algo}")

manifest_path.write_text("\n".join(needs) + ("\n" if needs else ""), encoding="utf-8")
print(f"repair_jobs={len(needs)}")
by = Counter(x.split("|")[0] + "__" + x.split("|")[1] + (("__" + x.split("|")[2]) if x.split("|")[2] else "") for x in needs)
for k, n in by.most_common(12):
    print(f"  {n:3d}  {k}")
if len(by) > 12:
    print(f"  ... {len(by)} configs total")
PY

if [[ ! -s "$MANIFEST" ]]; then
  log "nothing to repair — all TPE arms algo-correct"
  exit 0
fi

NJOBS=$(grep -cve '^$' "$MANIFEST" || true)
log "launching $NJOBS repair jobs"

if [[ "$DRY_RUN" == "1" ]]; then
  log "DRY_RUN: manifest only at $MANIFEST"
  exit 0
fi

gpu_free_mb() {
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}
is_alive() { [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null; }

quarantine_siblings() {
  local opt_dir="$1" algo="$2" qdir="$3"
  mkdir -p "$qdir"
  [[ -d "$opt_dir" ]] || return 0
  local run trajs mode
  for run in "$opt_dir"/CASH-*; do
    [[ -d "$run" ]] || continue
    trajs=("$run"/*_traj.csv)
    if [[ ! -e "${trajs[0]:-}" ]]; then
      mv "$run" "$qdir/"
      continue
    fi
    mode="$("$PYTHON_SCAN" - "$run" <<'PY'
import csv, sys
from collections import Counter
from pathlib import Path
run = Path(sys.argv[1])
trajs = sorted(run.glob("*_traj.csv"))
if not trajs:
    print("")
    raise SystemExit
rows = list(csv.DictReader(trajs[-1].open(newline="", encoding="utf-8")))
algos = [str(r.get("algorithm") or "").strip() for r in rows if str(r.get("algorithm") or "").strip()]
print(Counter(algos).most_common(1)[0][0] if algos else "")
PY
)"
    if [[ "$mode" != "$algo" ]]; then
      mv "$run" "$qdir/"
    fi
  done
}

restore_quarantine() {
  local opt_dir="$1" qdir="$2"
  [[ -d "$qdir" ]] || return 0
  mkdir -p "$opt_dir"
  local item
  for item in "$qdir"/*; do
    [[ -e "$item" ]] || continue
    mv "$item" "$opt_dir/"
  done
  rmdir "$qdir" 2>/dev/null || true
}

run_one_repair() {
  local opt_base="$1" prompt="$2" hybrid="$3" tt="$4" ds="$5" algo="$6" gpu="$7"
  local tag="${opt_base}__${prompt}"
  [[ -n "$hybrid" ]] && tag="${tag}__${hybrid}"
  local opt_dir="$OUTPUT_ROOT/$tt/$ds/$tag"
  local qdir="$OUTPUT_ROOT/$tt/$ds/.quarantine_repair_${algo}_${TS}"
  local logf="$LOGDIR/qwen_attr_t2_repair_${tag}_${ds}_${algo}_${TS}.log"
  local runner

  log "  gpu=$gpu repair $tag $tt/$ds/$algo"
  quarantine_siblings "$opt_dir" "$algo" "$qdir" >>"$logf" 2>&1 || true

  runner=( "$PYTHON_LLM" examples/benchmark/run_toy_categorical_optimizers.py
    --algorithm "$algo"
    --n-trials "$N_TRIALS" --seed "$SEED" --skip-test-eval
    --llm-backends qwen-sft6 --qwen-backend hf
    --qwen-model-path "$BASE_MODEL"
    --qwen-adapter-path "$ADAPTER_PATH"
    --llm-max-tokens 768
    --llm-response-format rationale_json
    --output-dir "$OUTPUT_ROOT"
    --all-datasets --include-datasets "$ds"
    --time-limit "$TIME_LIMIT"
    --per-run-time-limit "$PER_RUN_TIME_LIMIT"
    --dataset-retries "$DATASET_RETRIES"
    --optimizer "$opt_base"
    --prompt-variant "$prompt"
    --resume
  )
  if [[ -n "$hybrid" ]]; then
    runner+=( --hybrid-variant "$hybrid" )
  fi

  set +e
  CUDA_VISIBLE_DEVICES="$gpu" "${runner[@]}" >>"$logf" 2>&1
  local rc=$?
  set -e
  restore_quarantine "$opt_dir" "$qdir" >>"$logf" 2>&1 || true
  if [[ $rc -ne 0 ]]; then
    log "  WARN repair rc=$rc $tag $ds/$algo (see $logf)"
  fi
  return 0
}

mapfile -t JOBS < "$MANIFEST"
declare -A SLOT_PID=()
declare -A SLOT_KEY=()  # serialize same dataset+variant
queue_idx=0
total=${#JOBS[@]}

log "repair queue: $total jobs"

while [[ $queue_idx -lt $total ]] || {
    any=0
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
      for ((slot=0; slot<JOBS_PER_GPU; slot++)); do
        is_alive "${SLOT_PID[${gpu}_${slot}]:-}" && any=1
      done
    done
    [[ $any -eq 1 ]]
  }; do
  for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    for ((slot=0; slot<JOBS_PER_GPU; slot++)); do
      pid="${SLOT_PID[${gpu}_${slot}]:-}"
      if is_alive "$pid"; then continue; fi
      SLOT_PID[${gpu}_${slot}]=
      SLOT_KEY[${gpu}_${slot}]=
      [[ $queue_idx -ge $total ]] && continue

      count=0
      for ((s=0; s<JOBS_PER_GPU; s++)); do
        is_alive "${SLOT_PID[${gpu}_${s}]:-}" && count=$((count+1))
      done
      [[ $count -lt $JOBS_PER_GPU ]] || continue
      if [[ $count -ge 1 ]]; then
        free=$(gpu_free_mb "$gpu")
        [[ "${free:-0}" -ge $PACK_MIN_FREE_MB ]] || continue
      fi

      # pick next job whose (tt,ds,tag) is not already running
      found=-1
      for ((j=queue_idx; j<total; j++)); do
        IFS='|' read -r opt_base prompt hybrid tt ds algo <<<"${JOBS[$j]}"
        tag="${opt_base}__${prompt}"
        [[ -n "$hybrid" ]] && tag="${tag}__${hybrid}"
        key="$tt|$ds|$tag"
        busy=0
        for ((g=0; g<NUM_GPUS; g++)); do
          for ((s2=0; s2<JOBS_PER_GPU; s2++)); do
            if is_alive "${SLOT_PID[${g}_${s2}]:-}" && [[ "${SLOT_KEY[${g}_${s2}]:-}" == "$key" ]]; then
              busy=1
            fi
          done
        done
        [[ $busy -eq 0 ]] || continue
        found=$j
        break
      done
      [[ $found -ge 0 ]] || continue

      # swap found into queue_idx position
      if [[ $found -ne $queue_idx ]]; then
        tmp="${JOBS[$queue_idx]}"
        JOBS[$queue_idx]="${JOBS[$found]}"
        JOBS[$found]="$tmp"
      fi

      IFS='|' read -r opt_base prompt hybrid tt ds algo <<<"${JOBS[$queue_idx]}"
      tag="${opt_base}__${prompt}"
      [[ -n "$hybrid" ]] && tag="${tag}__${hybrid}"
      key="$tt|$ds|$tag"
      queue_idx=$((queue_idx+1))
      log "dispatch repair $queue_idx/$total gpu=$gpu $tag $ds/$algo"
      ( run_one_repair "$opt_base" "$prompt" "$hybrid" "$tt" "$ds" "$algo" "$gpu" ) &
      SLOT_PID[${gpu}_${slot}]=$!
      SLOT_KEY[${gpu}_${slot}]="$key"
    done
  done
  sleep "$POLL_SEC"
done
wait

log "=== Qwen attr_t2 P5+ repair done ==="
log "manifest=$MANIFEST master_log=$MASTER_LOG"

# quick coverage report
"$PYTHON_SCAN" - "$OUTPUT_ROOT" "$TASKS_JSON" "$SUMMARY_CSV" <<'PY' | tee -a "$MASTER_LOG"
import csv, json, sys
from collections import Counter
from pathlib import Path
from plot_mltest20_algo40_avg_rank import find_traj, load_tasks

root = Path(sys.argv[1])
tasks = load_tasks(Path(sys.argv[2]))
hist = Counter()
short = []
for row in csv.DictReader(Path(sys.argv[3]).open()):
    gv = row["variant"].strip()
    if gv.startswith("ptllm__") and not gv.startswith("ptllm_tpe"):
        continue
    disk = gv.replace("ptllm_tpe__", "ptllm_tpe_qwen__", 1).replace("ptllm__", "ptllm_qwen__", 1)
    n = sum(1 for t in tasks if find_traj(root, t["task_type"], t["dataset"], disk, 80, algorithm=t["algorithm"]))
    hist[n] += 1
    if n < 40:
        short.append((n, disk))
print("algo-correct coverage after repair:")
for k in sorted(hist, reverse=True):
    print(f"  {k}/40: {hist[k]} configs")
if short:
    print("still short:")
    for n, v in sorted(short)[:20]:
        print(f"  {n}/40  {v}")
PY
