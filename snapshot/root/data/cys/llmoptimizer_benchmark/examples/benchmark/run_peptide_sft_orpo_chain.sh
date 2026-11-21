#!/usr/bin/env bash
# Peptide SFT -> ORPO chain on the instrumented v2 BO run, family-disjoint 800/200 split.
#
#   stage 1  SFT   elite50   (GPUs 0,1)  and  SFT winbatch50 (GPUs 2,3), in parallel
#   stage 2  ORPO  from the elite50 SFT adapter (all 4 GPUs), after stage 1 finishes
#
# cutoff_len is 2048, NOT the 1024 used by the batch-10 configs: batch-50 targets are
# p50 1674 / max 1940 tokens, so 1024 silently truncates 100% of rows.
set -uo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
LF="/root/data/cys/LLMOptimizer/LlamaFactory"
CFG="$REPO/configs/peptide_sft"
LOG="/root/data/logs/peptide_sft_chain"
ENVBIN="${ENVBIN:-/root/miniconda3/envs/llamafactory_qwen35/bin}"
CLI="${CLI:-$ENVBIN/llamafactory-cli}"
# The launcher shells out to bare `torchrun` for multi-GPU, so its env bin must be on PATH.
export PATH="$ENVBIN:$PATH"
mkdir -p "$LOG"
cd "$LF"

run() {  # name gpus config
  local name="$1" gpus="$2" cfg="$3"
  echo "[chain] $(date -Is) START $name on GPU $gpus"
  CUDA_VISIBLE_DEVICES="$gpus" nohup "$CLI" train "$cfg" >>"$LOG/$name.log" 2>&1
  local rc=$?
  echo "[chain] $(date -Is) END   $name rc=$rc"
  return $rc
}

echo "[chain] stage 1: SFT (two arms in parallel)"
run sft_elite50    0,1 "$CFG/llamafactory_qwen35_9b_lora_peptide_elite50_sft.yaml"    & P1=$!
run sft_winbatch50 2,3 "$CFG/llamafactory_qwen35_9b_lora_peptide_winbatch50_sft.yaml" & P2=$!
wait $P1; R1=$?
wait $P2; R2=$?
echo "[chain] stage 1 done: elite50 rc=$R1  winbatch50 rc=$R2"

if [[ $R1 -ne 0 ]]; then
  echo "[chain] ABORT: elite50 SFT failed (rc=$R1); ORPO would start from a bad adapter"
  exit $R1
fi

ORPO_CFG="$CFG/llamafactory_qwen35_9b_lora_peptide_elite50_orpo.yaml"
ORPO_DATA="$REPO/benchmark_data/rl_task_selection/peptide_sft_v3/../peptide_orpo_v3/peptide_orpo_train.jsonl"
ORPO_DATA="$REPO/benchmark_data/rl_task_selection/peptide_orpo_v3/peptide_orpo_train.jsonl"
if [[ ! -f "$ORPO_CFG" ]]; then
  echo "[chain] ABORT: $ORPO_CFG missing"; exit 1
fi
# The pair builder runs concurrently with stage 1; make sure it actually finished.
for _ in $(seq 1 60); do
  pgrep -f build_peptide_orpo.py >/dev/null 2>&1 || break
  echo "[chain] waiting for the ORPO pair builder to finish..."; sleep 60
done
if [[ ! -s "$ORPO_DATA" ]]; then
  echo "[chain] ABORT: $ORPO_DATA missing or empty"; exit 1
fi
echo "[chain] ORPO pairs: $(wc -l <"$ORPO_DATA") rows"
echo "[chain] stage 2: ORPO from the elite50 adapter"
run orpo_elite50 0,1,2,3 "$ORPO_CFG"
echo "[chain] $(date -Is) chain complete"
