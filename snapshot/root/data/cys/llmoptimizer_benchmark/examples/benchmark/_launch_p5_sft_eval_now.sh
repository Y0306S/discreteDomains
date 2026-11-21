#!/bin/bash
set -euo pipefail
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"
LOGDIR="benchmark_data/real_hpo_v2/logs"
mkdir -p "$LOGDIR"
TS=$(date +%Y%m%d_%H%M%S)

launch_model() {
  local model="$1"
  local gpu="$2"
  local log="$LOGDIR/p5_sft_${model}_h0_h4_${TS}.log"
  nohup bash -c "
    set -euo pipefail
    cd '$REPO'
    for H in H0 H4; do
      echo \"[\$(date -Is)] START ${model} \$H\"
      MODEL=${model} HYBRID=\$H CUDA_VISIBLE_DEVICES=${gpu} TIME_LIMIT=7200 \
        bash examples/benchmark/run_p5_sft_eval8_h0_h4.sh
      echo \"[\$(date -Is)] DONE ${model} \$H\"
    done
  " >"$log" 2>&1 &
  echo "${model}_pid=$! log=$log"
}

launch_model s1a 0
launch_model s1b 1

WATCH_LOG="$LOGDIR/p5_sft_rank_watcher_${TS}.log"
nohup bash -c "
set -euo pipefail
cd '$REPO'
sleep 3
LOGA=\$(ls -t benchmark_data/real_hpo_v2/logs/p5_sft_s1a_h0_h4_*.log | head -1)
LOGB=\$(ls -t benchmark_data/real_hpo_v2/logs/p5_sft_s1b_h0_h4_*.log | head -1)
echo \"[watcher] waiting on \$LOGA and \$LOGB\"
while true; do
  a_done=\$(grep -c 'DONE s1a H4' \"\$LOGA\" || true)
  b_done=\$(grep -c 'DONE s1b H4' \"\$LOGB\" || true)
  if [[ \"\$a_done\" -ge 1 && \"\$b_done\" -ge 1 ]]; then
    echo \"[watcher] both finished at \$(date -Is)\"
    break
  fi
  if ! pgrep -f 'run_p5_sft_eval8_h0_h4.sh' >/dev/null && ! pgrep -f 'run_toy_categorical_optimizers.py.*p5_sft' >/dev/null; then
    echo \"[watcher] runners exited at \$(date -Is); regenerating\"
    break
  fi
  sleep 120
done
export PYTHONPATH=examples/benchmark\${PYTHONPATH:+:\$PYTHONPATH}
/root/miniconda3/envs/openbox_eval/bin/python examples/benchmark/plot_all_valid_best.py \\
  --out-dir benchmark_data/real_hpo_v2/analysis/eval8/all_valid_best
echo '[watcher] regenerated avg_rank_table.csv'
grep -E 'qwen_s1|P5/H0|P5/H4' benchmark_data/real_hpo_v2/analysis/eval8/all_valid_best/avg_rank_table.csv | head -20
" >"$WATCH_LOG" 2>&1 &
echo "watcher_pid=$! log=$WATCH_LOG"
echo "TS=$TS"
