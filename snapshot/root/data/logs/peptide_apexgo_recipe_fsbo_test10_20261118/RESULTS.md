# Peptide VAE / FSBO adaptation pilot

Completed 10 held-out tasks × 2 comparison arms. Lower MIC is better.

| Arm | Mean final MIC (µM) | Mean online minutes | Mean updates |
|---|---:|---:|---:|
| Original FSBO (completed baseline) | 24.1826 | 52.94 | 0.00 |
| APEXGo recipe VAE + online FSBO | 28.9512 | 296.31 | 6.70 |

Offline VAE + embedding refresh + FSBO meta-training: 0.99 hours.

See summary.json for paired quality differences, bootstrap intervals, runtime ratios and amortized offline costs.

These are 10 tasks and one optimizer seed. A better VAE or improved search is an experimental outcome, not an assumption.

- Ten heldout tasks, one optimizer seed. Historical source800 corpus replaces original UniRef corpus for continued training; FSBO is our extension, not part of APEXGo.
- The author baseline reuses completed historical FSBO archives. Software, device placement and contention differ; runtime ratios do not isolate adaptation overhead.
- GPU hardware is shared with existing work; matched task/arm hardware and recorded utilization do not eliminate temporal contention.
- Every accepted online reconstruction/KL VAE update is followed by a fresh 100000-step FSBO meta-training run using source800 plus this task observed so far, all encoded with the updated VAE.
- Completed incumbents carry forward after proposal exhaustion; paired_results.csv records actual fresh-call support for every task.
- Offline VAE sees sequences from all 800 source tasks. FSBO uses all 800 source tasks with disjoint validation observations. The 10 held-out evaluation tasks remain excluded.
- APEXGo optimizer/loss settings are reused for continued training on source800, not a reproduction of its original UniRef pretraining run.
- Online VAE updates use reconstruction/KL on current-task recent50 and feasible top1000 unique observed sequences; rejected decoder updates roll back.
- The FSBO training population also changes relative to the historical baseline; this comparison evaluates the full revised method and does not isolate the VAE effect.
