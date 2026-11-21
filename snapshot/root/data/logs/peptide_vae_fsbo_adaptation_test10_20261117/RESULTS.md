# Peptide VAE / FSBO adaptation pilot

Completed 10 held-out tasks × 2 comparison arms. Lower MIC is better.

| Arm | Mean final MIC (µM) | Mean online minutes | Mean updates |
|---|---:|---:|---:|
| Original FSBO (completed baseline) | 24.1826 | 52.94 | 0.00 |
| Train800 VAE + online updates | 44.0502 | 11.86 | 6.50 |

Offline VAE + embedding refresh + FSBO meta-training: 1.48 hours.

See summary.json for paired quality differences, bootstrap intervals, runtime ratios and amortized offline costs.

These are 10 tasks and one optimizer seed. A better VAE or improved search is an experimental outcome, not an assumption.

- Single optimizer seed; sequential shared offline VAE replay is a new transfer adaptation, not a fresh 800-task LOLBO campaign.
- The author baseline reuses completed historical FSBO archives. Software, device placement and contention differ; runtime ratios do not isolate adaptation overhead.
- GPU hardware is shared with existing work; matched task/arm hardware and recorded utilization do not eliminate temporal contention.
- Online joint optimization uses exact FSBO marginal likelihood plus native constraint predictive likelihood; it is a new FSBO adaptation variant.
- Completed incumbents carry forward after proposal exhaustion; paired_results.csv records actual fresh-call support for every task.
- Offline VAE sees all 800 source tasks, including FSBO internal validation tasks. The 10 held-out evaluation tasks remain excluded.
