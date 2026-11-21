# Peptide comparison under a 4,000 cap

The user’s “400 cap” is interpreted as 4,000, consistent with the preceding 80 rounds × 50 and explicit request for 4,000-evaluation timing. Original FSBO is truncated retrospectively; no rerun or modification of production results.

| Method | Mean endpoint MIC (µM) | Estimated hours / 4,000 fresh evaluations | Range across 10 tasks |
|---|---:|---:|---:|
| Original frozen FSBO | 28.405 | 0.75 | 0.34–1.06 |
| Online VAE + FSBO | 28.951 | 5.12 | 3.52–6.64 |
| SemanticOPT with fallback | 14.284 | 7.11 | 4.69–8.76 |

SemanticOPT with the observed ten-task queue: 34.87 h per task estimated. The solid-bar estimate removes queue wait using broker request start/finish timestamps; it is not a new isolated-task benchmark. Both use ×4000/3000 continuation scaling. Input prompts and source mixture may differ during the first 1,000, so this is a planning estimate.

Frozen/online times use measured capped task wall ×4000/actual fresh. Offline preparation for online VAE/FSBO was 0.99h shared across the cohort; it is excluded from per-task estimates.

All methods capped at 4,000 acquisition proposals or 4,000 fresh evaluations, whichever first. Frozen history uses the last fully measured batch within both caps; crossing batches excluded, so two tasks stop at 3,990 fresh. Stops after 80 rounds can leave fewer than 4,000 fresh. No fabricated extra observations.

Same causal trailing mean over 200 evaluation positions for every method; smoothing applies to displayed aggregate curves only; reported MIC values and per-task plots remain exact.

All 4,000-fresh runtimes are linear estimates using actual recorded timings, not fresh isolated benchmarks. Original/online historical hardware and contention differ; SemanticOPT estimates use 3,000 continuation measurements and the continuation request telemetry. Offline training/loading excluded.

| Task | Frozen MIC | Online MIC | SemanticOPT MIC | Frozen fresh | Online fresh | SemanticOPT fresh |
|---|---:|---:|---:|---:|---:|---:|
| pep_117 | 2.681 | 3.069 | 2.661 | 3990 | 3943 | 4000 |
| pep_219 | 8.751 | 4.078 | 6.264 | 3990 | 3823 | 4000 |
| pep_221 | 21.607 | 12.956 | 4.045 | 3710 | 3869 | 4000 |
| pep_575 | 20.228 | 14.261 | 15.123 | 3944 | 4000 | 4000 |
| pep_673 | 20.861 | 20.866 | 21.668 | 3747 | 3903 | 4000 |
| pep_921 | 42.159 | 34.664 | 11.005 | 3575 | 3762 | 4000 |
| pep_925 | 25.954 | 25.966 | 16.020 | 3831 | 3900 | 4000 |
| pep_967 | 95.205 | 60.756 | 33.904 | 3755 | 3693 | 4000 |
| pep_980 | 3.297 | 3.438 | 2.573 | 3695 | 3950 | 4000 |
| pep_999 | 43.308 | 109.458 | 29.575 | 3624 | 3841 | 4000 |

[Mean convergence](mean_convergence_4000.png) · [Per-task convergence](per_task_convergence_4000.png) · [Final MIC](final_mic_4000.png) · [Runtime](runtime_per_peptide_4000.png)

SemanticOPT retains its labeled random-fallback policy (78.3% of scored proposals in this cohort). All methods use the identical 100 imported starting measurements, excluded from evaluation counts. One seed per task/method; no claim of statistical superiority.
