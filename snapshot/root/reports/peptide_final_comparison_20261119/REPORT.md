# Completed comparison: ten peptide tasks

All30 method/task results complete. Best feasible predicted MIC in µM; lower is better. Endpoints reuse exact100 starting measurements.

| Task | Original frozen FSBO | Online VAE + FSBO | SemanticOPT | Frozen runtime(min) | Online runtime(h) | SemanticOPT continuation(h) |
|---|---:|---:|---:|---:|---:|---:|
| pep_117 | 2.681 | 3.069 | 2.661 | 45.0 | 4.67 | 26.55 |
| pep_219 | 7.690 | 4.078 | 6.264 | 41.5 | 4.30 | 26.43 |
| pep_221 | 16.109 | 12.956 | 4.045 | 44.8 | 3.41 | 25.71 |
| pep_575 | 19.065 | 14.261 | 15.123 | 31.4 | 3.78 | 26.54 |
| pep_673 | 20.861 | 20.866 | 21.668 | 74.8 | 5.29 | 26.13 |
| pep_921 | 42.159 | 34.664 | 11.005 | 64.7 | 6.15 | 25.99 |
| pep_925 | 20.802 | 25.966 | 16.020 | 33.9 | 6.47 | 26.50 |
| pep_967 | 81.311 | 60.756 | 33.904 | 66.3 | 5.77 | 25.48 |
| pep_980 | 3.297 | 3.438 | 2.573 | 63.9 | 4.52 | 26.45 |
| pep_999 | 27.849 | 109.458 | 29.575 | 63.2 | 5.02 | 25.74 |

Mean endpoint MIC: Original frozen FSBO: 24.1826, Online VAE + FSBO: 28.9512, SemanticOPT with fallback: 14.2838.

Budget note: original endpoint uses older5000 cap; online uses4000 proposals or fresh evaluations (3693–4000 fresh); SemanticOPT4000 distinct fresh. These endpoints are not equal-budget.

Matched at3500 fresh evaluations per task: {"tasks": 10, "mean_mic": {"Original frozen FSBO": 28.55900993347168, "Online VAE + FSBO": 31.51464626789093, "SemanticOPT with fallback": 14.7135999917984}, "online_vs_original_wins": 6, "online_vs_original_ties": 0, "online_vs_semanticopt_wins": 3}.

Matched at each online task actual final fresh count: {"tasks": 10, "mean_mic": {"Original frozen FSBO": 28.405395698547363, "Online VAE + FSBO": 28.951193404197692, "SemanticOPT with fallback": 14.492411065101624}, "online_vs_original_wins": 5, "online_vs_original_ties": 0, "online_vs_semanticopt_wins": 3}.

Original mean task runtime: 52.94min; online 4.94h; aggregate task sums 8.82h and 49.39h, respectively. Online offline preparation adds 0.99h. Online campaign start through final report took 17.02h with four concurrent task lanes and restarts.

SemanticOPT continuation from imported1000 to4000 took 26.56h elapsed for all ten concurrent tasks; this excludes the original1000 generation and model training. The7.9h last-process timer covers only the post-restart segment. Full4000 runtime cannot be reconstructed as one clean active-duration timer from current evidence.

Source re-encoding/full-source scratch FSBO accounts for 67.8% of accumulated online evaluation wall time. SemanticOPT proposals include 78.3% labeled random fallback.

Audits: All initial100 sequences and measured scores identical; frozen archive payload hashes and SemanticOPT imported1000 hashes verified; unique/cache accounting and75% similarity constraints checked; batch-causal curves; endpoints and counts validated.

- Endpoint budgets unequal: original stops at5000 proposal or fresh cap; online at4000 proposal or fresh cap; SemanticOPT exactly4000 fresh. See matched budgets to assess quality independently of this discrepancy.
- SemanticOPT first1000 observations imported from an earlier campaign.26.6h continuation runtime is not the runtime for all4000; last process timer is only about7.9h and is not total.
- Frozen and online task wall sums are aggregate task hours, not elapsed campaign duration. SemanticOPT tasks run concurrently and its task hours must not be interpreted as CPU/GPU compute.
- Historical hardware/software, inference concurrency, shared resources and pause/restart histories differ; runtimes are operational measurements, not a controlled speed benchmark.
- One seed per method/task. Online changes initial VAE and FSBO training population as well as online updates; this does not isolate the VAE contribution.
