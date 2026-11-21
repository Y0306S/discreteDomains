# Target-loss rollback comparison

Same ten held-out peptide tasks, initialization100, batch50 and proposal/fresh caps4000. FSBO uses seed1; SemanticOpt uses seed42.

| Method | Mean final MIC (µM, lower is better) |
| --- | ---: |
| Original frozen FSBO | 28.4054 |
| Online VAE + FSBO | 28.9512 |
| SemanticOPT with fallback | 14.2838 |
| Online VAE + FSBO + rollback (20 rounds, max2) | 27.1539 |

![Convergence](comparison.png)

Earlier comparison trajectories come from /root/data/logs/peptide_target_loss_gate_20round_max2_test10_20261119/comparison_inputs/capped_convergence.csv and use the same proposal cap.
This run reuses the previous offline VAE/FSBO assets. Accepted updates pass the
new fixed-target no-worsening gate and original source gate; rejected updates
restore the VAE exactly. See task_results.csv for per-task updates, MIC and runtime.
Runtime is descriptive: previous experiment used A800, current experiment uses
A16 with shared workloads. Ten tasks and one optimizer seed do not establish
general significance. No future test observations enter validation.

## Direct comparison with SemanticOpt

SemanticOpt with fallback achieved mean final feasible MIC 14.2838 µM versus 27.1539 µM for online VAE + FSBO with rollback: 47.4% lower mean MIC. SemanticOpt won 6 tasks and retrained FSBO won 4. Retrained FSBO still improved over frozen FSBO by 4.4% in mean MIC (9 wins, 1 loss).

| Task | Frozen FSBO | Retrained FSBO + rollback | SemanticOpt with fallback | Winner vs retrained |
| --- | ---: | ---: | ---: | --- |
| pep_117 | 2.681 | 2.574 | 2.661 | Retrained FSBO |
| pep_219 | 8.751 | 3.783 | 6.264 | Retrained FSBO |
| pep_221 | 21.607 | 16.605 | 4.045 | SemanticOpt |
| pep_575 | 20.228 | 17.589 | 15.123 | SemanticOpt |
| pep_673 | 20.861 | 18.873 | 21.668 | Retrained FSBO |
| pep_921 | 42.159 | 13.435 | 11.005 | SemanticOpt |
| pep_925 | 25.954 | 13.372 | 16.020 | Retrained FSBO |
| pep_967 | 95.205 | 81.323 | 33.904 | SemanticOpt |
| pep_980 | 3.297 | 2.978 | 2.573 | SemanticOpt |
| pep_999 | 43.308 | 101.007 | 29.575 | SemanticOpt |

MIC values are in µM; lower is better. This comparison uses the same ten tasks, shared initialization100, batch50, and a 4,000-proposal cap with a 4,000-fresh-evaluation ceiling, excluding initialization. Actual fresh evaluations differ: retrained FSBO used 3,729–3,992 per task; SemanticOpt trajectories record 4,000 per task. This compares final results under the shared caps, rather than exactly equal fresh-evaluation counts.

The mean gap is driven especially by pep_967 and pep_999. These are ten tasks with one seed per method (FSBO seed1; SemanticOpt seed42). No controlled SemanticOpt versus FSBO runtime comparison is available. See [per-task CSV](semanticopt_comparison.csv) and [convergence plot](comparison.png).

[Per-task quality and runtime](per_task_quality_runtime.md) · [CSV](per_task_quality_runtime.csv)
