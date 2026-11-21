# Peptide rollback comparison: nine completed tasks

The same nine completed tasks are included for every method. `pep_967` is still running and excluded from all means, ranks and plots.

| Method | Mean MIC (µM) ↓ | Median MIC (µM) ↓ | Average rank ↓ |
|---|---:|---:|---:|
| Original frozen FSBO | 20.983 | 20.861 | 3.22 |
| Online VAE + FSBO | 25.417 | 14.261 | 3.00 |
| SemanticOPT with fallback | 12.104 | 11.005 | 1.89 |
| Online VAE + FSBO + rollback (20 rounds, max2) | 21.135 | 13.435 | 1.89 |

Rollback improves on the previous online run in 7/9 tasks and reduces its mean MIC by 16.8%. It beats frozen FSBO in 8/9 tasks, but mean MIC is 0.7% worse because pep_999 remains poor. SemanticOPT has the lowest mean and median MIC; its average rank ties the new rollback run at 1.89. The new run beats SemanticOPT in 4/9 tasks.

| Peptide | Frozen | Previous online | SemanticOPT | New rollback |
|---|---:|---:|---:|---:|
| pep_117 | 2.681 | 3.069 | 2.661 | 2.574 |
| pep_219 | 8.751 | 4.078 | 6.264 | 3.783 |
| pep_221 | 21.607 | 12.956 | 4.045 | 16.605 |
| pep_575 | 20.228 | 14.261 | 15.123 | 17.589 |
| pep_673 | 20.861 | 20.866 | 21.668 | 18.873 |
| pep_921 | 42.159 | 34.664 | 11.005 | 13.435 |
| pep_925 | 25.954 | 25.966 | 16.020 | 13.372 |
| pep_980 | 3.297 | 3.438 | 2.573 | 2.978 |
| pep_999 | 43.308 | 109.458 | 29.575 | 101.007 |

The new run accepted 11 of 18 VAE retraining attempts and rolled back 7. Accepted updates pass both target-loss comparisons; rejected updates have exactly zero VAE weight delta. This protects the measured reconstruction-loss criterion but does not guarantee improved optimization or MIC. pep_999 accepted both attempts and still ends at 101.007 µM, versus frozen 43.308 and SemanticOPT 29.575 µM.

Mean new-run wall time was 9.19 h per completed peptide. Different hardware/shared workloads prevent a controlled speed comparison with prior methods.

Each history capped at 4000 acquisition proposals OR 4000 fresh evaluations, whichever first. Frozen retrospective cut uses last full batch within both caps. Endpoints may be earlier than 4000; carried incumbents are not new measurements.

All aggregate displays use the same trailing-200-position causal smoothing; raw mean panels, final scores and per-task plots are unsmoothed. Initialization100 is excluded from the x-axis. Nine completed tasks selected by completion, not the final ten-task cohort. One seed per task/method. Predicted MIC, not wet-lab MIC. Hardware/contention differ; runtime is not a controlled method comparison. SemanticOPT includes its original random fallback policy.

[Mean convergence](mean_convergence_fresh.png) · [Per-task convergence](per_task_convergence.png) · [Final MIC](final_mic.png) · [Proposal-axis view](mean_convergence_proposals.png)
