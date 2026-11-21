# Online peptide diagnosis and 500-evaluation trajectories

All ten online tasks are complete at the established cap; these graphs analyze existing results. Batch size 50 and at most 80 acquisition rounds, not 500 rounds. Identical initial 100 measurements are excluded from fresh budgets.

## pep_999

The reference is KLYFLSFAITLAL (13 residues). The 75% similarity constraint permits at most 3 Levenshtein edits. Online FSBO produced 9 feasible fresh acquisition candidates out of 3,837 (0.235%); 3,828 were infeasible. Original frozen FSBO produced 17 feasible fresh candidates within its capped history. Both were inefficient; online never found the better late candidate reached by frozen FSBO.

Online MIC improved only once: 112.650  → 109.458 µM at round 52 (2,483 total fresh, at completion of the scoring batch). It remained flat through round 80 (3,841 fresh), a 1,358-evaluation tail without improvement. Frozen FSBO found 109.463 at round 69 and 43.308 at round 72. The slightly different oracle readings for their shared sequence are preserved rather than silently reconciled.

There were zero feasible fresh acquisitions in online rounds 1–30 (1,420 fresh), before and across its first updates. All seven update cycles were accepted; each completed 100,000-step scratch FSBO retraining with verified checkpoint reload. Every source reconstruction probe retained 128/128 feasible reconstructions, but those probes cover 32 source tasks, not pep_999 acquisition neighborhoods. No task-specific decoder/feasibility rollback is implemented.

Recenter attempts produced 2,138 decoded proposals, only 4 fresh sequences, and zero fresh feasible sequences. This was largely replay of known seeds, not an expansion of the useful search neighborhood. Source re-encoding and scratch FSBO consumed 69.3% of task wall; target refitting consumed 29.5%. More compute did not ensure better constrained exploration.

This supports a constrained-search/decoder bottleneck, not a demonstrated training crash or proven causal regression from online updates. The initial VAE and FSBO prior also differ from frozen baseline, and only one optimizer seed was run. Short length makes the discrete constraint tight, but length alone does not explain outcomes. A controlled adapted-frozen arm with matching initial weights/seeds is needed to isolate online updates.

## Observed online plateaus

| Task | Initial MIC | Final MIC | Last improvement round | Fresh at last improvement | Subsequent flat fresh evaluations |
|---|---:|---:|---:|---:|---:|
| pep_117 | 7.074 | 3.069 | 79 | 3893 | 50 |
| pep_219 | 9.262 | 4.078 | 69 | 3372 | 451 |
| pep_221 | 27.484 | 12.956 | 68 | 3274 | 595 |
| pep_575 | 26.841 | 14.261 | 65 | 3265 | 735 |
| pep_673 | 21.668 | 20.866 | 49 | 2383 | 1520 |
| pep_921 | 67.433 | 34.664 | 59 | 2768 | 994 |
| pep_925 | 27.490 | 25.966 | 78 | 3800 | 100 |
| pep_967 | 136.724 | 60.756 | 76 | 3561 | 132 |
| pep_980 | 3.877 | 3.438 | 70 | 3455 | 495 |
| pep_999 | 112.650 | 109.458 | 52 | 2483 | 1358 |

5/10 online tasks had no improvement during their final 500 actual fresh evaluations. This describes the observed tail and does not establish an asymptotic plateau.

## Graphs and numerical data

- [All tasks at 500-evaluation checkpoints](all_tasks_every_500_evaluations.png)
- [Online interval improvements](online_improvement_per_500.png)
- [Last improvement and flat tails, all methods](last_improvement_and_flat_tail.png)
- [pep_999 diagnostic panels](pep999_diagnostics.png)

Exact scores and interval reductions are in mic_changes_per_500_evaluations.csv; final intervals are partial where runs stopped below 4,000. The plots preserve all batch-causal measured steps and actual stopping points, use the same method colors as the previous report, and apply no smoothing. These graphs intentionally expose plateaus.

Suggested next diagnostic, not launched: compare author-frozen, adapted-frozen and adapted-online from matched seeds; log constraint posterior calibration and target-specific reconstruction validity; test smaller trust-region perturbations and valid decode filtering in a separate arm. Any filtering/retry policy needs physical oracle counts reported consistently. Do not silently replace infeasible candidates or alter existing scores.

## Mean convergence requested by the user

The requested primary graph is the cohort mean best MIC at each evaluation stage, using the same ten tasks and established 4,000 cap. No new 500-round experiment was requested. The CSV provides every 50-evaluation budget; dots and the table below show 500-evaluation checkpoints. Stopped incumbents remain in the fixed cohort, without implying new measurements.

[Exact mean convergence](mean_mic_by_stage.png) · [Same 200-evaluation smoothing](mean_mic_by_stage_smoothed.png) · [Vector PDF](mean_mic_by_stage.pdf)

| Fresh budget | Frozen mean MIC | Online mean MIC | SemanticOPT mean MIC |
|---|---:|---:|---:|
| 0 | 44.050 | 44.050 | 44.050 |
| 500 | 44.050 | 44.050 | 24.941 |
| 1000 | 44.050 | 42.733 | 23.946 |
| 1500 | 44.050 | 42.733 | 20.509 |
| 2000 | 41.929 | 36.676 | 14.912 |
| 2500 | 37.696 | 32.888 | 14.903 |
| 3000 | 37.181 | 32.113 | 14.790 |
| 3500 | 28.559 | 31.515 | 14.714 |
| 4000 | 28.405 | 28.951 | 14.284 |

Monitoring remains active through the existing 600-second durable job. Its completed-result status handling and last-improvement diagnostics were corrected and the first updated healthy check was verified. Completed experiments are not automatically extended or restarted.
