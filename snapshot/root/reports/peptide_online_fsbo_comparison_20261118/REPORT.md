# Peptide: online VAE/FSBO versus frozen FSBO and SemanticOPT

Snapshot: 2026-11-18T06:59:24.078638+00:00 (14:59 China time).

The current replacement online run has started 4/10 tasks; none has completed. Six remain pending. All ten tasks use exactly identical 100 initial measurements across methods.

At 2,500 new evaluations per task, online VAE/FSBO beats frozen FSBO on 4/4 started tasks and SemanticOPT on 1/4. Lower predicted MIC is better.

| Task | Frozen VAE + FSBO | Online VAE + retrained FSBO | SemanticOPT with fallback |
|---|---:|---:|---:|
| pep_117 | 7.074 | 4.194 | 2.661 |
| pep_219 | 8.751 | 5.945 | 6.264 |
| pep_221 | 27.484 | 14.308 | 4.045 |
| pep_575 | 26.841 | 15.279 | 15.123 |
| pep_673 | 20.861 | Pending | 21.668 |
| pep_921 | 47.310 | Pending | 13.091 |
| pep_925 | 27.490 | Pending | 18.821 |
| pep_967 | 95.205 | Pending | 33.904 |
| pep_980 | 3.297 | Pending | 3.877 |
| pep_999 | 112.650 | Pending | 29.575 |

Four-task mean MIC: frozen 17.537; online 9.932; SemanticOPT 7.023. Online is 43.4% below frozen in mean MIC; SemanticOPT is 29.3% below online. These are partial four-task results, not final ten-task averages.

The online run is using the selected epoch-1 offline VAE checkpoint followed by task-specific reconstruction/KL updates. Accepted VAE changes are followed by a fresh 100,000-step FSBO retraining on source800 plus only the current task observations. Four accepted updates per active task are saved. The combined method changes both offline and online training, so these results do not isolate the contribution of each component.

Each active task has 2,528–2,865 saved fresh evaluations. Source re-encoding and FSBO retraining consume about 83% of recorded evaluation wall time. Historical hardware/software and contention differ, so runtime is not a controlled speed comparison.

SemanticOPT has 30,650 scored proposals across the ten tasks, of which 23,327 (76.1%) are labeled random fallback. Its reported results include this fallback policy.

Evaluation budgets exclude initialization, count infeasible evaluations and recenter calls, and exclude cache hits. Curves reveal full batches causally and stop at measured support; pending work is never imputed. Frozen archives, SemanticOPT imported prefixes, exact initial measurements, feasibility and cache accounting passed audits. One seed is available for each method/task.

[Convergence plot](comparison.png) · [Paper PDF](comparison.pdf) · [Matched values](matched_budgets.csv) · [Progress](progress.csv) · [Snapshot summary](summary.json)
