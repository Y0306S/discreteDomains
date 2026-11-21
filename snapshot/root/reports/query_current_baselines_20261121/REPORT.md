# Current query ORPO + FSBO versus baselines

Snapshot: 2026-11-21T02:40:36.833580+00:00. Campaign running; 107/200 complete. RL disabled.

Percentages below are reductions in best measured query execution runtime, using geometric means of per-query ratios; they are not reductions in optimization wall-clock time.

| Comparator | At 200 proposals: 131 shared queries | At 770 proposals: 107 shared queries |
|---|---:|---:|
| FSBO | 33.9% lower | 17.4% lower |
| LOLBO | 42.0% lower | 35.6% lower |
| TPE | 36.9% lower | 31.2% lower |
| SemanticOPT | 34.8% lower | No measured results beyond 200 |

Each column uses the same query cohort for all methods in that column; cohorts differ between columns, so changes in the gap are not a fixed-cohort trend.

At 770 new proposals our method beats FSBO by more than 5% on 68 queries, loses by more than 5% on 22, and is within 5% on 17. Average rank among the four methods: our method1.47, FSBO2.01, TPE3.16, LOLBO3.36; lower is better.

As an unequal-budget reference, our770-proposal results have12.0% lower runtime than FSBO at1000proposals on those same107queries. This is not an equal-budget test.

These are full-pipeline comparisons: our method uses BOLT30 initialization, while baselines use Bao49. Different seeds, hosts, dates, storage and timeout policies apply. Equal-total-observation sensitivity is saved in summary.json. The campaign remains partial and completion-dependent. These comparisons cannot attribute gains to VAE retraining.

SemanticOPT ends at200measured proposals; no results were extrapolated. Baseline histories and provenance were audited in /root/reports/query_convergence_four_methods_20261117; the cached raw runtime curves are hashed in summary.json. Current production journals were opened read-only, using the last committed complete-round optimization state.

Artifacts: comparison.png/pdf/svg, per_query_200.csv, per_query_500.csv, per_query_770.csv, summary.json, compare_current.py.

SemanticOPT is also drawn in the 770-proposal panel, using exactly that panel's107-query cohort, through its measured endpoint of200proposals (20rounds at batch size10). The line stops there; no values are extrapolated.
