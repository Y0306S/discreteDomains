# Per-task peptide quality and runtime

Best feasible predicted MIC in µM; lower is better. Bold marks the best of the four methods.

| Peptide | Frozen FSBO | Earlier online | Online + rollback | SemanticOpt |
|---|---:|---:|---:|---:|
| pep_117 | 2.681 | 3.069 | **2.574** | 2.661 |
| pep_219 | 8.751 | 4.078 | **3.783** | 6.264 |
| pep_221 | 21.607 | 12.956 | 16.605 | **4.045** |
| pep_575 | 20.228 | **14.261** | 17.589 | 15.123 |
| pep_673 | 20.861 | 20.866 | **18.873** | 21.668 |
| pep_921 | 42.159 | 34.664 | 13.435 | **11.005** |
| pep_925 | 25.954 | 25.966 | **13.372** | 16.020 |
| pep_967 | 95.205 | 60.756 | 81.323 | **33.904** |
| pep_980 | 3.297 | 3.438 | 2.978 | **2.573** |
| pep_999 | 43.308 | 109.458 | 101.007 | **29.575** |
| **Mean** | 28.405 | 28.951 | 27.154 | 14.284 |

Runtime in hours. SemanticOpt columns have different scopes, explained below.

| Peptide | Frozen FSBO | Earlier online | Online + rollback | SemanticOpt observed continuation | SemanticOpt estimated 4,000, no queue |
|---|---:|---:|---:|---:|---:|
| pep_117 | 0.60 | 4.67 | 6.00 | 26.55 | 8.44 |
| pep_219 | 0.52 | 4.30 | 11.21 | 26.43 | 8.31 |
| pep_221 | 0.63 | 3.41 | 8.77 | 25.71 | 5.06 |
| pep_575 | 0.34 | 3.78 | 10.20 | 26.54 | 8.76 |
| pep_673 | 0.95 | 5.29 | 6.76 | 26.13 | 6.75 |
| pep_921 | 0.95 | 6.15 | 9.84 | 25.99 | 6.13 |
| pep_925 | 0.43 | 6.47 | 8.34 | 26.50 | 8.30 |
| pep_967 | 0.92 | 5.77 | 11.49 | 25.48 | 4.69 |
| pep_980 | 0.90 | 4.52 | 9.43 | 26.45 | 8.13 |
| pep_999 | 0.76 | 5.02 | 12.14 | 25.74 | 6.53 |
| **Mean** | 0.70 | 4.94 | 9.42 | 26.15 | 7.11 |

Same ten tasks, identical initialization100, batch50, and a 4,000-proposal / 4,000-fresh ceiling; counts exclude initialization. Actual fresh counts differ and are exported in the CSV. FSBO seed1; SemanticOpt seed42; one trajectory per task/method. SemanticOpt includes its labeled random fallback (78.3% of scored proposals in this cohort).

FSBO times are accumulated evaluation wall time through the capped endpoint, including online adaptation where applicable. Historical frozen/earlier-online runs used A800; current rollback used shared A16s. Initial offline VAE/embedding/FSBO preparation (about 0.99 h shared) is excluded and was reused by the rollback experiment. Times do not isolate the cost of the retraining policy.

SemanticOpt observed continuation time covers only evaluations 1,000 to 4,000 and includes queue waits and restart gaps; the first 1,000 observations were imported. Ten tasks ran concurrently, so these task elapsed hours must not be summed as campaign elapsed or compute time. A clean measured runtime for all 4,000 evaluations is unavailable. Its estimated 4,000 no-queue time sums measured inference-generation and oracle service for the 3,000 continuation observations, then multiplies by 4,000/3,000. It excludes queue wait, model loading and client overhead, and is a planning estimate, not a measured isolated run. Input prompts and source mixtures may differ in the imported first 1,000.

Sources: capped_convergence.csv; task_results.csv; /root/reports/peptide_cap4000_graphs_20261119/estimated_runtime_per_peptide.csv; /root/reports/peptide_final_comparison_20261119/REPORT.md; SemanticOpt extension.json (seed42).
