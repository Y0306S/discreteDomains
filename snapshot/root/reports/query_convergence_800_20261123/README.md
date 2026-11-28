# Query convergence through800total evaluations

2026-11-23T23:12:08.997212+08:00

Primary: query_convergence_800.png/.pdf/.svg. Left: all200queries. Right: current BOLT cohort.

Additional: query_summed_runtime_800, query_pilot10_800, query_pilot10_per_task_800, query_new_proposals_800.

- 800 evaluations means 800 total counted observations/slots per query, including initialization; it is 80 batches of 10 for the hybrid and BOLT. SemanticOPT proposal targets exclude its49Bao starts.
- LLM + FSBO is the completed retrained-VAE ORPO + FSBO hybrid (RL disabled),200/200queries,800total evaluations each.
- FSBO,LOLBO,TPE are historical standalone baselines; their raw49Bao+1000proposal histories are represented by the previously audited unpadded curve cache. Charged cache-hit proposals remain slots, so this is not an equal fresh-SQL-call comparison.
- Each aggregate uses a fixed query cohort and truncates each method at the minimum available prefix across that cohort. No forward filling, variable-cohort averaging, smoothing, or extrapolation.
- The full200-query BOLT trajectory currently has only its shared30-observation warm-start point. Its active-query panel and per-task pilot plots expose actual ongoing progress.
- Incumbents improve only on uncensored successful measured runtimes. Timeouts count as evaluation slots but cannot improve the incumbent. Decoder rejection attempts without SQL do not count as evaluations.
- LLM + FSBO and BOLT share exactly the same30warm measurements (verified); SemanticOPT retains its original200-record prefix (verified) and shares Bao initialization with the audited standalone baselines.
- Relative runtime uses a fixed per-query Bao49 best anchor, then a geometric mean; lower is better. Summed-runtime plots show rawseconds. Initial pools differ between method families.
- Historical devices, dates, seeds, SQL timeout and storage policies differ; these comparisons are descriptive. No seed-replication uncertainty is available.
- The extra new-proposals figure excludes initialization; the completed800-total hybrid therefore ends at770new proposals. This alternative axis must not be confused with the primary total-evaluation axis.
- SQLite journals were opened read-only in per-task consistent transactions. Running tasks can advance during the snapshot; only complete saved batch states are included for BOLT and hybrid.

Exports: convergence.csv,per_query_coverage.csv,observed_total_evaluation_curves.npz,audit.json.
Refresh read-only with: python3 build_plot.py.
