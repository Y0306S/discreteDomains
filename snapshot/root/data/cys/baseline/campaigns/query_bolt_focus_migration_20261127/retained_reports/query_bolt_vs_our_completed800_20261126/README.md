# Query comparison: 42 completed BOLT queries

2026-11-26T10:14:40.587589+08:00

At 800 total evaluations, LLM + FSBO wins 32 queries; BOLT / LOLBO wins 8; ties 2.
LLM + FSBO reduces summed best runtime by 9.677% relative to BOLT / LOLBO.

- Cohort fixed by BOLT completion at snapshot time; it is not a random sample of all 200 queries.
- Both methods use identical 30 saved warm measurements and 770 further counted evaluations per query.
- Only uncensored measured runtimes improve incumbents; timeouts remain counted slots.
- Normalized summed runtime: sum query incumbents first, then divide by the shared sum of Bao49 best runtimes for these same queries; lower is better. This matches the reference convention of the previous completed-method plots.
- Mean normalized score: pool uncensored runtimes from both methods through 800 slots within each query, discard the slowest floor(20%) only to set the upper normalization bound, use the pooled minimum as the lower bound, clip (upper-incumbent)/(upper-lower) to [0,1], then average equally across queries. Bounds are retrospective, fixed across the curve and specific to these two methods.
- Paper-style summed aggregation is used with an explicit Bao49 denominator; the paper does not specify its exact normalization denominator.
- Current BOLT continuation runs the LOLBO optimizer. Historical standalone LOLBO is a separate experiment.
- SQL execution dates, host contention and storage conditions can differ; one saved run per method, no uncertainty bands.
- Running campaign journals are read only; no unfinished query is extrapolated.

Outputs: normalized_score.png/.pdf/.svg; normalized_summed_runtime.png/.pdf/.svg; per_query.csv, convergence.csv, normalization_bounds.csv, summary.json, curves.npz, row_digests.json.
Refresh with python3 build_report.py.
