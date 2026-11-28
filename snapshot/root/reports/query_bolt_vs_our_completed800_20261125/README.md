# Query comparison: 30 completed BOLT queries

2026-11-25T23:08:54.323896+08:00

At 800 total evaluations, LLM + FSBO wins 23 queries; BOLT / LOLBO wins 6; ties 1.
LLM + FSBO reduces summed best runtime by 8.989% relative to BOLT / LOLBO.

- Cohort fixed by BOLT completion at snapshot time; it is not a random sample of all 200 queries.
- Both methods use identical 30 saved warm measurements and 770 further counted evaluations per query.
- Only uncensored measured runtimes improve incumbents; timeouts remain counted slots.
- Sum query incumbents first, then divide by the shared sum of best warm30 runtimes; lower is better.
- Current BOLT continuation runs the LOLBO optimizer. Historical standalone LOLBO is a separate experiment.
- SQL execution dates, host contention and storage conditions can differ; one saved run per method, no uncertainty bands.
- Running campaign journals are read only; no unfinished query is extrapolated.

Outputs: normalized_summed_runtime.png/.pdf/.svg, per_query.csv, convergence.csv, summary.json, curves.npz, row_digests.json.
Refresh with python3 build_report.py.
