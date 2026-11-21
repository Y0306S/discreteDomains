# Query convergence: one minus normalized summed runtime

`score(n) = 1 - sum_q best_runtime_q(n) / sum_q initial_Bao_best_runtime_q`

| Method | Initial sum (s) | Final sum (s) | Final score ↑ |
|---|---:|---:|---:|
| FSBO | 617.205 | 345.688 | 0.4399 |
| TPE | 617.205 | 451.241 | 0.2689 |
| LOLBO | 617.205 | 458.500 | 0.2571 |

- Sum best-so-far execution runtimes across the 200 queries at each proposal count, then normalize by the common starting sum.
- The sum is the runtime of the best plans for the workload at this point, not cumulative optimization/search cost.
- Zero is the initial Bao baseline; one is the zero-runtime ideal limit. Endpoints are not forced to one.
- This aggregation weights absolute saved seconds; it is not a geometric mean of per-query ratios.
- All three runs complete 1000 proposals per query, excluding the shared 49 Bao starts. Only uncensored measurements update incumbents.
- FSBO SQL cap 50 s, seed 123; TPE cap 100 s, seed 0; LOLBO adaptive cap, seed 0. Hardware, storage and dates differ.
- BOLT Figure 1 (page 7) confirms aggregation by sum across test tasks. Its caption does not specify an exact normalization denominator; this figure explicitly uses the shared Bao starting total.

Reproduce: `python3 ../plot_summed_score.py`.
The original geometric-mean plots remain in the parent directory.
