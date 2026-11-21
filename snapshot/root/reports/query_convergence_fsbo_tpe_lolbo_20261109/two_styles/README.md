# Two query convergence styles

Both figures use the shared ML/TaskSet PCHIP interpolation and method styles. Same
200 complete held-out queries and 1,000 charged proposals per method, plus the same
49 historical Bao starts. Observed integer scores, endpoints and monotonicity are
preserved. Interpolation creates no new measurements and cannot erase real jumps.

1. Summed runtime score: `1 - sum(best runtime) / sum(initial Bao best runtime)`;
   displayed y-axis 0–0.5, with no numeric clipping of scores.
2. Mean min–max score: pool uncensored runtimes per query across FSBO, TPE and
   LOLBO, drop the slowest floor(20%) only to set bounds, and compute
   `clip((worst_retained - best_so_far) / (worst_retained - pooled_best), 0, 1)`.
   Average all 200 queries equally. Displayed y-axis 0.5–1, matching the ML view;
   any earlier scores below 0.5 remain in the CSV. This follows the ML reference
   function exactly for all nondegenerate query scales.

| Method | Summed-runtime score | Mean min–max score |
|---|---:|---:|
| FSBO | 0.4399 | 0.9777 |
| TPE | 0.2689 | 0.8909 |
| LOLBO | 0.2571 | 0.9027 |

Normalization is fixed retrospectively using full observed trajectories, and
depends on the included methods. Timeout values are lower bounds, so they do not
enter the scale pool or update incumbent runtimes. Initial and cached uncensored
observations remain in the pool as charged observations. No tasks are excluded.
SQL caps, seeds, dates, storage and hosts differ across the original campaigns.

PNG, PDF, SVG, aggregate CSV, per-query scores, fixed normalization bounds and
hash-identified source pools are retained here. Production results are unchanged.
Reproduce: `python3 /root/reports/query_convergence_fsbo_tpe_lolbo_20261109/plot_two_styles.py`.
