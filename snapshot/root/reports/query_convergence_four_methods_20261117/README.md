# Query convergence: FSBO, LOLBO, TPE, SemanticOPT

2026-11-17T16:08:35.740976+08:00

Main plot: `query_convergence.png` (also PDF and SVG). Left: equal 200-proposal budget; right: all observed proposals. SemanticOPT ends at 200.

| Method | Complete queries | Proposals/query | Score at 200 ↑ | Runtime ratio at 200 ↓ | Speedup at 200 ↑ |
|---|---:|---:|---:|---:|---:|
| FSBO | 200/200 | 1000 | 0.8996 | 0.8361 | 1.196× |
| LOLBO | 200/200 | 1000 | 0.8671 | 0.9261 | 1.080× |
| TPE | 200/200 | 1000 | 0.8906 | 0.8596 | 1.163× |
| SemanticOPT | 200/200 | 200 | 0.9041 | 0.8306 | 1.204× |

- All 200 fixed held-out queries are included. No task filtering or extrapolation.
- All four methods share exactly 49 Bao initialization runtime/censoring measurements; source plan IDs agree for baselines and join groups agree with SemanticOPT.
- Only uncensored successful measured runtimes improve incumbents. Charged baseline cache hits remain proposal slots.
- Min–max bounds use full observed uncensored pools across all four methods, dropping the slowest floor(20%) for bounds only, and clipping scores to [0,1]. Bounds are retrospective and method-dependent; these scores differ from the previous three-method normalization.
- Summed-runtime and geometric-relative-runtime views do not use pooled min–max bounds.
- Individual points are observed proposal prefixes; lines connect observations without smoothing or resampling.
- SemanticOPT batch size is 10; within-batch ordering is recorded evaluation order, not feedback between its ten proposals.
- SemanticOPT includes 32690 constrained random fallback proposals out of 40000; not a pure LLM-only trajectory.
- SQL timeout: FSBO 50s, SemanticOPT 50s, TPE 100s, LOLBO adaptive. Seeds: FSBO123, SemanticOPT42, TPE/LOLBO0.
- One run per method; seeds, hosts, dates and storage differ. No seed-replication uncertainty is estimated.
- Production histories were opened read-only and were not modified.

Additional plots: `query_summed_runtime_score` and `query_relative_runtime` (PNG/PDF/SVG).
Exports: convergence.csv, per_task_comparison_at_200.csv, per_task_curves.npz, normalization_bounds.csv, audit.json.
Reproduce: `python3 build_plot.py`. Only this report directory is written.

- The displayed normalized score axis is zoomed to 0.8–1.0; exported scores retain the full [0,1] scale.
