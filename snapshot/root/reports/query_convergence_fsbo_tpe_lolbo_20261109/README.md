# Query FSBO / TPE / LOLBO comparison

2026-11-09T17:42:50.075777+08:00

| Method | Complete queries | Geometric runtime ratio ↓ | Speedup vs Bao ↑ | Average rank ↓ |
|---|---:|---:|---:|---:|
| FSBO | 200/200 | 0.5766 | 1.734× | 1.373 |
| TPE | 200/200 | 0.7516 | 1.330× | 2.308 |
| LOLBO | 200/200 | 0.7906 | 1.265× | 2.320 |

- All 200 fixed held-out queries; all three methods complete 1000 charged proposals per query.
- Proposal 0 is the best uncensored runtime among the same 49 imported Bao observations; equality checked by plan_id, runtime and censoring flag. Recovered token representations can differ.
- Only uncensored runtimes update the best-so-far incumbent. Cache hits still consume proposal slots.
- Primary aggregation is exp(mean(log(best_runtime / initial_best_runtime))) across queries; lower is better.
- Average rank uses per-query raw runtimes with average ranks for exact ties; lower is better.
- FSBO seed 123; TPE and LOLBO seed 0. One run per method, no repeated-seed uncertainty estimate.
- SQL timeout differs: FSBO 50 seconds; TPE 100 seconds; LOLBO adaptive.
- Campaigns ran on different dates, hosts and storage; imported Bao timings are historical. This is a descriptive comparison, not a controlled same-hardware timing experiment.
- FSBO is frozen-VAE pretrained transfer GP; LOLBO is the frozen-VAE latent TuRBO variant.
- No query evaluations, workers, or production results were modified.

Reproduce: `python3 build_plot.py`. The script only reads production histories and writes this report directory.
Exports: PNG, PDF, SVG; convergence.csv; per_task_final.csv; per_task_curves.npz; audit.json.
