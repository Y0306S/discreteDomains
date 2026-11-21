# Peptide convergence: completed test200 baselines

Read-only extraction from 600 completed archives: 200 each for FSBO, LOLBO and TPE. All archive member checksums and recorded recipe identities passed. No model state was loaded and no oracle evaluations or training were run.

The objective is best feasible APEX bacteria-0 MIC (negative of the maximized score); lower is better. All three methods use the same 200 held-out references, same revalidated 100 starting sequences and scores per task, similarity threshold 0.75 and seed1.

The x-axis counts fresh post-initialization logical oracle evaluations, including LOLBO recentering; cached duplicate scores cost zero. Each logged batch is revealed at its endpoint. Production stops at either 5000 acquisition proposals or 5000 shared fresh evaluations, so runs ending earlier in fresh-evaluation count retain their final measured incumbent through x=5000. This does not imply new measurements or simulate continuation. Exact terminal costs and support counts are exported.

The right panel averages 100 × (1 − current best MIC / initial best MIC) across tasks. Shading is a pointwise 95% percentile bootstrap CI over 200 tasks, with 2000 paired resamples. There is one optimizer seed per task; these bands do not estimate across-seed variation. RGPE remains incomplete and BOLT SFT is not peptide optimization, so neither is plotted.

| Method | Tasks | Final mean best MIC | Mean per-task improvement |
|---|---:|---:|---:|
| FSBO | 200 | 20.9009 | 59.06% |
| LOLBO | 200 | 34.1429 | 39.51% |
| TPE | 200 | 63.1201 | 0.10% |

Files: `peptide_convergence.{png,pdf,svg}`, `peptide_convergence_mic.{png,pdf,svg}`, `convergence.csv`, `task_summary.csv`, `per_task_curves.npz`, `audit.json`. Reproduce with `python3 plot_convergence.py`.

Follow-up TPE audit: `/root/reports/peptide_tpe_diagnosis_20261101/README.md`. This TPE curve is the global independent 256-dimensional latent-space configuration: only 171/888,995 fresh measurements were similarity-feasible. Its poor convergence should be interpreted with that configuration and feasibility failure explicitly stated.
