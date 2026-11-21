# Peptide convergence: four methods

2026-11-17T16:32:14.241797+08:00

| Method | Tasks | Score at 1000 ↑ | Mean MIC at 1000 ↓ | Mean improvement at 1000 ↑ |
|---|---:|---:|---:|---:|
| FSBO (frozen VAE) | 200 | 0.8967 | 60.9103 | 2.82% |
| LOLBO | 200 | 0.9024 | 58.5990 | 6.18% |
| TPE (global) | 200 | 0.8911 | 63.1201 | 0.10% |
| SemanticOPT | 200 | 0.9806 | 25.7932 | 49.18% |

- All 200 held-out tasks; identical 100 starting peptides and measured scores checked across methods.
- FSBO uses the latest completed frozen-author-VAE local/recenter run (November 2 campaign). LOLBO uses completed native V5. TPE uses completed global independent latent-space TPE; prepared TPE-TR has not launched.
- X counts fresh logical APEX calls, including infeasible outcomes and recentering. Cached duplicates cost zero. Batch outcomes appear together at their endpoints.
- All baselines have at least 1000 fresh calls per task, so the left panel uses a common fully observed budget.
- Completed baseline tasks retain their last measured incumbent after exhausting either 5000 acquisition proposals or 5000 fresh calls; no continuation is simulated. CSV exports actual task support.
- SemanticOPT ends at 1000; no extension to 5000.
- Only feasible measurements update incumbents or enter normalization pools. Min–max bounds use pooled measurements from all four full trajectories, dropping slowest floor(20%) for bounds only. Scores are retrospective and method-dependent.
- SemanticOPT includes 136255/200000 constrained random fallback proposals.
- Global TPE has very low similarity feasibility (171/888995 fresh measurements in the prior audit); its nearly flat curve describes this configuration.
- Baseline history hashes match the previous full archive-member audit; regenerated 25-call checkpoints match the previously validated curves exactly. SemanticOPT constraints and uniqueness were checked for all observations.
- One optimizer seed per task: baselines1, SemanticOPT42. Different devices/dates and numerical execution settings; no replicated-seed uncertainty estimated.
- Production data opened read-only; no inference, oracle calls or training performed.

Plots: peptide_convergence, peptide_relative_mic, peptide_improvement, peptide_mean_mic (PNG/PDF/SVG).
Reproduce: `python3 build_plot.py`. Cached validated extractions avoid rereading unchanged baseline archives.
Exports: convergence.csv, per_task_comparison_at_1000.csv, per_task_curves.npz, normalization_bounds.csv, audit.json.
