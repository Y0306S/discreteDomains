# Peptide optimization: updated frozen-VAE FSBO

All 200 new FSBO test-task archives passed validation. The new run is the batch-50 local/recenter wrapper with author VAE weights frozen, seed 1, similarity threshold 0.75, 100 initialization sequences, and a maximum of 5000 proposals or 5000 fresh evaluations shared with recentering.

The plot compares the same 200 held-out tasks and identical starting sequences/scores. New archives were read and all member SHA256 hashes checked, feasible labels independently recomputed from Levenshtein similarity, budgets reconciled, and final incumbents checked. Prior FSBO, LOLBO and TPE curves reuse the validated November 1 extraction; these optimizers were not rerun.

| Method | Final mean best feasible MIC (lower is better) | Mean per-task improvement from initialization |
| --- | ---: | ---: |
| FSBO (previous) | 20.9009 | 59.06% |
| FSBO (new, frozen VAE) | 29.7457 | 45.26% |
| LOLBO | 34.1429 | 39.51% |
| TPE | 63.1201 | 0.10% |

The new FSBO improves on initialization in 196/200 tasks. It improves the aggregate terminal MIC over LOLBO, but performs worse than the previous FSBO. The previous FSBO continues improving more strongly after about 3000 fresh evaluations; the new curve flattens earlier. This comparison does not isolate the effect of any individual wrapper change.

New FSBO used 3682–5000 fresh calls per task (mean 4664.835). Proposal-limited completed runs retain their final measured incumbent through x=5000, matching the previous plot convention; no extra evaluations are implied. Fresh calls include recentering, while cached duplicates cost zero. Each logged batch is revealed at its endpoint.

Shading: pointwise 95% task bootstrap intervals, 2000 paired resamples. One optimizer seed per task; not across-seed uncertainty. RGPE is omitted because that campaign remains incomplete. The old TPE baseline is shown; this is not a new frozen-wrapper TPE run.

Paired terminal results for the new FSBO (wins/losses/ties; lower MIC wins): {"FSBO": {"new_wins": 55, "new_losses": 140, "ties": 5}, "LOLBO": {"new_wins": 115, "new_losses": 84, "ties": 1}, "TPE": {"new_wins": 196, "new_losses": 0, "ties": 4}}.

Artifacts: `peptide_convergence_updated_fsbo.{png,pdf,svg}`, `convergence_updated_fsbo.csv`, `fsbo_local_task_summary.csv`, `per_task_curves_updated_fsbo.npz`, `updated_fsbo_audit.json`.

Reproduce from source archives: `/root/data/cys/baseline/envs/bl_fsbo_exact/bin/python plot_updated_fsbo.py`. To redraw the already validated extraction without rereading archives, append `--reuse-validated`.
