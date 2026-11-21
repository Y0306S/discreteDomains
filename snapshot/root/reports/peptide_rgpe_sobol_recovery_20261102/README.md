# Peptide RGPE Sobol recovery — November 2, 2026

The ten failed tasks had 1,150 saved observation rows after round 15. Ranking requested a joint normal vector of that size; legacy PyTorch's Sobol engine only supports 1,111 dimensions (the default normal transform also rounds odd dimensions up to even).

The separate, hash-verified repair in `code/` keeps the original campaign recipe, parent code, acquisition amendment, source models, GP fitting, ranking formula, posterior factorization, RNG checkpoints, and observation history. Legacy draws through 1,110 dimensions are bit identical. Larger vectors get seeded scrambled Sobol uniforms from the existing modern SciPy environment in a persistent CPU-only helper, using the legacy normal transform and full posterior Cholesky. Above SciPy's 21,201-dimension table, independently scrambled uniform blocks are combined before the full posterior transformation. This changes the Sobol stream only where the old implementation could not run.

Six regression checks passed in the actual legacy model environment: exact legacy output compatibility, the 1,150-dimension failure and repeatable seeds, later budgets through 25,000 dimensions, empirical signed joint covariance, a real pinned source GP on saved 1,150-row coordinates, and narrowly scoped retry behavior. The 1,150-dimensional draw also passed on LLMOptimizer4.

All ten original runtime databases were backed up using SQLite backup before restart. Each saved identity and row/coordinate alignment was verified; none had a pending oracle request. See `before.json`, `before/*.sqlite`, `validation.json`, and `validation.log`.

The original dispatchers were replaced without signalling their optimizer children; all 36 live workers were adopted. The new plan preserves the 20 local + 26 remote worker caps and existing quota/memory guards. Remote reservation was subsequently adjusted from 8 to 7 GiB per worker after the 8 GiB floor held two restarts; the measured peak was approximately 3.1 GiB, the dynamic 1.75× peak allowance remains in effect, and 16 GiB host headroom is preserved. See `memory_adjustment.json`. The original ten tasks have first priority. Adopted legacy workers are retried with the repair only if their log contains this exact Sobol dimensionality error. Any error in a repaired worker, or unrelated legacy error, still pauses new launches.

The original `dispatch/plan.json` remains intact. The active campaign plan is `dispatch/sobol_repair_plan.json`, recorded in `dispatch/active_plan.json`. The manifest and validation files are replicated at the same `/root/reports/peptide_rgpe_sobol_recovery_20261102` path on LLMOptimizer4; optimization data are shared under `/root/data`.

Run `python /root/reports/peptide_rgpe_sobol_recovery_20261102/live_audit.py` for live status. Inspect `live_audit.json`, both dispatcher logs, task `sobol_repair.json` activation records, and `dispatch/logs/*.sobol-repair.log` for runtime evidence. A restarted task can remain at round 15 while refitting its target and ensemble; a live process alone does not establish completion of its next optimization round.

Subsequent authorized expansion: `/root/reports/balanced_bora_rgpe_20261102/README.md`. Use campaign `dispatch/active_plan.json` for the latest plan; totals became 15 BORA and 53 RGPE workers.
