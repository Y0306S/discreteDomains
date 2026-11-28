# Peptide FSBO feasibility search

Implementation based on `peptide_target_loss_gate_20round_max2_test10_20261119`.
The evaluator and private search policy live in this directory. Retained offline
VAE/FSBO checkpoints and source data are linked from that campaign; new output
belongs here. The entry point is `code/evaluate.py`.

## Requested behavior

- Recenter starts only after an accepted VAE retraining update, after the refreshed
  FSBO prior and re-encoded coordinates are ready. Rejection, rollback, interrupted
  retraining, stagnation alone, and frozen-VAE rounds do not schedule recenter.
- Recenter uses the latest 50 observations plus up to 200 feasible sequences,
  deduplicated within the top group and sorted by descending observed score.
  It executes one pass (`recenter_epochs=1`). Overlap with the recent group is
  retained. Its cursor is checkpointed with the published models.
- Each acquisition round centers on the historical feasible incumbent. It draws
  5,000 Sobol points and one sparse perturbation mask, decodes the entire pool,
  and checks the existing reference edit-distance constraint. Below 10% feasibility,
  length halves; above 50%, it doubles. The inclusive stable range is 10–50%.
  The length stays in `[0.0078125, 1.6]`, preserving the existing upper limit.
- The Sobol points and mask are reused across length changes. There are at most
  three adjustments per round, with the final pool decoded and checked too
  (at most four pools / 20,000 decodes). Decoder batches contain at most 128 rows.
- Native objective Thompson sampling selects up to 50 candidates from the exact
  feasible pool. If fewer than 50 are feasible, all feasible candidates are retained
  and the shortage is selected from the infeasible pool. Existing remaining-budget
  limits still apply. Predicted constraints cannot veto exactly feasible peptides.
- Selected candidates retain their original decoded sequences; no second stochastic
  decode occurs. Sampling row IDs preserves identity even when different decoded
  peptides share the same latent vector. MIC calls apply only to selected, uncached
  sequences, with the existing pending-event recovery and accounting.
- Trust-region restarts are removed. Native success/failure length updates remain,
  with the length clamped to the same bounds and the restart flag cleared.

Online VAE training data/epochs, rollback gates, 20-round retraining patience,
two-attempt limit, source800 FSBO rebuilding, and evaluation budgets are inherited.
The Top-200 and one-pass changes apply to recentering. `config.json` records the
new search parameters; `parent_provenance.json` identifies inherited assets.

## Validation

All 32 CPU regression tests passed, including the inherited retraining-schedule
and retention-gate tests. See `validation/tests.json`.

The real VAE/FSBO GPU integration check decoded full pools of 5,000 candidates
on `pep_117`. Length changed from 0.8 to 0.4 to 0.2, with feasible rates
0%, 0.48%, and 21.42%. It selected 50 feasible candidates, retained their decodes,
and made zero MIC calls. See `validation/real_search_5000.json`.
This integration check used a 16-update target fit; it does not measure sequential
optimization performance or validate a new online training run.

Run regression tests:

```bash
/root/data/cys/envs/bolt_clean_venv/bin/python -m unittest discover \
  -s /root/data/logs/peptide_fsbo_feasible_search_20261122/code -p 'test_*.py' -v
```

Run the new evaluator with the chosen GPU visible to the process:

```bash
/root/data/cys/envs/bolt_clean_venv/bin/python \
  /root/data/logs/peptide_fsbo_feasible_search_20261122/code/evaluate.py \
  --task pep_117 --arm adapted_online
```

The same-ten rerun is managed by `code/pipeline.py` on LLMOptimizer3.
`pipeline_status.json`, `workers/slot*.json`, and `RESULTS.md` track progress.
See `execution_policy.json` for allocation and `implementation.patch` for the changes.
