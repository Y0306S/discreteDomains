# Query rerun with a fine-tuned VAE and retrained FSBO

User request: fine-tune the current Query VAE on **80 training tasks**, retrain FSBO in the resulting latent space, then restart the same ORPO+FSBO experiment from saved BOLT checkpoints. The existing split is 800 training / 200 held-out tasks. This variant honors the literal requested count of 80; it does not silently substitute 800.

The exact 80 IDs and selection procedure are in `training/config.json`: proportional template stratification, seed 123, selected without consulting test results. Of these, 64 supply gradient updates and 16 provide training-side validation/checkpoint selection for both models. There are 2,663 canonical, unique, uncensored measured training plans. No held-out task is used for this retraining or its checkpoint selection. The original pretrained model is the same starting point as the prior run.

## Training

1. Fine-tune the original native `VAEModule` checkpoint, retaining its tokenizer, vocabulary, architecture, 64-dimensional posterior means, reconstruction/KL objective, and native codec. AdamW: learning rate 0.0001, weight decay 0.01, betas 0.9/0.99; up to 30 epochs, 10% learning-rate warmup, validation patience 8. Select by validation loss, including the initial model as the baseline. Do not promote if no trained checkpoint improves validation.
2. Re-encode all selected source plans with the selected VAE on the same GPU/highest-precision serving path, retaining measured runtimes and source-history hashes. Recompute coordinate normalization bounds. Never mix the new VAE with old FSBO latent features.
3. Train FSBO for the original 100,000-update budget and the unchanged existing FSBO meta-training method: 128/128 hidden sizes, RBF kernel, no ARD, batch 50, Adam 0.001, per-task response normalization, validation every 100 updates, minimum predictive NLL checkpoint selection, verified reload.

`training/pipeline.py` runs the sequence durably. Stage logs and status are in `training/`; selected checkpoints are `training/vae/best.ckpt` and `training/fsbo/weights`. Each checkpoint is hash-pinned before use.

## Rerun

- Same 200 held-out queries, task ordering, seed, BOLT and ORPO checkpoints.
- Same 80 total rounds × 10 evaluations, including three BOLT rounds: 30 saved warm measurements + 770 new optimization evaluations per task.
- Import all 6,000 saved BOLT measurements unchanged, preserving generation/random-fallback labels, plans, runtimes, censoring and provenance. Validate every plan and frozen pool. Archive old optimization histories in the untouched parent campaign; do not copy those observations into this rerun.
- Same retrieval schedule, selector, cooldown, ORPO sampling, generation limits, fallback sequence, candidate pool, and 50-second SQL timeout.
- FSBO uses the new encoder, newly embedded corpus, normalization bounds and new meta-trained weights, frozen during evaluation.
- **Representation compatibility:** the existing retrieval source GPs remain frozen in their original latent coordinates. Retrieval therefore retains the original VAE encoder through a separate GPU encoding operation and cache; it must never use the new FSBO embeddings with the old source GPs. This preserves the previous retrieval ranking procedure. A regression test keeps those features separate, and a live cross-host smoke test compares both serving encoders against their matching saved source corpora.
- Same resource topology: 21 optimizer workers / private PostgreSQL instances, 12 LLM replicas, 16 VAE service workers. Existing verified PostgreSQL data and temp paths are retained.

`training/promote.py` requires real model reload checks, unit/recovery tests, fresh training-only live ORPO/FSBO batch smoke tests, long-context inference, cross-host transport/locking and VAE equivalence checks, and launch pins before starting workers. Failed checks leave the rerun paused and preserve the original results.

## Recovery and interpretation

The existing 600-second remote watchdog is retained, with its restricted tick entry point routed to the new campaign after promotion; no second watchdog is started. Its scheduler remains under the parent campaign's `operations/recovery/`, while the active recovery code and incident logs live here. The existing local read-only monitor is updated to this target. The shell scheduler performs defined recovery checks; it does not create future chat messages or independent model reasoning.

VAE reconstruction/validation improvement is an internal training result, not evidence of better query optimization. Report this as a distinct retrained variant and compare actual held-out trajectories. Do not replace or relabel the parent's results or claim an improvement over baselines before the rerun provides evidence.

Parent: `/root/data/cys/baseline/campaigns/query_our_fsbo_test200_20261116`.
