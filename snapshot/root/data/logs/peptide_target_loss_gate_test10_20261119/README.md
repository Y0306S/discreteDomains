# Peptide VAE target-loss rollback

This version adds a target-retention acceptance check to the completed
`peptide_apexgo_recipe_fsbo_test10_20261118` implementation. The parent code,
checkpoints, configuration and scientific results are preserved. This directory
has its own code/configuration identity and output paths. It reuses the same
offline VAE/FSBO assets, ten target tasks, initialization100, batch50 and
4,000-proposal/fresh-evaluation limits. The same-ten production campaign was launched on 19 November 2026 on
LLMOptimizer3, with four parallel workers. See LAUNCH.md for allocation and status paths.

## Meaning of mean target loss

The current task's fixed, deduplicated initialization sequences are fed through
the VAE in evaluation mode, with dropout disabled, fp32 and three fixed paired
posterior seeds (8131, 8132, 8133). For each seed, the native model computes
`reconstruction cross-entropy + 0.0001 * KL divergence`; we average the total
over the three seeds. The native reconstruction loss averages token positions
in the padded batch. The entire fixed anchor batch and its order are identical
before/after. Components and per-seed values are also logged.

This is a sequence-reconstruction/latent regularization metric. It is neither
MIC nor the FSBO prediction error. These initialization sequences were already
observed and can also appear in training, so this is a retention check, not
independent held-out generalization validation. No later unobserved sequences,
other test-task outcomes or extra oracle measurements are used.

## Acceptance and rollback

1. Before retraining, save the complete current generation in
   `evaluation/adapted_online/<task>/vae_updates/round_NNN/pre_update.pt`.
   It includes VAE, FSBO prior/scaling, encoded observations, constraint model,
   trust-region/search state and RNG state. Completed observations and costs
   remain in the optimization history.
2. Measure fixed target loss and the existing source probe before training.
   Keep the target measurement before the first online update as an immutable
   task baseline in the checkpointed state.
3. Train the existing two VAE epochs. Log each executed optimizer step's total,
   reconstruction, KL, gradient norm and scaler value. Nonfinite training loss,
   gradients or resulting weights reject the update.
4. Measure the candidate on exactly the same anchors and seeds. Require both
   `candidate_loss <= pre_update_loss + 1e-12` and
   `candidate_loss <= initial_baseline_loss + 1e-12`. There is no percentage
   degradation allowance; the absolute epsilon handles numerical equality.
   The original source validity/loss acceptance conditions must also pass.
5. On rejection, restore every VAE state-dict tensor and buffer exactly, clear
   candidate gradients, and discard the update-local optimizer/scaler. Skip
   FSBO retraining and re-encoding, so existing downstream coordinates/models
   remain consistent. Continue the search with the retained VAE.
6. Only accepted VAEs trigger the original source800+current-task FSBO rebuild,
   coordinate normalization refresh, target re-encoding and constraint refit.
   Durable search checkpoints are written at the original event boundaries;
   interrupted rebuilds cannot publish a mixed generation. Infrastructure
   exceptions restore attempted VAE weights and propagate rather than being
   reported as loss-based rejection.

Each attempt saves `decision.json` (before/after/baseline metrics and reasons)
and `retained_vae.pt`. The normal result/checkpoint also records the decision.
Validation probes preserve search RNG and use the same device/precision/batch
identity for all compared measurements. Dependency hashes are part of run
identity; old-run checkpoints cannot silently resume under the new rule.

## Verification

- Eleven CPU behavioral tests pass: accept improvement/equality; reject target
  degradation, cumulative baseline drift, source failure and nonfinite weights;
  restore parameters and buffers; restore before propagating infrastructure
  errors; preserve probe RNG/mode; reject changed anchor sets; execute/report a
  checked optimizer step.
- The new gate was exercised with the real saved pep_999 pre-online and final
  VAE tensors. It measured target loss **0.000309528773 → 0.000895203712**,
  rejected the candidate and restored every tensor exactly (weight delta0).
  No training or oracle evaluation occurred. The source probe in this replay
  was an explicitly passing fixture to isolate the target gate.
- The new evaluator imports successfully; changed modules compile. A live A16 integration test additionally passed a real VAE update, a
  four-source+current scratch FSBO rebuild and downstream decoding/scoring.
  The production configuration uses all800 sources and100000 FSBO steps.

The real-checkpoint replay does not reconstruct missing historical intermediate
updates and does not establish improved MIC. That requires a new controlled
optimization run under this version.

Verification commands:

```bash
/root/data/cys/envs/bolt_clean_venv/bin/python code/test_target_gate.py
/root/data/cys/envs/bolt_clean_venv/bin/python code/validate_saved_pep999.py
```

The evaluator entry point is `code/evaluate.py --task pep_999 --arm adapted_online`.
The same entry point supports each of the ten configured tasks. The production controller queues all ten configured tasks on the recorded
resource allocation; results will be compared automatically on completion.

See `validation/saved_pep999_gate.json`, `validation/implementation.json` and
`parent_provenance.json` for verification and artifact identities.
