# Peptide VAE training audit

The adaptation starts from the correct frozen author VAE used in the completed peptide FSBO baseline. All three recipes point to the same checkpoint, and the checkpoint file, experiment configuration and frozen embedding manifest agree on SHA256 `e360dc044e2695f5c329d1e8ed4ce9b42b7091f1f5477926e3121e138b7e2edd`.

Eight source tasks sampled with seed 4521 passed payload hash verification, reference checks, aligned observation lengths, recomputed Levenshtein constraint checks, finite labels, and frozen-VAE embedding reproduction (maximum absolute difference less than 5e-7). The train800 and test200 sets are disjoint. The inspected current VAE checkpoint has finite weights and matches its configuration hash. These checks do not audit every training observation.

## Controlled reproduction

Starting independently from the same author VAE, run the same first eight historical source tasks, seed 123, with the native constrained joint-update function semantics and two epochs per task. Evaluate all variants on the same fixed source-only set of 32 tasks, four initialization peptides per task (128 peptides). The original VAE reconstructs 118/128 into peptides meeting their reference similarity constraints.

| Response scaling | VAE learning rate | Feasible after 1 task | After 4 tasks | After 8 tasks | Final reconstruction loss |
|---|---:|---:|---:|---:|---:|
| Raw scores | .001 | 81/128 | 0/128 | 0/128 | 2.27493 |
| Per-task min-max | .001 | 91/128 | 14/128 | 23/128 | .31236 |
| Raw scores | .00001 | 118/128 | 116/128 | 113/128 | .0003514 |
| Per-task min-max | .00001 | 118/128 | 116/128 | 110/128 | .0003528 |

The large learning rate reproduces collapse even when scores are scaled. Raw-score training magnifies the objective gradient: on the first minibatch, objective encoder-gradient norm is 26.52 versus reconstruction 0.516; scaling reduces the objective norm to 0.129. This is direct evidence for excessive representation updates and loss-scale imbalance. It does not establish that min-max scaling always outperforms raw scores at low learning rates.

The completed old adapted VAE has 0/128 feasible reconstructions and reconstruction loss 2.21565 on the same source-only check. The first lower-rate long-run snapshot after 264 source tasks had 106/128 feasible reconstructions; after 308 tasks it had 97/128. This demonstrates ongoing gradual drift, so the eight-task success cannot be extrapolated to all 800 tasks.

## Corrections and monitoring

- Keep the original frozen VAE and all completed baseline results immutable.
- Use scaled objective labels and lower VAE learning rate in the isolated rerun.
- Log reconstruction, objective and constraint losses separately; reject nonfinite VAE/GP weights or loss components.
- Check source-only decoder retention every 50 tasks and after full offline training. The configured gate requires at least 80% of original feasible reconstructions and at most 1.25 times its reconstruction loss.
- Preserve a rejected attempt and restart from the original VAE at 10x lower learning rate, up to the existing three-attempt limit. Do not export embeddings or start test evaluation with a rejected VAE.
- Keep the requested per-trigger scratch FSBO training on the re-encoded 800 source tasks plus current-task observations.
- Real-model smoke and online-replay/resume validation pass. A deliberately injected decoder-quality failure was correctly rejected with its atomic checkpoint retained in an isolated validation directory.
- A detached Python monitor checks progress, controller liveness, reconstruction and scratch-training provenance every 10 minutes. It writes `monitor_latest.json`, `monitor_history.jsonl`, and `/root/.codex/loops/peptide-vae-training-10m/state.json`. It does not provide future chat notifications or autonomous Codex reasoning.

The monitor's latest report is the current operational status. These are small fixed source-only diagnostics, not proof of better held-out optimization. Source-sequential VAE replay is a new adaptation procedure, distinct from resetting a VAE independently for each task in ordinary LOLBO optimization.

## Live follow-up

The new intermediate guard rejected the .00001 attempt at source task 350. The checkpoint at task 345 reconstructed 79/128 feasible peptides, below the 94.4/128 retention threshold. The controller preserved the rejected attempt and restarted from the original author VAE at learning rate .000001 (attempt 1). This is a fresh offline training attempt, not continued fine-tuning of the rejected checkpoint. See monitor_latest.json for the current progress and decoder check.
