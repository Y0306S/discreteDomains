# pep_999 VAE loss and rollback investigation

The saved evidence supports target-specific VAE degradation, not a demonstrated numerical collapse. A new CPU-only comparison of actual saved checkpoints found that target evaluation loss increased 2.89×, despite stable source retention metrics. Reconstruction still works for nearly all known target sequences. This can contribute to poor search, but does not establish how much of the MIC gap was caused by VAE updates versus the changed FSBO prior and search trajectory.

## Direct checkpoint comparison

The same 100 initialization sequences from pep_999 were evaluated before and after online tuning, with three paired seeds, no dropout, no new training and no oracle calls. These sequences were already observed and used during optimization: this is a retention diagnostic, not an independent held-out generalization test. Computation uses CPU fp32, so these losses should not be numerically equated with the original bf16 training losses.

| Saved VAE | Mean target evaluation loss | Exact autoregressive reconstruction | Feasible reconstruction | Feasible fixed local probes, length0.1 | Length0.2 | Length0.8 |
|---|---:|---:|---:|---:|---:|---:|
| Author frozen | 0.0003122 | 300/300 | 300/300 | 121/128 | 41/128 | 0/128 |
| Before online tuning | 0.0003095 | 300/300 | 300/300 | 128/128 | 85/128 | 0/128 |
| After online tuning | 0.0008952 | 298/300 | 299/300 | 120/128 | 78/128 | 0/128 |

Both before- and after-online checkpoint weights are finite. The final target loss increase comes from reconstruction cross entropy; mean KL falls slightly. Teacher-forced token/string argmax accuracy stays at100%, illustrating that argmax accuracy alone misses deteriorating probability calibration and autoregressive failures. The final VAE is not completely broken, but preserves the target distribution less well.

The local decoding test fixes the reference/incumbent/initial anchors, random seeds, sparse masks and perturbation scale across models. At the native starting trust-region length0.8, all three VAEs produced0/128 feasible samples. Smaller perturbations dramatically improve decoder validity in this diagnostic. These samples were not selected by the GP, do not reproduce the historical candidate trajectory, and have no MIC measurements. The figures support testing a tighter neighborhood; they do not establish an optimal trust-region setting or better MIC.

## What the seven actual update logs show

Each pep_999 update trained for two epochs on128–154 unique selected sequences, below the batch size512: only two attempted optimizer steps per update. Training uses dropout and random posterior samples, and bf16 with GradScaler. The saved first/last loss values are stochastic training-mode observations, not evaluation of the same checkpoint under paired randomness. The log calls them optimizer_steps, but does not record actual GradScaler-skipped steps or gradient norms. There is no complete within-step history of reconstruction/KL components.

| Update round | First training loss | Last training loss | Source retention check change | Accepted |
|---|---:|---:|---:|---|
| 10 | 0.0003763 | 0.0006905 | -0.029% | True |
| 20 | 0.0005678 | 0.0003694 | -0.225% | True |
| 30 | 0.0004563 | 0.0008741 | +0.351% | True |
| 40 | 0.0003490 | 0.0003268 | -0.226% | True |
| 50 | 0.0004056 | 0.0003584 | -0.329% | True |
| 62 | 0.0003705 | 0.0003307 | -0.336% | True |
| 72 | 0.0010153 | 0.0003859 | +0.390% | True |

All logged pep_999 losses are finite. Source probe validity remained128/128 at every update, with source loss changes below0.4% in magnitude per update. This is inconsistent with an obvious numerical blow-up. It does not rule out cumulative target forgetting, which the new checkpoint test detects.

## Rollback already implemented, and its limits

The online update copies the VAE state before training. It restores those weights when the source-only gate fails. Acceptance requires at least80% of the larger pre-update/author source validity count and at most1.25× the smaller pre-update/author source loss. With128 feasible source reconstructions before an update, the count gate can allow a fall to103, so it is not a strict retention guarantee.

There is a real rollback example in this same campaign: pep_673 round69. Its training loss fell from0.0003538 to0.0003469, but the source retention loss rose56.0%; the gate rejected it and the logged post-rollback weight change was0. Thus a rule based only on falling training loss would have accepted an update that the existing evaluation check correctly rejected.

The source metric is named reconstruction_loss in the audit, but the called vae_forward returns total reconstruction-plus-KL loss. The components should be logged separately. The source gate covers32 source tasks and128 sequences, not pep_999, so it cannot certify retention of the current target or feasibility in its search neighborhood.

Nonfinite VAE loss currently raises an exception. It does not take the normal rejected-update rollback branch. GradScaler can skip nonfinite-gradient steps, but skipped-step counts are not logged. The code should distinguish numerical rejection from infrastructure failures rather than treat every exception as model degradation.

## Recommended transactional acceptance policy

1. Before attempting an update, retain a durable pre-update checkpoint and baseline measurements. Preserve the current known-good VAE, FSBO prior, normalization, embeddings, constraint model and trust-region state as one consistent generation. Keep probe RNG independent of search RNG.
2. Log reconstruction and KL separately, finite loss/gradients/weights, gradient norm, scaler value and actual optimizer steps. On nonfinite values or a confirmed numerical failure, reject the attempted update and restore the pre-update generation. Propagate storage/device faults distinctly.
3. Evaluate the old and candidate VAE on exactly the same fixed source set and target-known anchor set, using paired random seeds and evaluation mode. Include total loss, reconstruction loss, exact/feasible autoregressive reconstruction and a fixed local decoder probe. A loss increase on a fixed set is much more meaningful than last-training-loss/first-training-loss.
4. Compare with both the immediately preceding accepted checkpoint and a fixed target-retention baseline from before online tuning, to catch gradual cumulative drift. Use numerical floors and measured repeat variability for near-zero reconstruction losses. Calibrate acceptance margins on development tasks; do not tune a threshold after seeing held-out pep_999 MIC outcomes. Target diagnostic anchors must come only from already observed sequences, never future proposals or the frozen run's later solution. A target subset withheld from VAE gradient updates can provide a stronger generalization check.
5. Accept only after those checks pass. Then re-encode and rebuild the matching downstream models. If a numerical failure occurs during rebuilding, retain the old consistent generation. On rejection, continue BO from the old checkpoint and record the reason, attempt and counters. Do not roll back measured observations, erase oracle costs or change the historical incumbent.

Target-retention checking would flag the final-versus-initial loss drift shown here under a reasonable calibrated criterion. Exact intermediate rejection decisions cannot be reconstructed: individual pre/post-update VAE tensor snapshots and target probe results were not retained. Therefore it is not established that the proposed policy would have rejected a specific historical update, prevented the plateau or achieved the frozen MIC.

For this task, rollback is a useful guardrail but not the entire fix. Feasibility was already poor before the first update; the native wide-neighborhood probe fails even for the author VAE. The next controlled experiment should separate initial adaptation, online updates and latent search radius, while retaining identical evaluation accounting. No training policy or scientific result was modified in this investigation.

[Loss and decoder comparison](vae_loss_and_decoder_probe.png) · [Raw checkpoint probe](vae_cpu_probe.json) · [Loss gate audit](vae_loss_gate_audit.csv) · [Metric CSV](vae_checkpoint_metrics.csv)
