# Peptide VAE + FSBO full-source replay rerun

The user requested FSBO retraining from scratch after each VAE update.
This isolated rerun supersedes the 20261117 target-only adaptation pilot.
The completed original FSBO test200 archives are reused, never rerun.

- Same prespecified ten held-out tasks, seed 1, 100 shared initial measurements,
  batches of 50, 5000 proposal cap and 5000 fresh-evaluation cap.
- Offline: shared VAE replay over 800 historical training tasks. Per-task
  response scaling and VAE learning rate 1e-5 address the previous decoder
  collapse. No new oracle labels are obtained during training.
- A fixed source-only decoder check must retain at least 80% of original feasible
  reconstructions and reconstruction loss <=1.25 times the original. Failed
  attempts are preserved and retried from original weights at 10x lower LR,
  at most three attempts. Evaluation never bypasses this check.
- Refresh all 4,053,113 source observations; train initial FSBO for 100000 steps.
- At 10 consecutive acquisition rounds without improvement: update the VAE
  using current-task observations; re-encode ALL 800 source datasets and the
  observed current-task prefix. Recalculate coordinate bounds using source800
  only. Initialize FSBO network, GP kernel/mean/noise and optimizer FROM SCRATCH;
  train 100000 steps using 800 independent source episodes plus current task.
  No observations from any other test task enter this training.
- FSBO uses [128,128], RBF, per-task response scaling, batch50, Adam .001.
  All 800 sources participate in training. Within the previous 160 validation
  source tasks, fixed disjoint support/query observations are reserved for
  checkpoint selection, leaving the remaining observations in training.
  Current task is trained only on measurements already available to its search.
- The selected source-trained prior is refit on the current task before BO.
  VAE, refreshed bounds, GP priors, observed labels and RNGs are checkpointed.
  Full re-encoding and scratch meta-training have their own resumable outputs.
- Four task workers use GPUs 0,1,2,3 after shared offline stages complete.
  Scratch FSBO training uses two CPU threads per active task worker.
  Unrelated Query and Taskset jobs remain running.

Validation uses four real source tasks plus the current task to exercise the
complete online retraining path, fresh initialization, selected model reload,
new coordinates, valid peptide generation, saved-score recovery and resumed
acquisition. Production uses all 800 sources and 100000 steps.

Progress: pipeline_status.json, vae/status.json, reconstruction_gate.json,
corpus/status.json, fsbo/status.json, evaluation/*/*/status.json and
source_refresh/round_*/refresh_status.json or meta_status.json.

Completed baseline archives and old method outputs remain separate. Runtime
comparisons against historical FSBO do not isolate adaptation overhead.
Better optimization is an experimental result, not guaranteed by retraining.

## Training audit corrections

The starting author VAE is byte-identical to the completed peptide baseline. Eight sampled source datasets reproduce their stored frozen-VAE embeddings (maximum difference <5e-7); their labels, references and constraints align. A controlled eight-source-task replay reproduced decoder collapse at learning rate .001. The .00001 replay retained decoder quality over this short check, but the full sequential replay still drifts. Every 50 source tasks now records source-only decoder feasibility/loss, and rejects a degrading attempt early. The driver preserves rejected attempts and retries from original weights at a 10x smaller learning rate, with the preexisting three-attempt limit. Training records reconstruction, objective and constraint losses separately. A 10-minute detached read-only monitor writes to /root/reports/peptide_vae_training_audit_20261118/monitor_latest.json; it cannot send chat notifications.
