# Peptide VAE and FSBO adaptation pilot

This is an isolated experiment authorized on 2026-11-17. Production models,
recipes, source corpus and completed runs remain read-only dependencies.
The final quality and slowdown results do not exist until `RESULTS.md` and
`summary.json` report completion. Follow `pipeline_status.json`, `vae/status.json`,
`corpus/status.json`, `fsbo/status.json` and `evaluation/*/*/status.json` for progress.

## Question and comparison

Ten held-out peptide optimization tasks were sampled from the existing test200
split using RandomState(20261117), before examining their results. Task IDs are
in `config.json`. Each task runs three arms with seed 1:

1. Original author VAE, frozen, and original train800 FSBO meta-weights.
2. A VAE further trained on the 800 source tasks, frozen at test time, and FSBO
   meta-weights retrained on its updated source embeddings.
3. The same offline models plus target-specific joint VAE and FSBO adaptation
   after 10 consecutive acquisition rounds without improvement.

All arms retain frozen-wrapper reconstruction/recentering, so comparing arms
2 and 3 isolates adding representation and kernel updates to that wrapper.
They share the existing 100 measured starting peptides per task, batches of 50,
5,000 candidates, a 5,000 acquisition-proposal limit and a shared 5,000 fresh
logical APEX-evaluation limit including recentering and infeasible outcomes.
Duplicates in a task's observation cache cost zero. Similarity is
`1 - Levenshtein(sequence, reference) / len(reference) >= 0.75`.
The decoder uses native BOLT processing, maximum length 30 and at most one C.

## Offline adaptation

The VAE starts from the real author checkpoint, SHA256 recorded in config.json.
It uses the **already completed LOLBO trajectories for all 800 training tasks**,
not 800 newly launched optimization searches. This avoids another 4M oracle
queries and uses their measured sequences and scores.

A single shared VAE visits all 800 tasks in a fixed order, permutation seed 123.
For each source task, recent 50 plus unique feasible top 1,000 source sequences
form an episode. Native LOLBO DKL objective and constraint GPs are initialized
on the current coordinates and trained for 20 epochs. The exact native
`update_models_end_to_end_with_constraints` function then updates the shared
VAE and episode GPs for two epochs at learning rate .001, retaining raw APEX
scores and native length-dependent minibatches. Each task is checkpointed.
This is a new sequential transfer/replay strategy, not the native independent
800-task campaign and not an average of per-task VAE checkpoints.

Afterward, **all 4,053,113 distinct per-task measured source observations** are
encoded again with the adapted posterior mean. Bounds are calculated only from
train800. The original FSBO CPU trainer then runs 100,000 steps with the same
640/160 internal task partition, [128,128] network, RBF kernel, min-max response
scaling, seed 123, and validation-based checkpoint selection as the old model.
The VAE itself sees all 800 source tasks, including the internal FSBO validation
tasks. The ten held-out evaluation tasks never enter these offline stages.

## Online adaptation and coordinates

The VAE and FSBO prior reset to the same offline checkpoints for each test task.
Ordinary FSBO target fitting already occurs at every acquisition, with maximum
1,000 updates and best-loss patience 16. A stagnation trigger adds two epochs
of joint VAE reconstruction/KL + **exact FSBO marginal likelihood** + native
constraint predictive likelihood. Both the VAE and FSBO feature/kernel weights
must change and gradients must remain finite. This is an FSBO adaptation of the
LOLBO strategy; native LOLBO uses its own variational objective GP instead.

After an update all observed target sequences are re-encoded, the target
constraint GP is rebuilt on current coordinates, and the FSBO kernel is refit
before subsequent acquisition. Online adapted FSBO weights initialize later
target fits. Source normalization bounds remain fixed. There is no repeated
100k-step source meta-training or RGPE source-bank training at each trigger.
Recentering consumes the common evaluation budget; VAE/GP training does not
itself request new oracle labels.

The new paired runner uses PyTorch 2.5.1 / GPyTorch 1.15.2 for all arms. It loads
the original FSBO state using GPyTorch's supported ConstantMean migration.
This differs from the older historical FSBO runtime, which is why a new author
control is included. GPU TF32 is disabled. Each task's arms use the same GPU,
and arm order rotates across tasks. Four GPU workers process different tasks.
Other workloads share the GPUs; timing estimates are specific to this setting.

## Measurement, recovery and outputs

Timing separates target fitting, constraint fitting, acquisition, VAE training,
re-encoding, decoding, oracle inference and checkpoint overhead. Per-task wall
time starts at runner entry, includes model loading and file I/O, and excludes
Python interpreter/module import. Offline training and embedding refresh costs
are reported separately and amortized over 10 and 200 tasks.

Model weights, observed embeddings, trust region, RNGs and pending oracle
events are saved atomically. A saved scored pending event is committed after a
fresh-process resume without requesting its oracle labels again. A separate
oracle-request ledger exposes physical requests. Exclusive task locks prevent
duplicate workers. Validation controls forcing an update or stopping early
are accepted only in a separate validation directory, never the evaluation arm.

`code/pipeline.py` waits for the active VAE job, refreshes source embeddings,
trains FSBO, runs the 30 held-out searches, audits observations and budgets, and
writes `paired_results.csv`, `convergence.csv`, `summary.json`, `RESULTS.md` and
`comparison.{png,pdf,svg}`. The driver records a failure rather than inventing
missing results. `historical_summary.json` and `historical_task_timings.csv`
provide supplemental timing from previous FSBO training and the same 10 tasks.
Historical cross-method runtime differences do not isolate adaptation overhead.

Report quality in physical MIC units and improvement from initialization,
not a newly pooled min-max score. Paired task bootstrap intervals cover the ten
tasks only; there is one optimizer seed and no across-seed uncertainty estimate.
A "better VAE" or better optimization is an outcome to measure, not assumed.
Convergence plots retain completed incumbents after proposal exhaustion. The
exports record actual fresh-call support; no additional oracle calls are implied.

Reproduce/continue the driver with system Python:

```
python3 code/pipeline.py
```

Review `config.json` for settings and source identities. Original dependencies
are imported from their recorded locations; experiment code is under `code/`.

## Active comparison policy (2026-11-18)

The active arms are `author_frozen` and `adapted_online` (20 paired task runs). The user removed `adapted_frozen`; its existing results and checkpoints are retained but excluded from scheduling and the final comparison. `config.json` remains immutable to preserve checkpoint identities; `run_policy.json` controls the active subset. Offline training and the online evaluation implementation are unchanged.

## Reuse completed baseline (2026-11-18)

Only `adapted_online` is scheduled. The comparison reuses all ten selected task archives from the completed original FSBO test200 campaign; archive hashes and identical 100-sequence initializations were verified and recorded in `historical_baseline_reuse.json`. New author-control reruns are stopped and preserved separately. Cross-environment runtime ratios are descriptive only.
