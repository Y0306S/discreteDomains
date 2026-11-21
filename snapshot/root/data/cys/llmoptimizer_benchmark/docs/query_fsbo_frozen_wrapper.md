# Query FSBO with the original frozen peptide acquisition wrapper

`tools/query_fsbo_eval.py` adds the first peptide FSBO run's native batch
Thompson sampler to the new query model. The model process reuses the same
`fsbo_target_refit.fit_target` implementation. The acquisition process imports
the exact saved peptide CPU `generate_batch`, `MaxPosteriorSampling`, and
`TurboState` implementation under a private module namespace. Existing
peptide code, query training code, readiness services and VAE services are
unchanged.

The VAE is frozen. There is no end-to-end VAE training and no call to the
stall-triggered recentering phase. TuRBO still adjusts its width, selects the
best measured plan as the ordinary acquisition center, and resets its width
if the trust region collapses. This is the behavior of the original peptide
FSBO wrapper. Encoding a newly measured plan once into the fixed coordinate
system is ordinary observation ingestion, not a recentering phase.

## Query-specific behavior

- Query-only meta-checkpoint and source-corpus normalization; 64 input
  dimensions. The saved author query VAE produces canonical posterior means.
- Import all 49 official Bao observations before acquisition. Exact target
  fitting uses the first distinct uncensored canonical plan, including the
  one-observation case supported by the capacity pilot.
- Maximize negative runtime. Refit the GP and its feature network from the
  query meta-checkpoint each round, with target min/max scaling. Maximum
  10,000 Adam updates, learning rate 0.001, patience 16, minimum improvement
  0.0001, best measured likelihood restored.
- Default query budget matches the source query protocol: 1,000 charged
  proposals after 49 imported observations; batch size 10; 5,000 candidates.
  Duplicate decoded plans consume proposal slots and use the measured cache.
- Fixed 50-second execution cap. Timeouts and query resource errors remain
  censored in the history and cache; they are excluded from exact FSBO fitting
  and cannot replace the measured incumbent. This does not introduce a
  censored-likelihood GP or the query LOLBO model's dynamic timeout rule.
- Reject cross joins using the existing query VAE service. Store executed
  plans canonically, retain proposed raw latents for provenance, and fit at
  their frozen encoder means. There is no peptide similarity constraint.
- Reuse the existing query execution oracle, checkpoint/RNG recovery, and
  transactional result journal. A committed SQL result is reused after a
  crash; an unresolved SQL call pauses recovery instead of being repeated.
- One CPU coordinator and model subprocess share the first CPU of each
  admitted three-core database worker. Inference uses the existing ten VAE
  services. Admit new tasks only with at least 32 GiB memory headroom.

## Preparation and operation

Preparation writes an immutable recipe, canonical Bao task data, pinned code
and model checksums, and a `worker_plan.json` with all 16 remote commands.
It does not launch a campaign:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /root/data/cys/envs/bolt_clean_venv/bin/python tools/query_fsbo_eval.py prepare \
  --campaign /root/data/logs/query_fsbo_test200_20261106_v2
```

The prepared workers run on LLMOptimizer4, using their existing private
PostgreSQL databases and the query VAE reverse forwards. For example:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /root/data/cys/envs/bolt_clean_venv/bin/python tools/query_fsbo_eval.py run-worker \
  --campaign /root/data/logs/query_fsbo_test200_20261106_v2 --worker w07
```

Each worker takes an exclusive database endpoint lock and one task lock at
a time. Completed tasks are skipped; failed tasks require inspection.
Creating `STOP` in the campaign directory prevents admission of another
task. `run-task --task ID --stop-after-batches 1` supports bounded checks and
recovery validation. Component recipes must select a training query with
`prepare --component-task ID`; production recipes use the held-out test200.

Adding the wrapper does not by itself launch the full optimization campaign.
