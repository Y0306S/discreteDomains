# Peptide TPE with an adaptive trust region

`tools/peptide_tpe_tr.py` implements a separate `TPE-TR` baseline. Each task has
exactly one outstanding latent recommendation: TPE proposes, the frozen VAE
decodes one peptide, the oracle evaluates it (or the task cache supplies its
previous score), and only then are TPE and the region updated.

This uses Optuna's constrained TPE density-ratio selection. It has no GP,
Thompson sampling, reconstruction stage, VAE fine-tuning, or parallel proposals
within a task. Different tasks can still run on separate evaluation workers.

The original TPE code, recipes, archives, and results are retained. The new
variant must use a new campaign directory and must not resume the old TPE
checkpoint under a different acquisition policy.

## Protocol

The recipe inherits the same held-out 200 tasks, seed 1, 100 measured feasible
initial peptides per task, frozen author VAE, reference-specific similarity
threshold of 0.75, global latent bounds of `[-4, 4]` in all 256 dimensions,
5,000 acquisition proposals, and maximum 5,000 fresh oracle evaluations.
Initial measured scores are imported without new oracle calls. Cached duplicates
consume a proposal but not a fresh call, as in the original baseline.

Similarity is checked on the decoded peptide. Constrained TPE learns from both
feasible and infeasible measurements; a latent box cannot itself guarantee
sequence similarity. Infeasible scores never determine the center or count as
improvements. The reported incumbent always satisfies the similarity constraint.

## Region and TPE observations

The center is the latent coordinate of the best observed feasible point. At
initialization this is the frozen encoding of the best measured seed; later it
is the actual latent proposal associated with an improved feasible peptide.

| Setting | Value |
|---|---|
| Initial side length, as fraction of global width | 0.8 |
| Initial radius before clipping | 3.2 per coordinate |
| Minimum / maximum side length | 1/128 / 1.6 |
| Consecutive qualifying improvements before expansion | 3 |
| Consecutive non-improvements before contraction | 32 |
| Relative improvement threshold for success counters | 0.001 × absolute prior best score |
| Expansion / contraction factor | 2 / 0.5 |
| Proposals awaiting evaluation per task | 1 |
| TPE density-ratio candidates per coordinate | 24 |
| TPE random-startup trials after importing seeds | 0 |

Any strictly improved feasible score moves the center. The relative threshold
only governs the expansion success counter. Non-improving, infeasible, and
cached non-improving proposals count toward contraction. The candidate box is
intersected with the global `[-4,4]` bounds; no observation is clipped to make it
fit. If contraction would take the region below its minimum, the initial size
is restored around the current feasible incumbent, preserving the observation
archive. This reset consumes no extra evaluations and performs no reconstruction.

Optuna operates in normalized local coordinates `[0,1]^256`. When the center or
length changes, the local study is rebuilt from actual archived points inside
the new box, using their true scores and constraints. Outside points remain in
the durable archive for future boxes. All 100 warm starts are archived, but only
those inside a particular local box enter its density estimates. At least the
feasible incumbent is present. The sampler uses these imported observations
immediately, even when a contracted box contains fewer than ten points.

The 24 acquisition candidates are internal TPE density-ratio calculations, not
24 decoded peptides, oracle evaluations, or a Thompson-sampling batch.

## Recovery and evidence

Each task stores the region center, length, success/failure counters, revision,
reset count, sampler state, RNG state, all actual latent coordinates, and a
single pending event in `runtime.sqlite`. The decoded proposal and exact oracle
request (including nonce) are committed before scoring. A lost oracle response
can reuse the existing service's durable receipt; a saved scored event is
committed without another decode or oracle call. Measurement and region state
are committed together. A transport or invalid-score error stops the task and
retains its evidence rather than skipping it.

`history.json` records scores, feasibility, cache accounting, region actions, and
the number of local training observations used for each proposal. Completed
archives include the SQLite ledger, history, progress, and checksummed manifest.
Existing archive/accounting verification is reused.

## Commands

Use the existing peptide environment and one existing frozen-VAE inference
endpoint. Preparation performs no inference or evaluation:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/data/cys/envs/bolt_clean_venv/bin/python \
  /root/data/cys/llmoptimizer_benchmark/tools/peptide_tpe_tr.py prepare \
  --campaign /root/data/logs/peptide_tpe_tr_test200_20261107_v1
```

Run one task after choosing an available CPU and healthy existing endpoint:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/data/cys/envs/bolt_clean_venv/bin/python \
  /root/data/cys/llmoptimizer_benchmark/tools/peptide_tpe_tr.py run-task \
  --campaign /root/data/logs/peptide_tpe_tr_test200_20261107_v1 \
  --task pep_104 --cpu <available-cpu> --server http://127.0.0.1:<existing-port> --launch
```

For a task queue use `worker` with `--index i --workers N` instead of `run-task
--task ...`, retaining `--cpu`, `--server`, and `--launch`. Each coordinator runs
one task at a time; task locks prevent duplicate ownership. CPU allocation and
worker count must follow the available resources when launching the campaign.
`--stop-after-events N` permits a bounded component run; `--stop-after-stage
proposal` or `score` supports recovery checks. These controls do not change the
pinned production budgets.

Run the behavioral tests (real Optuna, synthetic objective, no real oracle calls):

```bash
cd /root/data/cys/llmoptimizer_benchmark
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/data/cys/envs/bolt_clean_venv/bin/python -m unittest discover \
  -s tests -p test_peptide_tpe_tr.py -v
```
