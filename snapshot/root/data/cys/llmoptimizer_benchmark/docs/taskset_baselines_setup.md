# TaskSet CASH Baselines

Use `/root/data/cys/llmoptimizer_benchmark`. The three supported baselines are
Random Search, Optuna TPE, and **LOLBO (no-VAE HPO adaptation)**. No offline
training or historical source-task observations are needed. TPE and LOLBO update
from the current task's evaluations.

The evaluation defaults match `taskset_corpus_train_1000s_100t/_queue/env.sh`:
1,000 training steps per configuration, validation every 1,000 steps, one
replica, and `min_normalized_valid_outer`. The runner defaults to 100 evaluated
configurations per task. Explicit `--smoke` remains a shortened 400-step check.

## LOLBO Implementation

`TaskSetLOLBOSampler` subclasses the ML `LOLBOTurboSampler`. It uses the same
deep-kernel GP, trust-region updates, Thompson sampling, and surrogate fitting.
This is the adapted ML baseline, not a complete reproduction of latent/VAE LOLBO.

CASH requires additional encoding: all family parameters occupy fixed vector
coordinates, the family choice is one-hot, and inactive coordinates have fixed
placeholders. A family switch samples newly active coordinates across their
domains. Candidates are canonicalized **before GP scoring**, and only the chosen
family's active parameters reach ConfigSpace validation and the evaluator.

Production defaults match the ML backend: 10 random initial evaluations, proposal
batches of 10, learning rate 0.01, 20 initial GP epochs, 2 subsequent GP epochs,
and the backend's 2,000-5,000 candidate pool. TaskSet runs the surrogate on CPU.
`--n-trials 100` includes initialization and failed evaluations. A final partial
proposal batch does not cause extra evaluator calls.

The previous TaskSet `lolbo` route selected CMA-ES. Those histories must not be
reported as LOLBO results or used to resume the corrected method.

## Environment

The isolated environment prepared for these checks is:

```text
/root/data/cys/baseline/envs/bl_taskset_lolbo/bin/python
```

It inherits the existing `hpo_benchmark` packages and installs the three pinned
GP dependencies in `requirements-taskset-lolbo.txt`. The original environment
and running jobs are not modified. The real oracle continues to use
`/root/data/cys/conda_envs/taskset_eval/bin/python` (TensorFlow 1.15, CPU).

To recreate the overlay:

```bash
/root/data/cys/conda_envs/hpo_benchmark/bin/python -m venv --system-site-packages /root/data/cys/baseline/envs/bl_taskset_lolbo
/root/data/cys/baseline/envs/bl_taskset_lolbo/bin/python -m pip install -r requirements-taskset-lolbo.txt
```

## Verification

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/data/cys/baseline/envs/bl_taskset_lolbo/bin/python -m unittest \
  tests.domains.test_taskset_lolbo tests.domains.test_taskset_cash -v
```

This checks all three methods with 100 dummy evaluator calls each, all five
conditional families, actual GP proposals, and uninterrupted-versus-resumed
configuration sequences. It also covers an interruption during a Random Search
evaluation and the unchanged flat ML sampler's GP path.

The real diagnostic runs six configurations per method on `mlp_family_seed12`
(an original TaskSet training-split task), with 1,000 inner steps, validation every
1,000 steps, and one replica. It stops after three evaluations, resumes in a new
process, and verifies ConfigSpace validity, actual oracle results, saved history,
and multiple LOLBO GP updates. Its two-point initialization, two-candidate
proposal batches and 256-point candidate pool are diagnostic overrides.

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  taskset -c 40-43 nice -n 10 \
  /root/data/cys/baseline/envs/bl_taskset_lolbo/bin/python \
  examples/benchmark/domains/check_taskset_bo.py --output-dir /path/to/fresh/diagnostic
```

The diagnostic writes `report.json`, per-process logs, and separate per-method
histories and oracle artifacts. It uses the minimum normalized validation loss;
these short runs are component checks, not comparative benchmark results.

The diagnostic copies the unchanged TaskSet source into its output directory,
links the original task definitions, and uses a private optimizer registry via
`TASKSET_RUNTIME_ROOT`. This avoids scanning the shared registry's roughly
95,000 generated JSON files for every evaluation and avoids registering dry-run
configurations in that shared directory. The TensorFlow interpreter and training
code are the same as the normal oracle.

## Resume And Campaigns

Random Search seeds ConfigSpace with the requested seed and advances the RNG
past completed and pending configurations on resume. TPE saves its Optuna study.
LOLBO saves the study, GP, trust region, pending proposal batch and Torch RNG.
LOLBO requires a matching checkpoint and rejects changed seed/sampler settings.

Use a distinct output root for each evaluation protocol. The runner places
oracle artifacts under each task/method/seed directory and rejects histories
with a different evaluation protocol or metric before resuming or skipping them.
The campaign evaluation protocol is set to the BO corpus settings above. The
selected held-out list is the `test` array in the major-family split below.

## Selected TaskSet Split

The agreed TaskSet split is **838 training tasks and 210 test tasks**, using the
12 task families with at least as many scoreable tasks as `mlp_vae_family` (55).
The 109 tasks in smaller families are excluded from both sides, regardless of
whether their BO histories exist. Existing train/test assignments are preserved.

Canonical manifest:

```text
/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/taskset_major_families_split_80_20_v1.json
```

The manifest includes all train, test and excluded task IDs, the 26-family
inventory, and hashes of the source split and task universe. It is separate from
the older 686/172 and 925/232 manifests. Existing scripts referring to those older
files do not automatically select the new split.

The adjacent `.audit.json` records training-history paths, hashes and coverage.
At creation, 827 of the 838 training tasks have complete 100-trial histories in
`/root/data/logs/taskset_corpus_train_1000s_100t`: 82,700 observations in total.
Eight word-RNN language-model tasks and three NVP tasks lack histories. They
remain training tasks; their IDs are listed in `missing_train_task_ids` in the
audit. This is a coverage audit, not a ConfigSpace or model-training preflight.

Reproduce the manifest and audit, or inspect current inputs without writing:

```bash
/root/data/cys/baseline/envs/bl_taskset_lolbo/bin/python \
  examples/benchmark/domains/make_taskset_major_families_split.py --dry-run
```

Without `--dry-run`, the script creates the artifacts and refuses to replace
different existing content. FSBO, RGPE and OptFormer preparation must select
offline sources only from this manifest's `train` array. Test and excluded
histories cannot be source data; observations collected during the current
target run remain allowed. Missing-history collection, model training, and
campaign launches are not started by the split builder.

## Verified On 2026-09-29

These completed real checks used the earlier 100-step / evaluate-every-50
diagnostic settings. Their reports remain unchanged. The 1,000-step protocol is
now configured for future evaluations; it has not been launched in this update.

- All 11 targeted tests passed, including 100 dummy evaluator calls per method.
- Random Search, TPE and LOLBO each passed all six real evaluations and resumed
  after three evaluations in a fresh process. All proposed configurations passed
  ConfigSpace validation.
- LOLBO made four GP proposals and completed two surrogate fits, including an
  update after restarting. Its encoding retained all 34 parameters in 43 vector
  coordinates.
- The final real diagnostic took 220 seconds on CPUs 40-43 with GPUs disabled.
- Dependency validation passed. The isolated TaskSet source matched the original
  byte-for-byte, excluding configuration directories and Python caches.

Report: `benchmark_data/taskset_baselines_dryrun_20260929_isolated/report.json`.
An earlier diagnostic was stopped after its first successful evaluation because
of shared-registry startup overhead; its directory contains `aborted.json`.

The broader `tests.domains.test_domain_transfer` module could not load because
of a pre-existing `from __future__` syntax error at line 152. That unrelated
test file was not modified. No full TaskSet campaign was launched.
