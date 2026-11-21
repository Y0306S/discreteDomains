# Peptide BO audit and prepared rerun

Date: 2026-10-03. Status: **production on hold after extended diagnostics**.
No production peptide campaign has been launched by this work.
The short component test passed, but five larger tests found no feasible new peptides.
The restart fix and updated component checks are in the versioned `_v4` release.

## Findings

### 1. Completed campaigns do not have comparable evaluation budgets

The latest campaign advertises 100 initialization observations plus 5,000 acquisition
evaluations. Its 1,000 index records report **6,966 to 42,008 actual post-initialization
oracle evaluations** (median 12,650.5). Native `LOLBOState.recenter()` decodes peptides
after VAE updates and scores the resulting strings, adding unbudgeted observations.
The acquisition counter alone passes `budget_respected`; it is not a total-budget check.

Deduplicate each `index.jsonl` by `task_id`, keeping its last record:

| Campaign under `domain_protein` | Tasks | Early candidate exhaustion | Median observations | Median post-init oracle evaluations |
| --- | ---: | ---: | ---: | ---: |
| `stbo_bsz10_r100_fixed` | 1,000 | 69 | 8,175 | 1,203 |
| `stbo_init100_r100_bsz50_e2e10` | 1,000 | 290 | 33,015 | 11,738.5 |
| `stbo_init100_r100_bsz50_e2e10_v2_instrumented` | 1,000 | 0 | 35,683 | 12,650.5 |

Latest campaign: all acquisition counters are 5,000, but extra E2E evaluations range
from 1,966 to 37,008 (median 7,650.5). Actual rounds range from 101 to 542 (median 171),
not the advertised 100. Cache hits make acquisition rounds and fresh evaluations differ.

These histories can contain useful scored peptides, but must not be presented as
equal-budget 5,100-evaluation results. Earlier `bo80_bolt` and `bo80_bolt_bsz10_r100`
campaigns had the already documented reference-0 / reference-0-initialization defects;
their previous void status remains appropriate. Originals were not changed or deleted.

Evidence root: `/root/data/cys/blockdata_archive/domain_protein`.
Index SHA-256 values in table order:

```text
1af0c4948f954054e6e8ebeb9a3549e27a77cbbc64da4199ebc4bb012b97de6b
f90651c877c7bbb70a28d06fd6579bcaf781a0ed66c79bb9760ce1ffce5bd005
95aa12e657c87d3498d28c3d348a79de118fc80303b91c57431dc3886b7d8d86
```

### 2. Three reference tasks cannot have feasible decoded proposals

BOLT's peptide decoder replaces extra cysteines until at most one remains, and replaces
every `X` with `A`. These rules conflict with three references' 75% Levenshtein similarity
requirement. Deleting a residue still costs one edit, so allowing variable lengths does
not remove the contradiction.

| Task | Reference | Mandatory edits under decoder rules | Maximum allowed edits |
| --- | --- | ---: | ---: |
| `pep_049` | `CYCRRRFCVCV` | 3 | 2 |
| `pep_073` | `GICYCICGRGICRCICGR` | 5 | 4 |
| `pep_082` | `GXXNNXXXXXXNXXNXXGY` | 12 | 4 |

All three latest histories have exactly **100 feasible observations**, all from
initialization, despite respectively 10,376, 11,004, and 7,748 post-init oracle evaluations.
They did not find a single feasible new peptide. The new recipe records them as blocked;
workers skip them visibly without marking them complete. The original train900/test100
membership stays fixed: 897 train tasks and all 100 test tasks are runnable.

Unresolved scientific choices: allowing multiple cysteines could unblock the first two.
The unknown-residue reference needs an explicit interpretation or corrected reference;
silently substituting residues would change the task. Neither change is made here.

### 3. The search center can be infeasible

Upstream `lolbo/utils/bo_utils/turbo.py:177` selects `X[Y.argmax()]` even for constrained
Thompson sampling. An infeasible peptide with a better objective can pull the search away
from the permitted neighborhood. The wrapper supplies feasible observations as center
candidates while leaving all observations in the objective and constraint GPs.

Only the best **feasible** MIC is reported. For example, latest `pep_900` records an
unconstrained objective of -4.3004 but best feasible objective -9.4154. Since score is
negative predicted MIC, the valid incumbent is MIC 9.4154, not 4.3004.

### 4. Initialization and checkpoint metadata were insufficiently bound to tasks

The legacy runner disables a global latent cache by deleting/recreating the same shared
file; workers can still race. It also restores observations and VAE weights while
reinitializing GP/trust-region state, does not restore RNG state, and can proceed with a
missing finetuned VAE. Its `start_seq` field comes from the reference-0 mutation file,
not necessarily the actual task reference. Old checkpoints cannot safely become new runs.

The new runner encodes per task without shared latent writes. It pins code, model files,
VAE vocabulary, initial pool, selected rows, dependency versions and recipe; checkpoints
bind to that recipe plus task ID. GP, constraint GP, VAE, trust region, RNGs, score cache,
observations and counters are restored together. Upstream creates a fresh Adam optimizer
for every training invocation, so there is no persistent optimizer state between rounds.
Missing/corrupt/incompatible checkpoints fail visibly; they never silently start over.

### 5. Initialization and generation have different residue policies upstream

Of the selected original 100,000 initialization rows, 2,702 have multiple cysteines.
Another 288 rows across `pep_014`, `pep_024`, and `pep_082` contain `X`. The VAE vocabulary
includes `X`; APEX's original encoder represents unknown letters as zero-valued positions.
Those initialization scores are preserved, not silently reinterpreted.

The prepared recipe keeps the original first 100 aligned sequence/score pairs per task,
as the upstream initialization path does. They are all finite, at most 30 residues long,
and satisfy the reference similarity constraint. New decoder outputs must pass the
standard-amino-acid / length / cysteine checks before scoring.

Enforcing the decoder's residue restrictions at initialization is a separate protocol
change. Filtering alone cannot supply 100 points for every reference (`pep_001` has only
53 suitable points in its 1,000-row pool), and cannot solve the three impossible tasks.
The optional `first_valid` preparation mode deliberately fails on insufficient pools.

### 6. Fresh-process checkpoint reload failed

The extended `pep_003` test stopped after round 10 / 500 proposals / one VAE update.
Restarting its CLI in a new interpreter failed at `torch.load` with
`ModuleNotFoundError: No module named 'lolbo'`. Checkpoints pickle upstream `TurboState`,
but the upstream import path was previously added only while constructing the models,
after deserialization. The earlier in-process component test had already imported LOLBO
and therefore missed this defect.

Fixed in `tools/peptide_bo_campaign.py`: add the verified upstream directory to the
import path before loading the checkpoint. The updated GPU component checker resumes
a real scored pending batch in a fresh interpreter, verifies the observation prefix,
and fails if the saved batch is rescored. Production preflight now requires this check.

The existing extended diagnostic was completed with a scoped `PYTHONPATH` workaround
so its original pinned recipe and 500 observations could be preserved. This workaround
is recorded separately from verification of the fixed `_v4` code. The resumed run
finished its final one-proposal batch, and reopening its archive made no changes.

### 7. Completed pipeline checks do not establish useful optimization

All five extended tests used the proposed production batch size and update interval:
100 cached starts, 501 proposals, batch 50, a joint GP/VAE update at round 10,
and a one-proposal final round. All five completed, with correct accounting and archives,
but **zero of 2,505 proposals met the similarity constraint**. Their feasible incumbents
therefore remained initialization points.

| Task | Fresh oracle evaluations | Cache hits | Feasible new proposals | Best feasible MIC, unchanged |
| --- | ---: | ---: | ---: | ---: |
| `pep_003` | 457 | 44 | 0 / 501 | 400.14376831 |
| `pep_001` | 468 | 33 | 0 / 501 | 475.76486206 |
| `pep_014` | 427 | 74 | 0 / 501 | 167.49795532 |
| `pep_018` | 479 | 22 | 0 / 501 | 178.54765320 |
| `pep_050` | 421 | 80 | 0 / 501 | 47.19991684 |

These are five diagnostic tasks, including ordinary controls and residue-policy edge
cases, not a population-wide convergence estimate. `extended_report.json`'s `passed`
field refers to pipeline assertions, not peptide feasibility or objective improvement.

A separate zero-oracle geometry probe used the verified author VAE and fixed random
sparse latent perturbations. Raw initialization reconstruction was 96-100% exact for
the four tasks without `X`, but only 11% exact for `pep_014`, whose reference includes `X`.
Thus this is not evidence of a generally wrong VAE checkpoint/vocabulary pair; unknown
residues still have a representation problem.

Upstream TuRBO's initial length 0.8 and scale factor 8 give a per-coordinate radius 3.2.
Its 32-failure shrink threshold is beyond these 11-round tests. The probe found:

| Task | Feasible / 50 at radius 3.2 | Feasible / 50 at radius 0.4 |
| --- | ---: | ---: |
| `pep_003` | 0 | 38 |
| `pep_001` | 0 | 40 |
| `pep_014` | 0 | 44 |
| `pep_018` | 0 | 0 |
| `pep_050` | 0 | 34 |

This supports investigating latent search scale; these random perturbations do not
include GP acquisition and do not prove that a smaller radius improves MIC. Very small
radii mostly reconstruct duplicates. `pep_018` also produced 0/50 feasible decodes at
zero perturbation after the one-cysteine postprocessing: its current center is unsuitable
under the decoder policy, despite being feasible under the recorded similarity constraint.

Next diagnostic: compare the unchanged schedule through its first trust-region shrink
against a separately labeled smaller-radius run on training tasks, recording feasibility,
distinct proposals and best feasible MIC. Resolve the cysteine/unknown-residue policy
before changing initialization or the center rule. Production settings remain unchanged.

Evidence:
- `/root/data/logs/peptide_bo_extended_20261003_v2/extended_report.json`
- `/root/data/logs/peptide_bo_extended_20261003_v2/geometry_report.json`
- `/root/data/logs/peptide_bo_extended_20261003_v2/pep_003.finish.retry1.log`
- `/root/data/logs/peptide_bo_extended_20261003_v2/driver.resume_workaround.log`

## Prepared Protocol

Production recipe: `/root/data/logs/peptide_bo_verified_20261003_v4/recipe.json`.
Release source: `/root/data/logs/peptide_bo_release_20261003_v4`.
Editable source: `/root/blockdata/llmoptimizer_benchmark`.
The similarly named repo under `/root/data/cys/llmoptimizer_benchmark` is a divergent copy.

- Fixed split: `pep_000..pep_899` training, `pep_900..pep_999` test; blocked tasks remain listed.
- Provisional latest-campaign budget: 100 cached initialization observations + 5,000
  proposed candidates, in 100 batches of 50. Duplicates count as proposals, with cache
  hits and fresh APEX evaluations logged separately. Initialization requires no fresh
  scoring. This is **not** a promise of 5,100 fresh oracle executions.
- Objective: maximize negative APEX bacteria-0 prediction; report feasible MIC, lower better.
- Actual BOLT/LOLBO deep-kernel variational GP and constraint GP (GPyTorch), TuRBO and
  constrained Thompson sampling using upstream BoTorch components; 256-dimensional VAE.
- Author VAE verified against the original APEXGo file by SHA-256. The unsuffixed local
  retrained checkpoint is not selected. Both author-copy hashes are pinned in the recipe.
- GP initial training: 20 epochs; later updates: 2 epochs; learning rate 0.001.
- Joint GP/VAE updates every 10 rounds. Re-encode the same sequences and retain their
  existing outcomes; refit the GP after coordinates change. No recenter oracle calls.
- Two-deep checkpoints at round boundaries; a scored pending batch is durable before
  commit. Completed runs become one verified `.tar.gz` containing history, final
  checkpoint, progress and checksums. No per-evaluation file tree.
- One worker per free GPU, four selected CPU cores plus their SMT siblings reserved
  cooperatively. Require no current GPU compute process, at least 12,000 MiB free GPU
  memory, 16 GiB free system RAM, no overlapping narrow-affinity process, and low CPU
  utilization at dispatch. Existing jobs are not stopped.

This is a benchmark adaptation, not an unchanged paper reproduction. The paper describes
1,000 initialization points and 20,000 inner-loop oracle calls; an upstream shell script
also has a different, larger budget. Choosing the larger protocol remains open. The
feasible-center correction, explicit proposal budget and encode-only E2E step must be
reported with results. No production run has started with these provisional settings.

## Original Short Verification

Passed on LLMOptimizer3, NVIDIA A16 GPU 0, CPUs 16-19. Total component-check elapsed
time: 46.91 seconds; peak GPU allocated memory 1,362 MiB, reserved 1,534 MiB.
The first dispatch correctly refused a briefly busy SMT sibling before GPU computation.

- 11 regression tests passed: counters, duplicates, finite scores, final partial batches,
  feasible centers, impossible references, task/score alignment, model/RNG/trust-region
  restore, checkpoint publication interruption, archive cleanup recovery, preflight identity.
- Real `pep_000`: 100 cached initialization observations + 11 proposals, 10 fresh oracle
  evaluations, 1 cache hit, 3 rounds and 1 joint VAE/GP update.
- Real `pep_050`: 100 cached initialization observations + 11 proposals, 9 fresh oracle
  evaluations, 2 cache hits, 3 rounds and 1 joint VAE/GP update.
- 183 VAE tensors changed. E2E re-encoding preserved observations and scores and added
  no oracle evaluations. Acquisition also ran after the update.
- Restored GP/constraint-GP/VAE tensors, GP predictions, trust-region fields and Python,
  NumPy, CPU/CUDA Torch RNG state. A real scored pending batch was resumed with the oracle
  function replaced by an error, proving its saved outcomes were used without rescoring.
- Final batch size was 3, not 4. Both archives passed content checksums and completion
  accounting; re-opening a completed task skipped model construction and evaluation.
- Six additional diagnostic oracle predictions matched cached initialization scores to
  `rtol=1e-4, atol=1e-3` (max absolute differences 0.00156 and 0.00647 for the two tasks).

Machine-readable report:
`/root/data/logs/peptide_bo_components_20261003_v2/component_report.json`.
Log: `/root/data/logs/peptide_bo_components_20261003_v2/component.retry2.log`.
The now-superseded production preflight marker bound the passing report to the exact production
recipe SHA-256, `46e7d3a23761268ddbd60a553db1d97dac099bc1b66273d80dc87d4099d3fad6`.

This verifies the components, not convergence or runtime for an entire 100-round run.
The diagnostic's batch size 4 and E2E interval 2 do not change production's batch size
50 and interval 10. Production memory/throughput will depend on history length and top-k.

## Final Restart-Fix Verification

The `_v4` release passed the updated real-GPU component check on LLMOptimizer3 GPU 5,
CPUs 16-19, in 56.63 seconds. All 13 unit tests passed, including the explicit launch
hold and rejection of a preflight without fresh-process resume evidence.

- Two real tasks each completed 100 cached initialization rows + 11 proposals,
  three rounds, one joint VAE/GP update, and a three-proposal final batch.
- `pep_000`: 10 fresh oracle evaluations and one cache hit; `pep_050`: nine and two.
- 183 VAE tensors changed; re-encoding added no oracle evaluations.
- Resume used a new interpreter with its working directory set to the release root,
  outside the upstream tree. No LOLBO module was preloaded. It restored the real
  checkpoint and committed its scored pending batch with rescoring disabled.
- The observation prefix, GP/constraint-GP/VAE parameters, predictions, trust-region
  fields, RNG restoration and completed archives passed the assertions.
- Six additional oracle predictions matched cached initialization outcomes at the
  specified tolerances. These are diagnostic calls outside the two BO budgets.

Report: `/root/data/logs/peptide_bo_components_20261003_v4/component_report.json`.
Log: `/root/data/logs/peptide_bo_components_20261003_v4/component.log`.
Production recipe SHA-256:
`74ae9850191cd33e3a557419912195e43930dd7f172d4af1276be7bf8c56e2d1`.

The final entrypoint's verification-only mode returned `valid:true, launched:false`.
A direct call to its preflight guard confirmed that the scientific launch hold remains
effective. No production worker was launched, and existing jobs were not stopped.

## Launch Hold

Prepared entrypoint:
`/root/data/logs/peptide_bo_verified_20261003_v4/launch_on_llmoptimizer3.sh`.
Without arguments it only verifies the prepared recipe; it starts no BO worker.

**Do not launch production yet.** `launch_hold.json` blocks the worker before resource
reservation or GPU computation. A passing component check does not clear this hold.
The `_v2` and `_v3` launchers forward to `_v4`, and `_v2`'s old preflight marker is preserved as
`preflight_passed.superseded.json` so direct use of that old campaign fails preflight.

After the findings above are resolved, use one worker initially. Future command only:

```bash
bash /root/data/logs/peptide_bo_verified_20261003_v4/launch_on_llmoptimizer3.sh --launch 5 16,17,18,19
```

The arguments are the physical GPU ID and CPU IDs. This is a foreground SSH worker;
use the machine's normal persistent session/job supervisor for a long campaign. It
checks resource availability before starting and between tasks. If resources are busy,
it exits visibly. No background watcher or automatic future launch has been installed.

Additional workers can use other free GPUs and nonoverlapping physical CPU cores; task
locks prevent duplicate runs. Conservative starting allocation is one worker, leaving
the existing GPU 1-4 jobs and CPU 4-11 evaluators alone. GPU/CPU occupancy must be checked
at launch time. The three blocked tasks remain visibly pending.

The earlier preparation directories are superseded by `_v4` and
have no production runs. Always use the explicit `_v4` entrypoint above. The legacy BO
launchers and their old checkpoints are not the rerun entrypoint.

## Operational Limits

The new archive reader is the campaign's `verify_archive`; old SFT exporters are not
automatically compatible with this new history schema. Adapt them before using new
histories for SFT. No new training job is part of this task.

SIGTERM/SIGINT requests stop after the current round and its checkpoint. A SIGKILL or
machine failure during oracle computation can occur before the result is durable; the
pending journal records the attempted batch, but exactly-once physical GPU execution is
not guaranteed across that failure. Corrupt latest checkpoints require explicit recovery.
Resource checks and cooperative locks cannot prevent an unrelated scheduler from later
placing another job on the same GPU; dispatch must remain coordinated across launchers.
