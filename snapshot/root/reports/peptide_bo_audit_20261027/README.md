# Peptide BO Fix Verification

## Current V5 Campaign

The user authorized launch after the initialization and recenter fixes.
The fresh campaign is `/root/data/logs/peptide_bo_train800_20261027_v5b`.
Its immutable recipe pins the initialization, split, models, code and environment.
The frozen release is `/root/data/logs/peptide_bo_release_20261027_v5`.

- 800 training tasks; the original 200 test references are retained and not launched.
- 100 strict measured initialization entries per task. Original compatible mutants
  are retained in file order; shortages and three replacement pools required 774
  new measured scores. Preparation of v5b reused those scores and made zero calls.
- Acquisition batches of 50, at most 5,000 acquisition proposals, and at most
  5,000 fresh post-initialization APEX calls shared with recentering. Recenter calls
  can exhaust the fresh budget before the 100th acquisition round.
- Joint VAE/GP updates after ten consecutive non-improving acquisition rounds.
  All observations are re-encoded after VAE changes. Native decode-and-score
  recentering is bounded before scoring and preserves Adam state on recovery.
- Cached batches update surrogate data and progress/trust-region counters.
  A durably scored pending batch resumes without another oracle call.
- Completed tasks are checksummed and packed into one archive each; unfinished
  tasks retain a bounded set of checkpoint and progress files.

The complete preflight passed all ten required gates. Seven focused tests passed.
The live extended test completed 36 acquisition rounds, 1,800 acquisition
proposals, 2,127 fresh calls (630 from recentering), and 422 feasible new
observations. Trust-region length contracted from 0.8 to 0.4. Its best MIC remained
400.143768; this verification establishes working mechanics, not an improvement.
The joint update changed 183 VAE tensors. Fresh-interpreter recovery and archive
verification also passed.

An actual LLMOptimizer4 CPU test completed two 50-proposal acquisition rounds and
one joint VAE update with remote GPU inference: 141 fresh calls, including 49
recenter calls. Its VAE also changed 183 tensors. No CPU gradient was detached.

Validation reports:

- `/root/data/logs/peptide_bo_validation_20261027_v5b/report.json`
- `/root/data/logs/peptide_bo_validation_20261027_v5b_remote4/report.json`

Worker inventory and logs are under the campaign's `workers/` directory.
The initial 31-worker launch was expanded to 82 managers: 34 on LLMOptimizer2
and 48 on LLMOptimizer4, each with a distinct physical CPU assignment.
LLMOptimizer2 reserves 34 worker cores, three service cores, and three BOLT
training cores, reaching its 40-CPU quota. LLMOptimizer4 uses its 48 physical
cores; sibling SMT threads are not assigned to another worker. LLMOptimizer3
is not used.

The user discontinued QueryOpt random search. Its active managers and private
PostgreSQL instances were stopped, preserving results, journals and checkpoints.
Fourteen reproducible tmpfs database clones were removed after verifying the
restore archive checksum; older clones with different provenance were retained.
Audits are in
`/root/data/cys/logs/query_hosts24_20261026/stopped_for_peptide_20261027/`.

LLMOptimizer2 workers use GPUs 0, 2 and 3. LLMOptimizer4 uses VAE/APEX services
on ports 18871, 18872 and 18873 over existing loopback SSH tunnels. Joint VAE
training remains on each remote worker's CPU; task model weights remain isolated.

The expansion exposed five remote decoding errors: eight cached task weight
banks were insufficient for the enlarged pool, and a bank could be evicted
again after the client's one reload. All remote workers were checkpointed.
The operational service now retains up to 64 task weight banks and removes
obsolete versions belonging to the same task. The frozen BO code and pinned
recipe are unchanged. The 48-bank reproduction failed on the earlier service
and passed on the revised service, including real encoding, decoding, scoring,
reply reuse and obsolete-version removal. All three revised services also
passed remote encode/decode/APEX checks. Remote workers resume from checkpoints;
their log offsets distinguish old errors from errors after recovery.

Operational scripts and verification artifacts are in
`/root/data/logs/peptide_bo_launch_20261027_v5/`.

### BOLT GPU Memory Handoff

BOLT TaskSet full SFT remains on GPU 1 and CPUs 21-23. A detached handoff waits
for its complete verified checkpoint 150, then resumes with GPU-resident model
parameters, a 72 GiB CUDA allocator allowance, and the same CPU Adam optimizer.
The effective batch, dataset, training extent, optimizer and RNG state are
preserved. A launch or pre-validation failure attempts CPU-parameter-offload
recovery from the latest verified checkpoint. DeepSpeed schema checks passed;
GPU-resident execution and its speedup still require live confirmation.
Handoff state and logs are in
`/root/data/cys/baseline/artifacts/bolt_taskset_top10_full_sft/gpu_resident_restart_20261027/`.

## Earlier V4 Audit

The historical verification below preceded the new V5 preparation and launch.
Existing training and QueryOpt jobs were not stopped.

## Verified Current Path

Production recipe:
`/root/data/logs/peptide_bo_verified_20261003_v4/recipe.json`

Release:
`/root/data/logs/peptide_bo_release_20261003_v4`

The verification-only command returned `valid: true, launched: false`.
It checked all pinned code/assets and the exact worker package versions.
All thirteen CPU regression tests passed with GPUs hidden. The editable campaign,
verified BO module, and regression tests match the frozen release byte-for-byte.

## Previous Bugs

| Issue | Status in the verified v4 path | Evidence |
|---|---|---|
| Every task uses reference zero or its initialization pool | Fixed | Per-task reference and aligned sequence/score pool; `create_state` checks against `REFERENCE_SEQUENCE[index]` |
| Infeasible high-scoring candidate becomes the search center/incumbent | Fixed | Center and reported incumbent use feasible observations; all observations remain available to the GPs |
| VAE recentering adds unbudgeted oracle calls | Fixed | Re-encode existing sequences, preserve scores, and assert unchanged oracle-call count |
| Fresh-process resume cannot import pickled LOLBO state | Fixed | Verified upstream import path is installed before checkpoint deserialization |
| Resume loses GP, VAE, trust-region, RNG, or score-cache state | Fixed | Full task/recipe-bound snapshot and restoration; previously verified fresh-interpreter scored-batch recovery |
| Cached batches skip GP feedback or failure/progress updates | Fixed | Cached scores still reach `update_next(..., acquisition=True)`; proposal/round/cache counters advance |
| Incomplete task silently restarts or overwrites saved data | Fixed | Missing, incompatible, and corrupt state fail explicitly; completed tasks use verified archives |

The October 3 live component checks remain applicable to this pinned release:
two real tasks, VAE updates, acquisition after updates, checkpoint restoration,
fresh-process pending-batch recovery without rescoring, and archive verification.
They were not rerun on GPUs during this check.

## Remaining Optimization Problems

The previously saved larger diagnostics still show zero feasible new proposals
out of 2,505 proposals across five tasks. Their feasible incumbents remained
initialization points. No newer fix or successful extended diagnostic was found
in the verified current path.

The upstream trust region still starts with length 0.8 and scale eight, giving
a per-coordinate radius of 3.2. Historical random-perturbation probes support
investigating that scale, but do not prove that a smaller region improves MIC.

The decoder still limits new sequences to at most one cysteine and replaces `X`
with `A`. Three references cannot satisfy the 75% similarity constraint under
those rules: `pep_049`, `pep_073`, and `pep_082`. They remain explicitly blocked.
Some original scored initialization sequences also use residue policies that
differ from the new decoder outputs; their scores have not been reinterpreted.

Production remains on hold in:
`/root/data/logs/peptide_bo_verified_20261003_v4/launch_hold.json`

The older generic runner still deletes/recreates shared initialization latent
cache files and does not provide the same state/accounting guarantees. The
verified fixes must not be attributed to every legacy launcher or historical run.

## Relevant Code

- `tools/peptide_bo_campaign.py`: recipe verification, import-before-load,
  scored pending batches, unconditional cached-batch feedback, accounting.
- `mindware/domains/protein/verified_bo.py`: per-task initialization, feasible
  center, score cache, encode-only VAE updates, complete state restoration.
- `tests/domains/test_verified_peptide_bo.py`: thirteen regression tests.

Earlier detailed diagnostic report:
`/root/reports/peptide_bo_audit_20261003/README.md`

## Authors' Original Implementation

Both checked author checkouts are at commit
`a164b7942fcbbe29eca9c0a99c939f0c7bf70995`. The relevant peptide decoder,
trust-region, reference, and constraint files in the working copy are unchanged
from that commit.

- The decoder does not enforce the 75% reference similarity. It replaces `X` with
  `A`, removes gaps, replaces extra cysteines until at most one remains, and uses
  `AAA` for an empty decoded string. There is no special repair for the three
  conflicting references.
- A constraint GP and constrained Thompson sampling model the reference-similarity
  restriction. The trust region halves after 32 consecutive unsuccessful batches.
  The previous extended tests ended after eleven batches, before their first
  contraction; zero feasible proposals in those tests does not prove failure over
  the paper's full optimization budget.
- `run_lolbo` triggers VAE updates and `recenter()` after ten consecutive
  non-improving acquisitions. This differs from v4's periodic every-ten-round
  schedule. Recenter can decode and score new peptides; the authors' `num_calls`
  loop includes those evaluations.
- The original acquisition uses the highest objective from all observed points
  as its center, even if that observation violates the constraint. The feasible-
  center fix in v4 should be retained to avoid reintroducing this inherited issue.
- Known feasible initialization points remain valid incumbents even when the
  decoder cannot generate a feasible new point. A useful initialization therefore
  does not demonstrate that the BO decoder overcame the residue-policy conflict.

The paper specifies 1,000 initialization peptides, acquisition batches of fifty,
and a 20,000-oracle-call optimization budget. The repository's example shell uses
a different 200,000-call threshold. Its total-call loop can also overshoot the
threshold during the final acquisition/recenter cycle; actual calls must be
reported explicitly.

The existing peptide SFT v3 manifests identify the family-disjoint 800/200 split:
`/root/blockdata/llmoptimizer_benchmark/docs/peptide_split_800_200.json`, SHA-256
`25cd85c12a30d46e5312e09cb5c4770418c9e04096f20d6dd5ae08a17a649138`.
All three conflicting references belong to that training split.

The earlier clarification questions were resolved. See the V5 campaign above
for the accepted initialization, replacement and recenter policies and launch.
