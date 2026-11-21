SemanticOPT query and peptide held-out evaluation
================================================

Current runtime amendment (2026-11-11): **peptide first, query second**, per
the user's updated priority. `runtime_policy.json` supersedes the original
stage order and `fallback: false` below. The original recipe and all prior
measurements retain their hashes. New measurements/status include the runtime
policy hash, and `identity.json` pins the amendment and updated code.

After five exhausted candidate pairs (ten LLM responses), a deterministic
random sampler proposes a distinct valid discrete candidate. Peptides use
bounded reference edits (including substitutions, insertions and deletions),
then the full alphabet/length/cysteine/edit-similarity checks. Queries use
canonical join trees over exactly the required table aliases and permitted
operators, favoring joins connected by predicates. This distribution is not
claimed to be uniform. All evaluated configurations and failed retry proposals
are excluded; a bounded search that cannot find a novel feasible candidate
still stops explicitly. Every fallback consumes one normal proposal/oracle
slot, is tagged `random_fallback`, and contributes to a separate fallback
count. No predicted mean/std is invented. Fallback outcomes enter subsequent
prompts as external measured observations, not fabricated assistant responses.
Durable selection/in-flight/measurement guards apply equally to fallback.

The existing query measurements are preserved for the queued second stage.
The following paragraphs describe the original stage order before this amendment.

Queue (revised 2026-11-11 by user request): query test200 runs alongside the
remaining ML ORPO+RL+FSBO120 tasks, followed by peptide test200. The explicit
start_alongside_ml authorization is in scheduling.json; the original dependency
record in recipe.json remains as provenance. The ML campaign continues running.
SemanticOPT excludes active ML task and inference CPU affinities, reserves ML
worker memory growth, and increases admissions automatically as ML releases
resources. Four SemanticOPT replicas use available GPU memory alongside ML.
Each stage loads its own verified, completed SemanticOPT SFT adapter, using
Qwen3.5-9B and four inference replicas (one on each local A800). Two of the
three 21 GiB replica slots on each GPU are reserved for the user's peptide/query
ORPO and RL models. SemanticOPT never expands into those eight reserved slots.
Dedicated ports 19360–19363 and broker 19990 isolate this deployment. A verified adapter
handoff releases the query replicas before loading peptide replicas.

Query: the existing fixed 200-task split, 49 canonical Bao observations plus
1,000 sequential proposals per task, 50-second PostgreSQL cap. Every worker
has a dedicated restored IMDb cluster, shared_buffers 1 GB, work_mem 64 MB,
and two PostgreSQL parallel workers. Databases live under this campaign in
/root/data/cys, with an 8 GB per-process temporary-file limit. No existing
database or active FSBO evaluator is appropriated. Database failures remain
errors, and timeouts are censored observations rather than exact runtimes.

Peptide: the existing fixed test200 split, the same 100 revalidated feasible
initial observations plus 5,000 sequential proposals per task. CPU inference
uses the existing eight-model APEX implementation and bacteria_0 MIC objective.
No VAE, GP, latent-space search or reconstruction stage is involved.

SemanticOPT generates two direct discrete candidates with mean/std predictions,
validates them and selects one with success-weighted expected improvement.
It evaluates exactly one candidate per step. Canonical query plans must preserve
the query tables and aliases. Peptides must satisfy the standard alphabet,
length <= 30, cysteine count <= 1, and reference similarity >= 0.75. Previously
evaluated configurations are rejected. Five candidate pairs are attempted;
exhaustion pauses the campaign instead of inserting fallback observations.
Prompt histories replay causal action/feedback turns from a recent 32-decision
window, preserve the incumbent and constraints, and shrink the window if needed
to fit the exact 15,360-token input limit. Peptide prompts retain 50 feasible
initial examples as used in SFT; all 100 imported evaluations remain in the
duplicate cache and incumbent accounting.

Both LLMOptimizer2 and LLMOptimizer4 admit the maximum number of CPU task workers
within cgroup CPU and memory limits after accounting for unrelated measured CPU
load. Nominal CPU ceilings: 34 local + 62 remote peptide workers; 11 local + 20
remote query workers at three CPUs per worker. Query memory requirements can
lower these counts. Six local CPU equivalents are reserved for the four
replicas, broker and coordination, and two remote CPU equivalents for overhead.
When unrelated CPU load rises, excess task workers yield after committing their
current evaluation and later resume from the same task checkpoint.
Admissions retain 12 GiB RAM and allow 12 GiB per new query worker or 2 GiB per
peptide worker. Each host records actual admissions in DOMAIN/HOST.status.json.
Completed query clusters are shut down before the peptide stage.

Recovery: per-task exclusive shared-filesystem locks, synchronous SQLite
transactions, per-sample inference RNG, and broker replayable requests. Every
measured result is committed before progress advances. A result already saved
is never re-scored. An interrupted, unresolved oracle call blocks automatic
replay and requires an audit. A task, service or controller failure pauses new
evaluations; peptide cannot launch until all 200 query budgets complete.

Checks: code/validate.py audits all 200 query initializations, all 200 peptide
initializations, adapter identities, candidate selection and interrupted resume.
It re-scores twenty existing peptide initializations solely to check CPU APEX
agreement against archived GPU results, within a recorded 0.1% relative
tolerance for kernel arithmetic differences. Each stage also requires a live one-proposal training-task smoke
check after its inference replicas load. No test task is used for this smoke.

Files: recipe.json freezes budgets, splits and model identities; identity.json
pins implementation and inputs. queue_status.json is live queue state;
queue_error.json and PAUSE indicate errors. DOMAIN/runs/TASK/observations.sqlite
contains initial observations, predictions, rejections, inference RNG and results.
DOMAIN/evaluation_complete.json exists only after all 200 tasks finish their
full budgets. Logs are in logs/ and service request journals in DOMAIN/services/.

Launch: nohup /usr/bin/python3 -u code/queue.py > logs/queue.log 2>&1 &

Resource revision: the last two ML worker allocations on LLMOptimizer4
(CPUs 0-5, six CPU slots) remain reserved for ORPO after ML finishes.
SemanticOPT excludes these cores in both query and peptide scheduling.


## Batch-50 continuation (2026-11-11)

The user requested one JSON batch of 50 direct configurations per model call.
The batch protocol is pinned separately in `batch_policy.json`; the base recipe,
initial observations, adapters and per-task evaluation budgets stay identifiable.
This is a batched-feedback continuation rather than the original two-sample,
weighted-EI sequential protocol. Predictions are not fabricated for batch items.

Round counts use only committed proposal evaluations (not the 100 peptide or
49 query initial observations): `rounds = n // 50`, `offset = n % 50`. A task
with 79 committed proposals first requests 21 to reach 100, then requests 50.
Peptide budgets remain 5,000 proposals (100 rounds) and query budgets 1,000
(20 rounds). The final request cannot exceed the remaining evaluation budget.

Each batch is saved in the task SQLite database before any oracle call. Each
oracle result commits separately. A yielded/restarted task consumes the saved
queue at its committed cursor, without regenerating or rescoring completed
items. A call marked in flight without a committed result requires audit.
Every item passes the existing domain validator and full-history novelty check,
including duplicates within its own batch. Invalid/duplicate items consume no
oracle budget; bounded top-up requests run before the previously authorized
constrained fallback. LLM and random fallback provenance remain separate.

Historical single-proposal measurements are preserved. New records include the
batch policy hash and zero-based proposal round/offset. Status files expose
completed rounds and current offset. The migration snapshots and prefix hash
audit are under `changes/batch50_*`.

The transport response is one numbered JSON object (keys `1` through the
requested count), with peptide sequence strings or query integer arrays as
values. This compact format is converted losslessly into the existing
`sequence`/`plan` configuration objects. No sequence edits or plan repairs occur.

Batch-50 is not guaranteed to provide 50 distinct valid LLM proposals: adapters
may emit duplicates, short responses or malformed JSON. The runner accepts
only valid novel entries, requests top-ups at most four times in total, and
uses the authorized constrained fallback for any remaining slots. Compare
LLM and fallback counts when interpreting speed and optimization quality.


## Peptide CPU batch scoring (2026-11-11)

APEX now scores up to 50 remaining proposals from a saved batch in one
vectorized CPU call per task worker. Each worker retains its one-core affinity,
with PyTorch intra-op and inter-op threads set to one. It does not create 50
processes. The eight-model ensemble and bacteria_0 MIC objective are unchanged.

`peptide_oracle_policy.json` records this execution amendment separately from
the proposal protocol. The benchmark on one core took about 1.82 seconds for
50 peptides together versus 6.13 seconds individually (about 3.4x for scoring).
Mixed-length inputs also agree within rtol=1e-5, atol=1e-4; batching can cause
small floating-point differences. Overall speed remains limited by LLM time.

A durable oracle journal marks the entire batch in flight before the call.
All results and the committed flag are inserted in one SQLite transaction,
so counts advance by the batch length together. An unresolved in-flight batch
requires audit before replay. Existing single-item pending proposals finish
individually before the batch path is used. Partial rounds and saved unscored
proposals resume without dropping observations or regenerating proposals.
Validation, benchmark, migration snapshots and recovery tests are under
`changes/peptide_cpu_batch50/`.
