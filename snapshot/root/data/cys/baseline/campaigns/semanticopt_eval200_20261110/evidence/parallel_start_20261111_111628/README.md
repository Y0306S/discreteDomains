SemanticOPT query and peptide held-out evaluation
================================================

Queue: current ML ORPO+RL+FSBO120 completes, then query test200, then peptide
test200. The current campaign is never stopped or reconfigured by this queue.
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
