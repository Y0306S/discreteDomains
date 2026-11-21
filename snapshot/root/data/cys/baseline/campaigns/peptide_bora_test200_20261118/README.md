# BORA peptide test200 queued after SemanticOPT

Authorized on 18 November 2026: run BORA on the complete existing 200-task held-out peptide split after the active SemanticOPT peptide continuation finishes.

The campaign imports the exact 100 revalidated initial sequences and scores per task. It runs 4,000 distinct fresh oracle evaluations per task (800,000 total), excluding initialization. All 800 training tasks are disjoint from this test manifest. SemanticOPT observations beyond the common initialization are never supplied to BORA.

The LLM is untouched Qwen3.5-9B with no adapter. The original author peptide VAE is frozen, its posterior means provide 256-dimensional GP inputs, and there is no VAE retraining or FSBO meta-training. The minimizing objective is the same eight-model CPU APEX ensemble's bacteria_0 mean predicted MIC; the GP/assistant maximize its negative.

The copied official BORA policy, expected improvement (xi=0.01), random search plus ten L-BFGS-B starts, and three-draft/critique/merge assistant workflow are retained. The target GP is sklearn Matérn 2.5, normalized targets, alpha 1e-6, five hyperparameter optimization restarts. Policy uncertainty is monitored at 5,000 fixed locations. Native one/few-point feedback is retained; every 50 evaluations is a reporting checkpoint, not a forced batch. Thus 80 reporting checkpoints do not mean 80 native BORA decisions.

## Explicit peptide adaptations

- Shared 100-point initialization replaces the paper's usual five LLM-generated initial points.
- The original frozen VAE represents sequences. Per-coordinate latent bounds/scales come solely from that task's shared initialization and are fixed before optimization.
- Latent EI proposals are decoded and checked against exact sequence constraints. Up to 64 highly ranked random latent candidates supplement L-BFGS-B outputs. A 256-point pool of distinct feasible reference/incumbent edits (substitutions, insertions, deletions) supplements the decoded candidates. The union is canonically re-encoded and ranked with the same EI; random edits are not scored blindly as random-fallback evaluations. Proposal provenance and feasible decode counts are retained.
- Sequences use only the 20 canonical amino acids, length 1–30, at most one C, and similarity 1 minus Levenshtein distance divided by reference length at least 0.75. Duplicate, malformed, and infeasible proposals consume no oracle slots.
- The assistant handles sequence strings instead of numeric experiment coordinates. It retains released reflection and duplicate-retry behavior; selection uses exact offered-sequence membership. Invalid sequences are rejected, with BO used after an unusable intervention. No external model/API is called.
- Long histories use an explicit best-40/recent-40 plus all-history score statistics summary when needed. Original observations and all prompts remain saved. This replaces upstream numeric correlation summaries, which do not apply directly to amino-acid strings.

These are benchmark adaptations of BORA, not an assertion of an unchanged paper benchmark. Exact GP fitting with up to 4,100 observations can be expensive. No full-campaign completion-time claim has been made.

## Queue and resources

`code/scheduler.py` runs on LLMOptimizer3 behind `/root/data/cys/baseline/campaigns/semanticopt_peptide_test10_5000_20261117`.

Before launch it requires complete/audited predecessor results, all ten 4,000-evaluation ledgers, and no remaining predecessor controller, worker, or inference processes. It does not kill predecessor services. It checks free storage, cgroup RAM, CPU availability, and GPU memory. Only the predecessor's GPU pairs [0,3] and [14,15], inference CPUs 6–7, and worker CPUs 8–17 are assigned. Other campaigns are not reconfigured.

Two untuned-Qwen replicas share one journaled broker. A separate live validation must pass real Qwen suggestion/selection, exact candidate membership, full worker/oracle scoring, resume, and long-context checks before production workers are admitted. Diagnostic oracle calls and three live validation proposals are validation overhead, excluded from production results and recorded separately. CPU scoring is checked against the existing SemanticOPT CPU implementation. The shared saved initialization scores are retained exactly; small CPU-versus-prior-GPU numerical differences are reported in offline validation evidence.

The first two production workers must save ten evaluations in total before admission grows toward ten workers. Further admission checks RAM and storage. Process/service/validation failures stop visibly and retain observations. There is no automatic replay of unknown oracle calls. Queue locking prevents duplicate schedulers; process identities include start ticks. On shutdown the scheduler terminates only its own children.

All configuration, initialization, numerical code, inference code, author VAE, and CPU upstream assets are pinned in `evidence/assets.json`. Standard library/package versions are recorded in `evidence/environment.json`.

## Inspect or stop

Read `queue_status.json`, `status.json`, `evidence/live_validation.json`, and `runs/<task>/status.json`. `code/scheduler.py --status` prints campaign totals. `services/queue_process.json` records the scheduler process identity.

Creating `STOP` prevents a waiting queue from starting. To stop an active campaign, send SIGTERM to the verified scheduler PID/start ticks in `services/queue_process.json`; it stops only owned children and preserves checkpoints. Resume is explicit; unresolved in-flight oracle requests require audit.

Final exports are `paired_results.csv`, `convergence.csv`, `summary.json`, and `RESULTS.md`. Per-task elapsed time separates GP fitting, acquisition/decoding, encoding, LLM calls, and oracle time. Validation runs are separate from the test ledgers.

## Recovery on 19 November 2026

See `operations/resume_20261119T022532Z/RECOVERY.md` for the reproduced GPU OOM, repaired prompt token accounting, explicit two-GPU text placement and checkpoint recovery audit. Previously scored observations are retained; the source revision and per-task recovery boundary are recorded.
