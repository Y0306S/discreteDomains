# Peptide Test200 Baselines

The user approved test200 evaluation and full train800 transfer training. Starting-pool revalidation and train800 audits are documented in `/root/reports/peptide_train800_test200_audit_20261027/README.md`; a shared snapshot of the required initialization/report files is in `audit_snapshot` for machine 4.

## Running Evaluation

- Machine 2: 11 batch-one TPE workers and 12 batch-50 LOLBO workers, 23 total. CPU18 remains allocated to export until its handoff.
- Machine 4: 18 TPE and 18 LOLBO workers, 36 total.
- GPU inference services: two per GPU on machine 2, with loopback SSH forwarding from machine 4. The first replica on each GPU has a 5% allocator cap; the second has a 2% cap. Idle queue supervisors use CPU40; the 40 computation cores remain distinct.
- Existing four-GPU BOLT SFT continues with the same model, optimizer and effective batch. Live CPU affinity is now two cores per rank: 3-4, 6-7, 9-10 and 21-22. Machine 3 is unused.
- Worker identities, commands, PIDs and logs: `workers/local.json`, `workers/remote.json`.
- The queues accept only the pinned 200 held-out task IDs. Original train800 archives remain unchanged.

### Additional VAE Replica

`inference_expansion.py` adds the GPU1 replica and migrates five local and ten remote TPE worker managers using graceful termination and the existing durable resume logic. Task budgets, seeds, initialization, evaluated scores and optimizer state are preserved. The original three services continue serving the remaining workers and exporter. Pending tasks are waiting for a CPU worker, rather than being submitted VAE requests.

`workers/vae_routes.json` preserves these assignments when the normal launcher replaces a worker. The new service, dedicated remote tunnel, validation results and migration records are saved under `workers/vae_gpu1*` and `workers/vae_migration_*`. Validation compared the frozen posterior-mean encodings and seeded decoding against an existing replica and made zero oracle calls. GPU1's training allocator remains capped at 90%, with the inference allocator capped at 5%.

### Two Replicas per GPU

The subsequent approved expansion starts four additional replicas and redistributes 28 existing workers through scored checkpoints. All 59 worker managers are retained. No optimization recipe, evaluated score, initialization or random seed is reset.

| GPU | First Replica CPU / Port | Second Replica CPU / Port |
| --- | --- | --- |
| 0 | 0 / 18871 | 5 / 18875 |
| 1 | 39 / 18874 | 8 / 18876 |
| 2 | 1 / 18872 | 11 / 18877 |
| 3 | 2 / 18873 | 23 / 18878 |

`workers/bolt_live_cpu_allocation.json` records the live affinity change for every training thread. The original three Torch intra-operation threads remain configured; OS affinity limits each rank to two physical cores. The supervisor retains its original CPU lock descriptors, so these four released cores remain protected from unrelated launchers. Its historical command line and periodically rewritten `status.json` still list the original twelve cores; use this allocation sidecar for the live CPU reservation. Stop these additional inference replicas before restoring the original training CPU affinity.

`workers/vae_pairs_plan.json` and `workers/vae_pairs_migration_*.json` record the new routes and resumptions. Each second replica has its own `workers/vae_replica_<port>*.json` health, launch, tunnel and validation records. All four passed exact frozen-encoding and seeded-decoding comparisons without oracle calls. Inference is still serialized within each replica; concurrency comes from the eight independent service processes.

Every run reuses the same 100 freshly revalidated measured starting peptides. There are at most 5,000 acquisition proposals and 5,000 fresh post-initialization logical oracle evaluations. LOLBO recentering uses that shared fresh budget. Cached duplicates consume proposal slots and are logged separately. Reports retain both counters; the limits do not mean every LOLBO run must complete exactly 5,000 acquisition proposals.

TPE uses plain Optuna TPE in the frozen author VAE's 256-dimensional latent space, bounds -4 to 4, one proposal per ask/decode/score/tell round and explicit similarity constraints. Its common objective setup constructs an unused GP object and can print `Using GP DKL surrogate model`; TPE never trains or uses that GP, and never updates the VAE. This setup message does not mean TPE is a GP optimizer.

LOLBO uses the immutable, tested V5 implementation: constrained Thompson sampling, feasible centers, 20 initial/two update GP epochs, learning rate 0.001, joint VAE/GP updates after ten consecutive non-improving acquisitions, re-encoding and bounded recentering.

Completed method/task runs are one checksummed `.tar.gz` each. TPE archives contain its SQLite recovery database, history, progress and manifest. LOLBO archives contain checkpoint, history, progress and manifest. Active files and worker logs remain bounded. Scored pending-event recovery avoids repeating a durably recorded score.

## Validation

`/root/data/logs/peptide_test200_component_20261027_v2/report.json` records passing real-inference, fresh-process recovery, accounting, task isolation and archive checks for both methods. TPE additionally passes an injected recorded-pending-score recovery with oracle scoring disabled. LOLBO completes an actual joint update and recentering. These are shortened component tests; production limits remain unchanged.

All 200 test initialization pools pass sampled-latent bound checks. Frozen posterior-mean encoding repeats identically for the same batch; single-versus-batch differences are at most 1.79e-7 in the probe. The earlier callback-pickle failure and remote audit-path failure were fixed before affected production workers started.

## Transfer Training Queue

Queue root: `/root/data/logs/peptide_transfer_train800_20261027_v2`.

The full train800 exporter is running on CPU18 with GPU2 encoder service. It reads audited archives, deduplicates repeated task/sequence measurements and re-encodes them with one frozen author VAE posterior mean. It performs zero oracle evaluations. Original saved task-specific VAE coordinates are not pooled. Tiny cached-score variants that are exactly float32-equivalent are retained in provenance; larger conflicts stop export. No best-score selection is used for duplicates.

After all 800 sources are exported and verified, the queue runs a separate 100-update FSBO component/reload check, then launches the full 100,000-update FSBO schedule on machine 4 CPUs 8,9. Architecture, optimizer, label scaling and validation selection reuse the established TaskSet FSBO training loop: [128,128], RBF/noARD, Adam 0.001, batch 50, per-task minmax, fixed 160-task training-only validation split, predictive-NLL checkpoint selection. The component check is not production training.

RGPE fits a full upstream Matern-2.5 ARD source GP for every train800 task, with normalized responses, all distinct measurements and reload checks. A single full source fit establishes peak memory before concurrency increases to at most eight allocated CPUs 0-7. Exact ARD fits at roughly 5,100 observations can allocate large covariance-gradient arrays; the queue does not truncate observations to meet memory limits. Models go into a compact SQLite bank.

FSBO/RGPE fitting remains queued until export completes. Their constrained target-evaluation adapters and RGPE model-bank inference integration remain separate work; trained source artifacts alone are not completed evaluation baselines.

CPU handoff supervisors reuse CPU 18 after export, CPUs 0-9 after their training releases, and completed TPE worker CPUs for unfinished LOLBO tasks. They check process release before starting replacements. At most 70 evaluation cores are available across machines 2/4 once transfer training completes, while BOLT and inference reservations remain in place.

## Operations

Use `tools/launch_peptide_test200.py local` or `remote` with the existing campaign to resume missing worker managers; active recorded managers are preserved. `tools/peptide_test200.py run-task` resumes its task's durable state under a task lock. Never modify pinned recipes or code in-place during production; prepare a new version for behavioral changes.

`tools/peptide_training_queue.py` records local/remote queue state and explicit failures. A failure prevents downstream fitting. Inspect `queue_local_failure.json`, `queue_remote_failure.json`, source export status and model/training statuses before resuming the same queue. Avoid launching duplicate supervisors.
