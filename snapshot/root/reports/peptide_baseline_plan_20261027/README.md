# Peptide Baseline And Transfer Training Plan

Status: approved and implemented. All 20,000 starting candidates passed rule checks and fresh-score revalidation. Train800 integrity/model-reload checks passed 800/800; one tiny oracle comparison discrepancy was reproduced as batch-size numerics. Both TPE/LOLBO component tests passed. Production test200 workers are launched at `/root/data/logs/peptide_test200_baselines_20261027_v2`; the full train800 frozen-VAE export is running, with FSBO/RGPE fitting queued behind it at `/root/data/logs/peptide_transfer_train800_20261027_v2`. See the campaign README and `/root/reports/peptide_train800_test200_audit_20261027/README.md` for evidence and actual launch state.

## Verified Current Progress

- The current peptide V5 campaign has 800 training tasks and 200 test tasks. A fresh filename-only inventory found all 800 training archives, no completed test archives and no remaining loose training task directories.
- Campaign: `/root/data/logs/peptide_bo_train800_20261027_v5b`.
- Frozen implementation: `/root/data/logs/peptide_bo_release_20261027_v5`.
- Its split replaces three incompatible training references while preserving the original held-out references. Training must use the campaign's pinned split and initialization, not an older train900/test100 split.
- Sample `pep_000` has 7,856 history rows but only 5,100 distinct measured sequences, including 1,588 feasible sequences. Cached rows must not be treated as independent training measurements.
- BOLT completed and validated checkpoint 200 on four ranks and the automatic controller launched a native resume from it. The new configuration saves every 500 updates and retains two checkpoints. First-update validation was still pending when this plan was prepared; check `current_run.json` and `checkpoint_policy_transition.json` for the current state.

## Shared Evaluation Protocol

Use the authors' verified peptide VAE checkpoint and the V5 per-task pools of 100 distinct measured initialization sequences. Optimize negative APEX bacteria-0 MIC: negative MIC is maximized, while reported MIC is minimized. Require at least 75% similarity to each task's own reference and report the best feasible sequence.

Preserve the V5 limits: at most 5,000 acquisition proposals and at most 5,000 fresh post-initialization APEX calls. LOLBO shares the fresh-call limit with recentering, so it can finish before 5,000 acquisition proposals. The 100 reused initialization observations count as the common initial budget. Cached proposals consume proposal slots; they do not make another fresh oracle call. Record both counts explicitly.

Persist the sampler/model state, RNG, task identity, counters, score cache and scored pending event. Recovery must not rescore a durably recorded result. Keep a bounded set of active files and pack each completed method/task run, including recovery and evidence data, into one checksummed archive.

## TPE

1. Implement plain Optuna TPE with one proposal followed by one decode/score/tell cycle. Batch size is exactly one; there are at most 5,000 post-initialization cycles.
2. Use the authors' frozen VAE throughout the task. TPE has no GP or VAE gradient updates.
3. Reuse the legacy latent search bounds of -4 to 4, and retain per-task initialization rather than the legacy shared reference-zero cache. Audit encoded initialization against those bounds; report any incompatibility rather than clipping it.
4. Record similarity violation as an explicit Optuna constraint. The verified native decoder and similarity check govern feasibility; do not silently repair candidates or let infeasible scores become incumbents.
5. Do not invoke the legacy runner unmodified. It silently converts `tpe` to `tr_tpe` unless `--allow-vanilla-tpe` is set, and its initialization code builds a shared reference-zero pool.
6. Test ask/tell restoration, initialization loading, infeasible candidates, duplicate scoring, budgets and final archive reload with real VAE/APEX inference before expanding workers.

## LOLBO

1. Reuse the frozen V5 implementation, with acquisition batch 50, GP initialization 20 epochs, GP update two epochs and learning rate 0.001.
2. Preserve the feasible search center, joint VAE/GP update after ten consecutive non-improving acquisitions, re-encoding and bounded recentering.
3. Add an evaluation worker queue selecting the pinned test IDs. The existing V5 worker explicitly selects training IDs; changing only a launch flag would not evaluate test200.
4. Preserve the training histories. Run evaluations in a separate recipe-bound output campaign.
5. Check a real task's acquisition, task-isolated remote inference, CPU joint update, restoration and accounting before expanding the evaluation queue.

## Transfer Corpus

Read each train800 archive's history with its manifest, recipe and reference identity. Verify the archive before using its observations. Include initialization and all distinct finite measured outcomes, including infeasible sequences with an explicit feasibility label. Deduplicate cached repeats within each task; retain inconsistent repeated scores for inspection rather than silently choosing a best score.

Re-encode recorded sequences with one immutable author VAE checkpoint. Saved task-specific latent coordinates cannot be pooled: V5 LOLBO changes each task's VAE during joint updates. Encoding is an offline preprocessing step and does not call APEX or rerun BO. Pin the encoding method, encoder RNG seed and normalization statistics using training sources only. Apply the same representation at target evaluation.

All eligible sources come from train800 only. Held-out test200 observations are never used for the corpus, normalization, checkpoint selection or source models. Current-target updates remain allowed. Produce a compact corpus and an eligibility/provenance manifest; no source-task or observation cap is proposed.

## FSBO

Use the existing original-FSBO implementation with a peptide-specific input adapter, not BOLT's selected-history GP baseline. Reuse the established TaskSet training recipe: hidden layers [128,128], RBF GP, no ARD, Adam learning rate 0.001, batch 50 and 100,000 task-batch optimizer updates. This is the existing benchmark recipe, not a claim that these are the upstream repository's defaults or that 100,000 optimizer updates equal its 100,000 outer epochs.

Use upstream-style per-task response min/max scaling, avoiding the problematic global-range label augmentation. Reserve a fixed 20% of eligible training tasks for validation and use predictive NLL to select the checkpoint, as in the validated TaskSet trainer. All eligible source tasks participate in training or validation; none are selected by first-ten retrieval. Record this division in the corpus manifest before fitting. Keep a recoverable optimizer/RNG checkpoint and reload-test the selected model.

At target evaluation, adapt the learned features and GP to the current task's observations. The peptide target adapter still needs EI candidate selection and explicit similarity handling; a trained feature checkpoint alone is not a complete runnable baseline.

## RGPE

Fit one source GP per eligible training peptide task, with at least two distinct measured sequences, using the shared frozen-VAE representation. Keep all distinct source observations; do not introduce an unapproved sample cap. Use the existing upstream transfer-HPO GP implementation, Matern-2.5 ARD kernel and normalized responses with MIC as a minimization target. Verify finite posterior means/nonnegative variances and identical predictions after reload.

Run eight independent one-core source-GP fit workers on machine 4. Each uses a deterministic task seed and checksummed source identity. Write the model bank through one coordinator into a resumable database/archive rather than creating thousands of independent model/metadata files. Only completed, verified fits enter the bank.

At evaluation, fit a target GP from that task's observations and combine source models through RGPE ranking weights. The constrained peptide target adapter and model-bank reader still need implementing. Full source fits can be expensive at approximately 5,100 observations per task; measure a representative and large task first to obtain a runtime estimate without reducing the production training extent.

## Resource Allocation

Machine 2's CPU quota is 40, even though its affinity exposes more CPUs. Its four-GPU BOLT job reserves CPUs 3-11 and 21-23, twelve physical cores. Machine 4 has 48 physical cores available within a 64-CPU quota and 256 GiB RAM, but its current container exposes no GPU device. Machine 3 remains unused.

| Work | Machine 2 CPU IDs | Machine 4 CPU IDs |
| --- | --- | --- |
| Existing BOLT full SFT | 3-11,21-23 | None |
| Three VAE/APEX services | 0,1,2 | None |
| TPE, one core per worker | 12-20,24-26: 12 workers | 10-27: 18 workers |
| LOLBO, one core per worker | 27-38: 12 workers | 28-45: 18 workers |
| RGPE source fits | None | 0-7: eight workers |
| FSBO meta-training | None | 8,9: one two-core trainer |
| Coordination/headroom | 39 | 46,47 |

This accounts for all 40 local quota cores and all 48 remote physical cores, including coordination. Set OMP/MKL/OpenBLAS threads explicitly and do not allocate SMT siblings as independent workers. The existing resource scanner has been corrected to detect the new four-GPU BOLT launcher and report CPU quota headroom.

Recreate the three previously tested VAE/APEX services on local GPUs 0,2,3 and loopback SSH tunnels for machine 4. Previous service measurements were approximately 1.9 GiB GPU memory each. Require a real peak-memory check, at least 8 GiB free after warmup, and adequate CPU RAM before expansion. The service keeps task weight banks separate and discards obsolete versions. FSBO and RGPE fitting remain on CPU initially; adding their GPU allocations is unnecessary for this layout.

Expand evaluation workers in stages after successful component checks. Monitor service wait time, RAM, CPU throttling and checkpoint I/O; CPU availability alone does not guarantee another worker increases throughput. As RGPE/FSBO training finishes, those ten remote cores join the evaluation queue, yielding up to 70 evaluation workers. Use shared queues with method/task locks so freed workers immediately take unfinished tasks.

## Storage

Machine 2's system disk has only about 2.6 GiB free and machine 4 about 11 GiB, so neither is suitable for a large complete source-GP bank. Shared storage has about 1.8 TiB free. Use compact, durable shared files for results and banks; stage bounded corpus/cache data in machine 4's tmpfs if memory allows, with durable source data and recovery state retained. RAM-backed scratch consumes the 256 GiB memory limit and disappears on container restart.

Schedule source export/model-bank bulk writes outside BOLT checkpoint saves. Checkpoint 200 took longer than the earlier 13-minute save on the same shared storage, supporting the move to the chosen 500-update cadence. Do not move or delete the verified checkpoint during its transition.

## Launch Order

1. Finish validating BOLT's native resume from checkpoint 200.
2. Use the confirmed test200 scope and freshly revalidated initialization pool; restore and test GPU services/tunnels, and validate the current recipe and package versions on both machines.
3. Implement and check the plain-TPE adapter and test-task LOLBO queue; use the new initialization and budget logic.
4. Start a few actual evaluation workers, then expand to the assigned 24 local and 36 remote workers after throughput/memory checks.
5. Export and validate the full train800 transfer corpus; no BO/oracle rerun is needed.
6. Validate source-GP reload and FSBO updates/reload, then launch eight RGPE fits and full FSBO training on their allocated remote cores.
7. Redirect completed training-worker cores to the evaluation queue. Keep one archive per completed method/task run and a compact resumable transfer model bank.
