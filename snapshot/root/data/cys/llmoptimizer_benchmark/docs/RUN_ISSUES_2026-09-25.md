# Experiment run issues - 2026-09-25

Follow-up at 15:36 Asia/Shanghai: the user selected the old 77-dataset, 132-pair split. A [versioned ML recovery campaign](ML_FIXED_RECOVERY_2026-09-25.md) is now running 110 fresh LOLBO tasks, 112 fresh random-search tasks, and all 132 TPE tasks, with revised objective time limits and launch safeguards. The old ML workers were retired with their artifacts preserved. The issue states and counts below remain the original audit snapshot; the follow-up records applied changes, remaining proposals, and newly identified environment drift.

Status: OPEN. Evidence checked on 2026-09-25, approximately 13:52-13:59 Asia/Shanghai (UTC+08:00).

Scope: local ckpt32, LOLBO and TPE; random search locally and on LLMOptimizer4; BOLT query-plan optimization on LLMOptimizer3; TaskSetBO on LLMOptimizer4.

This is a problem register and a snapshot of the existing runs. Proposed fixes below have not been applied. Counts will change while workers continue.

## Current snapshot

| Run | Machine | Tasks reaching the configured trial count | Recorded progress |
| --- | --- | --- | --- |
| ckpt32 | Local | 132/132 | 13,200 trajectory rows; last trajectory update at 04:11 today |
| LOLBO | Local | 131/132 | 13,166 rows; black_friday/extra_trees at 66/100 |
| TPE | Local | 131/132 | 13,191 rows; black_friday/extra_trees at 91/100 |
| Random search | Local and LLMOptimizer4 | 124/132 | 12,863 rows; eight tasks incomplete |
| BOLT query plans | LLMOptimizer3 | 800/800 completion markers | All 800 CSVs present; all 800 logged 1,010 calls against a configured 1,000-call budget |
| TaskSetBO | LLMOptimizer4 | 635/686 | 64,674/68,600 observations; 24 BO workers active |

ML counts use the longest saved trajectory for each requested dataset/algorithm/optimizer, following the launcher's completion convention. They are not a sum across retries, do not necessarily describe the currently active attempt, and do not mean that every evaluation succeeded. BOLT completion uses markers plus logs, since its CSV contains the retained training points rather than one row per budgeted proposal.

## Issue register

P1: affects valid comparisons, result integrity, or completion behavior. P2: affects throughput, budget accounting, or interpretation.

| ID | Priority | Problem | State |
| --- | --- | --- | --- |
| RUN-01 | P1 | ML methods use different test splits under the same manifest filename | Confirmed, open |
| RUN-02 | P1 | Existing ckpt32 retrieval bank contains tasks from the newer dataset-held-out test set | Confirmed, open; checkpoint training provenance still needs auditing |
| RUN-03 | P1 | Eight random-search tasks run concurrently on two machines and share history paths | Confirmed, open; resulting corruption has not been established |
| RUN-04 | P1 | Partial ML runs are restarted rather than resumed, repeatedly consuming the budget | Confirmed, open |
| RUN-05 | P1 | ML evaluation time limits differ, and CUDA initialization disables the timeout wrapper | Confirmed, open |
| RUN-06 | P2 | TPE repeatedly spends an hour on failing absolute_error configurations | Confirmed, open |
| RUN-07 | P1 | Shared ML summaries lose algorithm results and can report incomplete work as complete | Confirmed, open |
| RUN-08 | P2 | TaskSetBO inner training receives one CPU despite four-thread configuration | Confirmed, open; exact affinity-changing code still needs locating |
| RUN-09 | P2 | BOLT executes an extra batch beyond its configured budget | Confirmed, open |

## RUN-01: Different ML test splits

The two repository copies have the same relative manifest path, `benchmark_data/ml_test_20_fixed/tasks_132.json`, with different contents.

| Runs | Repository | Exact source match | Datasets | Dataset/algorithm pairs |
| --- | --- | --- | --- | --- |
| LOLBO and random search | `/root/data/cys/llmoptimizer_benchmark` | `ml_ds_test_19.jsonl` | 19 | 132 |
| ckpt32 and TPE | `/root/data/cys/blockdata_archive/llmoptimizer_benchmark` | `ml_test_20_fixed.jsonl` | 77 | 132 |

There are only 23 common pairs across 17 common datasets. Each suite has 109 pairs absent from the other. The equal task counts do not make the suites comparable.

The older split holds out dataset/algorithm records: all 77 test datasets also appear in `ml_train_80_fixed.jsonl` under other algorithms, although the train/test pairs themselves are disjoint. The newer split keeps complete datasets on one side: 76 train datasets and 19 test datasets, with zero dataset overlap. Its purpose is documented in [split_ml_by_dataset.py](/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/scripts/dsplit/split_ml_by_dataset.py:2).

Some baseline launcher headers say `ml_ds_test_19` even in the blockdata copy that reads the older manifest. Both the labels and the reused filename conceal the mismatch. The repository has no commits for these launchers, so the files establish the mismatch but do not establish who chose to mix the splits or why.

Impact: full-suite means, ranks, win rates, and anytime curves cannot be compared across all four methods as though they used the same tasks.

Proposed resolution: select and name one evaluation protocol, assign immutable manifest IDs/hashes, validate the same task IDs before launch and aggregation, and collect missing tasks under that protocol. Preserve existing results with their original split labels.

## RUN-02: Reusing ckpt32 on the newer test set needs a training/retrieval audit

The current ckpt32 results identify this retrieval bank:

`/root/data/cys/blockdata_archive/llmoptimizer_benchmark/history_data/experience_bank_v3_train80.jsonl`

Direct parsing found 515 unique dataset/algorithm pairs in the bank. It includes 107 of the newer test set's 132 pairs and covers all 19 newer test datasets. It includes zero pairs from the older 132-pair test set. Separately, the old `ml_train_80_fixed.jsonl` contains 109 of the newer test pairs.

The active evaluation used adapter:

`/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_v4_pareto32_v12/ckpt32`

Evidence: saved ckpt32 `model_info.config.optimizer_params`, [evaluation launcher](/root/data/cys/blockdata_archive/llmoptimizer_benchmark/examples/benchmark/run_ml_test_20_fixed_ckpt32_rec4_ws3.sh:17), both source splits, and the bank itself.

Impact: simply changing ckpt32's evaluation task list to the 19-dataset split while retaining this bank would expose the optimizer to test-dataset experience. This finding does not establish that the present older pair-held-out evaluation used its own test pairs for training. The adapter's complete SFT/ORPO/GRPO lineage has not been audited here.

Proposed resolution: audit checkpoint training, warm-start presets, and retrieval inputs against the chosen split; rebuild/retrain any incompatible components before claiming performance on unseen datasets.

## RUN-03: Duplicate random-search work and shared writers

At the snapshot, eight tasks were active both locally and on LLMOptimizer4:

| Dataset | Algorithms |
| --- | --- |
| APSFailure | adaboost |
| airlines | extra_trees, random_forest |
| kick | adaboost |
| sf-police-incidents | adaboost, extra_trees |
| black_friday | extra_trees, random_forest |

Both fleets use seed 1 and output root:

`/root/data/cys/llmoptimizer_benchmark/benchmark_data/ml_test_20_fixed_rs100`

They also use the same history root. History files are keyed by task type, metric, dataset, optimizer, and algorithm; machine and run ID are absent. Completed attempts can therefore replace the same history file. Individual timestamped CASH directories are separate, but the history exports and dataset summaries are shared.

The local fleet still had nine parent workers, including `kick/extra_trees`, for which a 100-row trajectory already existed from another attempt. LLMOptimizer4 had eight parent workers. The shared launcher does not have the per-task `flock` guard present in the blockdata launcher.

Impact: duplicate compute, conflicting final histories, and ambiguous result selection. Simultaneous work and shared write targets are confirmed; corrupted JSON or a specific overwritten observation sequence has not been proven.

Proposed resolution: establish one owner per task across machines; preserve and inventory existing attempts before retiring duplicates; use shared task claims and run-specific history/summary paths.

## RUN-04: Partial runs restart from scratch

In the shared [runner](/root/data/cys/llmoptimizer_benchmark/examples/benchmark/run_toy_categorical_optimizers.py:835), `--resume` checks whether an existing trajectory has reached 100 rows. An incomplete task calls `run_one_optimizer` again, which resets the knowledge-database singleton and creates a new CASH optimizer. The runner can do this for up to 12 rounds. It does not restore the partial random-search optimizer state.

Observed for `black_friday` by approximately 13:58:

| Algorithm | Saved attempts with trajectories | Longest attempt | Sum of recorded rows across attempts |
| --- | --- | --- | --- |
| extra_trees | 22 | 60/100 | 1,147 |
| random_forest | 22 | 60/100 | 1,261 |

The timestamps show repeated roughly six-hour runs, consistent with the local 21,600-second task limit. These totals count repeated work and must not be added together to claim a valid 100-trial optimization run. The newest remote extra_trees attempt had only about 18 rows while the reported best historical attempt had 60.

Impact: substantial compute has been spent without finishing these tasks; active processes and a high aggregate row count can hide repeated restarts. With the same seed, repeated sampling prefixes are also possible, but exact prefix equality was not established in this audit.

Proposed resolution: implement real persistence and restore for optimizer history and random state, or use an explicitly labeled fresh run with a sufficient task budget. Add a guard against repeatedly starting a task that cannot reach its target within its allotted time.

## RUN-05: Inconsistent evaluation limits and disabled timeouts

Current worker arguments:

| Method/fleet | Task limit | Per-evaluation limit |
| --- | --- | --- |
| Local random search | 21,600 s (6 h) | 1,800 s (30 min) |
| Random search on LLMOptimizer4 | 250,000 s (69.4 h) | 10,000 s (2.78 h) |
| Local TPE | 604,800 s (7 days) | 3,600 s (1 h) |
| Local LOLBO | 604,800 s (7 days) | 3,600 s nominal; bypassed after CUDA initialization |
| ckpt32 launcher | 604,800 s (7 days) | 3,600 s nominal; same CUDA-sensitive wrapper requires audit |

[wrapper_evaluater](/root/data/cys/llmoptimizer_benchmark/mindware/components/optimizers/llm_optimizer_utils.py:14) sets `timeout = None` whenever `torch.cuda.is_initialized()` is true. The comment explains that this avoids hangs when forking after model/CUDA initialization, but it also removes the evaluation time bound. LOLBO uses this wrapper through `PT4OptOptunaOptimizer`.

The remaining LOLBO task logged nine successful evaluations taking more than 3,600 seconds; the longest logged evaluation took 28,781.85 seconds, approximately eight hours. It was still CPU-active at 66/100, with its last saved trial at 12:04.

Impact: identical configurations can be accepted for one method and timed out for another; runtimes and scores reflect different effective budgets. A nominal one-hour command-line setting does not bound the CUDA-initialized evaluation path.

Proposed resolution: choose a common evaluation resource/time policy and implement a timeout-safe evaluation process for CUDA-based optimizers. Record the effective policy in each run manifest and avoid silently mixing existing attempts with different limits.

## RUN-06: TPE is repeatedly evaluating configurations that time out

For local TPE on `black_friday/extra_trees`, 73 of the 91 saved trajectory rows have non-finite performance. All 73 use `extra_trees:criterion = absolute_error`. The last 24 saved rows were consecutive failures, with the log showing approximately one timeout per hour. Only 18 rows have finite performance.

Evidence:

- [TPE worker log](/root/data/cys/baseline/logs/ml_test_20_fixed/tpe_rgs_black_friday_extra_trees_20260921_100331.log)
- `benchmark_data/ml_test_20_fixed_tpe100/rgs/black_friday/extra_trees/tpe/CASH-tpe(1)-holdout_black_friday_extra_trees_tpe_2026-09-22-10-59-49-801500/1790045990_opt_tpe_traj.csv` under the blockdata repository

Impact: the final task is advancing its attempt counter while obtaining little new usable performance information. The evidence associates the failures with expensive absolute-error tree fitting; it does not prove the worker is deadlocked or that every failed candidate is identical.

Proposed resolution: investigate evaluation cost and failed-trial handling, and decide a consistent policy for infeasible configurations across methods. Any change to the allowed search space or timeout policy must be versioned and applied consistently.

## RUN-07: Summaries are not a reliable task-level result index

The shared runner writes `output/<task_type>/<dataset>/summary.json`, while each job evaluates one algorithm. Summary entries are keyed by optimizer rather than by optimizer and algorithm. Algorithms therefore share the same summary destination and temporary filename.

Both shared ML suites had 19 dataset summary files for 132 tasks. As a concrete example, `cls/credit-g/summary.json` listed only `xgboost` for both LOLBO and random search, although the manifest contains all seven algorithms. Other algorithm trajectories still exist in CASH directories.

There is a second completion-reporting problem: after the bounded retry loop gives up, the shared runner's final summary call passes `completed_variants=optimizers` unconditionally. A successful process exit or a completed flag can therefore coexist with an incomplete trajectory. The ckpt32 tree also had 131 summary result entries despite 132 complete task trajectories, reinforcing the need to derive coverage from task artifacts.

Evidence: [summary writer](/root/data/cys/llmoptimizer_benchmark/examples/benchmark/run_toy_categorical_optimizers.py:594), [final summary call](/root/data/cys/llmoptimizer_benchmark/examples/benchmark/run_toy_categorical_optimizers.py:900), and saved dataset summaries.

Impact: aggregating `summary_all.json` or dataset summaries alone can lose algorithm results or misclassify completion. Different jobs may also race on a shared `.json.tmp` file; that mechanism is present, but this audit did not establish a current exception caused by it.

Proposed resolution: key artifacts by dataset, algorithm, optimizer, seed, and protocol/run ID; rebuild task indexes from preserved trajectories; compute completed status from the actual budget reached; use unique temporary files and appropriate locking for shared indexes.

## RUN-08: TaskSetBO inner training is restricted to one CPU

Two process snapshots on LLMOptimizer4 confirmed:

- All 24 BO parent processes had affinity to four CPUs each.
- All 12 active `task_set.train_inner` subprocesses in the sampled snapshots had affinity to one CPU each.
- Sampled inner processes had five OS threads, `OMP_NUM_THREADS=4`, and `TF_NUM_INTRAOP_THREADS=4`.
- The host exposes 96 logical CPUs, but its cgroup `cpu.max` is `6400000 100000`, a 64-CPU-equivalent quota. Eight random-search parent jobs are also active there.

This does not match the four-CPU inner-training configuration described in [TASKSET_BO_64CORE.md](/root/data/cys/llmoptimizer_benchmark/docs/TASKSET_BO_64CORE.md). The exact code that narrows child affinity and the resulting slowdown have not yet been measured.

Impact: inner-training threads must share a single allowed CPU, restricting throughput. Parent affinity alone gives a misleading view of the resources available to the actual training process. Restoring four CPUs to all 24 workers would also require revisiting concurrency against the 64-CPU quota and the concurrent random-search workload.

Proposed resolution: locate the child-affinity change, preserve the intended CPU set through training startup, and benchmark worker concurrency against the real cgroup capacity.

## RUN-09: BOLT executes 1,010 calls with a 1,000-call budget

The launcher configures `--bsz 10 --max_n_oracle_calls 1000`. Parsing final-result blocks across the two train800 logs found:

- `bolt_train800_bsz10.log`: 588 completed runs, all reporting 1,010 calls.
- `bolt_train800_bsz10_20260922_124255.log`: 212 completed runs, all reporting 1,010 calls.
- Neither log reported a task exhausting all retry attempts.

In [optimize.py](/root/data/cys/BOLT/optimization/query_plans/query_plan_optimization/lolbo_scripts/optimize.py:348), the loop updates its continuation flag at the start of the body but still executes acquisition after the flag becomes false. With batches of ten, it executes one extra batch.

Impact: all 800 runs used a reported call budget 1% above the configured limit. Final best scores may include improvements from that extra batch. The retained-point CSV has no proposal ordinal or oracle-call column, so it cannot on its own certify the best score exactly at call 1,000. Duplicate proposals count toward the budget while the GP data are kept unique, so CSV row counts below 1,000 are expected and are not evidence of missing runs.

Proposed resolution: enforce the budget before acquisition, persist an ordered call/batch trace, and label existing results as 1,010-call runs unless the exact 1,000-call result can be reconstructed. Completion markers should carry the run protocol instead of only `ok`.

## Evaluation and reporting limits

These are limitations of the current evidence, not all software failures:

- All inspected ML summaries have `test_perf = null`; launchers use `--skip-test-eval`. Existing scores describe optimization on validation data, not a final held-out-row test evaluation. This is distinct from holding out entire datasets for meta-training.
- ML runs use seed 1; the BOLT fleet uses seed 0; TaskSetBO uses seed 1 with one inner-training replica. The current suites do not provide variability across independent seeds.
- A 100-row trajectory can include failed or timed-out observations. Completion and valid-evaluation counts must be reported separately. In the sampled TaskSetBO training queue, 17 of 64,674 observations had recorded state 1 and 64,657 had state 0; failure states should be retained when reporting corpus quality.
- TaskSetBO is running the documented corpus protocol of 1,000 inner steps and one replica, not the paper's 10,000-step/five-replica protocol. Its 686 queued tasks exactly match the current training split, with zero queued test tasks. No TaskSet train/test queue mismatch was found.
- This audit did not verify every model's training lineage, data preprocessing equivalence, or numerical correctness. Those remain outside the confirmed findings above.

## Remaining ML tasks at the snapshot

| Method | Dataset | Algorithm | Longest saved trajectory |
| --- | --- | --- | --- |
| LOLBO | black_friday | extra_trees | 66/100 |
| TPE | black_friday | extra_trees | 91/100 |
| Random search | APSFailure | adaboost | 61/100 |
| Random search | airlines | extra_trees | 22/100 |
| Random search | airlines | random_forest | 98/100 |
| Random search | kick | adaboost | 41/100 |
| Random search | sf-police-incidents | adaboost | 46/100 |
| Random search | sf-police-incidents | extra_trees | 75/100 |
| Random search | black_friday | extra_trees | 60/100 |
| Random search | black_friday | random_forest | 60/100 |

## Recommended order of work

1. Preserve current artifacts and settle the common ML split, training/retrieval boundary, and resource policy before making cross-method claims.
2. Establish single ownership of random-search tasks and retain an explicit record of every existing attempt.
3. Repair partial resume, timeout enforcement, and per-task summary/completion reporting; verify with bounded runs before relaunching expensive tasks.
4. Correct TaskSetBO child affinity and choose concurrency using the 64-CPU quota.
5. Correct BOLT budget enforcement and retain honest labels for the completed 1,010-call runs.
6. Finish missing tasks and perform the final evaluation/seed checks required by the selected study protocol.

## Manifest evidence

Source split files live under `/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data`.

| Manifest | SHA-256 at audit time |
| --- | --- |
| `/root/data/cys/llmoptimizer_benchmark/benchmark_data/ml_test_20_fixed/tasks_132.json` | `94ac5277bfcb9ebf875494fe5bfe615ed10d2d10c260cdadcc8a94a97a7005a3` |
| `/root/data/cys/blockdata_archive/llmoptimizer_benchmark/benchmark_data/ml_test_20_fixed/tasks_132.json` | `498029faf93894031bf612c67255be63c73f4b92e0a4d4c31e625febf41f5eec` |

The newer test set contains: APSFailure, GesturePhaseSegmentationProcessed, SAT11-HAND-runtime-regression, airlines, black_friday, colleges, credit-g, elevators, eucalyptus, house_prices_nominal, kick, kr-vs-kp, mfeat-factors, ozone-level-8hr, segment, sf-police-incidents, shuttle, socmob, and steel-plates-fault. It has seven algorithms per dataset except airlines, which excludes adaboost: 19 * 7 - 1 = 132 tasks.

The old test set contains 17 of those datasets, excluding APSFailure and house_prices_nominal, plus: Allstate_Claims_Severity, Amazon_employee_access, Click_prediction_small, Diabetes130US, Fashion-MNIST, Internet-Advertisements, KDDCup09_appetency, Mercedes_Benz_Greener_Manufacturing, Moneyball, PhishingWebsites, QSAR-TID-11, Santander_transaction_value, Satellite, abalone, ada, adult, albert, amazon-commerce-reviews, bank-marketing, blood-transfusion-service-center, boston, car, churn, cnae-9, covertype, diamonds, dilbert, dionis, dna, fabert, gina, house_16H, house_sales, jannis, jasmine, jungle_chess_2pcs_raw_endgame_complete, kc1, madeline, micro-mass, nomao, numerai28.6, nyc-taxi-green-dec-2016, okcupid-stem, pc4, phoneme, pol, porto-seguro, qsar-biodeg, quake, sensory, sylvine, tecator, topo_2_1, us_crime, volkert, wilt, wine-quality-white, wine_quality, yeast, and yprop_4_1. Only the algorithm pairs explicitly listed in its manifest are included.
