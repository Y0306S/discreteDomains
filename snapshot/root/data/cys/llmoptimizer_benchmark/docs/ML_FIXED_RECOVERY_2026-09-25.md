# ML fixed-split recovery - 2026-09-25

Follow-up: [Approved 48-hour policy, worker stops, and remaining-task inventory](ML_FIXED_READINESS_2026-09-25.md). LOLBO/random search are now stopped; TPE continues with the launch settings documented below. The following sections preserve the initial campaign record.

Launched at 15:36 Asia/Shanghai on 2026-09-25. This is the active follow-up to [the run issue register](RUN_ISSUES_2026-09-25.md). Scope: LOLBO, random search, TPE, and proposals for the remaining ML comparison issues.

## What is running

| Method | Fresh tasks queued | Completed tasks reused | Worker location | Concurrent workers |
| --- | ---: | ---: | --- | ---: |
| LOLBO | 110 | 22 | Controller | 4 |
| Random search | 112 | 20 | LLMOptimizer4 | 4 |
| TPE | 132 | 0 | Controller | 4 |

Each method targets the same 132 dataset/algorithm pairs over 77 datasets. The fresh queues total 354 tasks. Each queue starts with `black_friday/extra_trees`, then proceeds through the remaining tasks in increasing dataset size. The slow failing task therefore runs alongside the coverage expansion.

Random search reuses only clean 100-observation trajectories. Its previously completed `rgs/colleges/extra_trees` task had five non-finite evaluations and is included in the fresh queue. All incomplete legacy attempts, including the 66-row LOLBO `black_friday/extra_trees` attempt, are preserved separately and restarted under the new timing policy. TPE starts fresh for all 132 tasks so its new campaign has one evaluation policy.

The superseded local LOLBO/TPE/random-search workers and the old random-search workers on LLMOptimizer4 were stopped after process inventories were saved. The stop checks found no surviving selected processes. No old result files were deleted. TaskSetBO and BOLT were not changed.

## Exact protocol

Campaign directory:

`/root/data/cys/llmoptimizer_benchmark/benchmark_data/ml_test_20_fixed_recovery_20260925`

| Setting | Value |
| --- | --- |
| Source manifest | `/root/data/cys/blockdata_archive/llmoptimizer_benchmark/benchmark_data/ml_test_20_fixed/tasks_132.json` |
| Manifest SHA256 | `498029faf93894031bf612c67255be63c73f4b92e0a4d4c31e625febf41f5eec` |
| Trial budget | 100 attempted evaluations per task; failures remain counted and reported |
| Seed | 1 |
| Per-objective time limit | 43,200 seconds (12 hours), enforced in a separate CPU evaluator |
| Task optimization limit | 4,406,400 seconds: 100 maximum-length evaluations plus 24 hours for setup/optimizer overhead |
| Reason for extended limit | Successful legacy evaluations took up to 28,781.85 seconds, about eight hours |
| CPU allocation | Four allowed CPUs per task; estimator `n_jobs` capped at four; BLAS/OpenMP thread counts set to one |
| Controller allocation | LOLBO CPUs 0-15 with GPUs 0-3; TPE CPUs 16-31; 32 allocated CPUs against the 40-CPU quota |
| LLMOptimizer4 allocation | Random search CPUs 80-95; 16 allocated CPUs against the 64-CPU quota, alongside the existing workload |
| Oracle isolation | `forkserver`; the evaluator does not inherit initialized CUDA from LOLBO |
| Validation | Existing holdout protocol, test fraction 0.25 and validation fraction 0.35 |
| Final held-out-row test | Still skipped, consistent with the existing collection protocol |
| Samplers/search spaces | Existing implementations and failure semantics; no algorithm-specific restriction of `absolute_error` |
| Automatic retries | None; incomplete/failed processes are recorded for inspection |

The 51-day optimization ceiling is worst-case budget headroom, not an estimated completion time. The per-objective ceiling does not bound the existing final incumbent-refit step; that extra step is listed below for cleanup. Longer limits can address timeouts but cannot repair unrelated estimator/data exceptions.

The campaign contains a copy of the exact manifest, a code snapshot with checksums, and copies of all 77 dataset files. Every shared/local dataset file matched by SHA256 at preparation. The similarly named shared 19-dataset manifest is not used.

All fresh jobs use `/root/data/cys/baseline/envs/bl_lolbo_harness_cu118/bin/python`, including CPU-only TPE/random search with CUDA hidden. Versions at launch: Python 3.10, scikit-learn 1.3.2, NumPy 1.24.4, Optuna 4.9.0, PyTorch 2.0.1+cu118, XGBoost 2.1.4, LightGBM 4.6.0.

## Safeguards and verification

- Worker slots and individual tasks hold filesystem locks. Methods have explicit owning hostnames. Output and history paths include campaign, method, task type, dataset, and algorithm.
- The shared aggregate index now updates under a lock with atomic replacement. Task status is derived from trajectory length and finite-score counts, rather than trusting legacy summary completion flags.
- A managed worker allows one execution round and one dataset attempt. There is no automatic loop starting another fresh study after a task runs out of time.
- Resume loads the saved Optuna sampler study and checks that its trial objectives match the saved trajectory. Missing or inconsistent sampler checkpoints stop recovery instead of silently replaying a fresh sampler.
- The timeout test passed for successful evaluation, an exception, evaluation after parent CUDA initialization, and timeout cleanup of the evaluator plus its subprocess.
- LOLBO completed a 12-trial smoke run that exercised its GPU surrogate. A subsequent resume preserved those 12 rows and executed exactly one additional objective call.
- TPE locally and random search on LLMOptimizer4 each completed three-trial smoke runs, then restored their sampler state and executed exactly two additional calls to reach five. Saved configurations and scores in the prefix were unchanged.
- After launch, all 12 supervisors and optimizer processes were alive with the configured CPU affinities. New finite observations were present for all three methods. The tests do not establish that every expensive dataset/configuration will finish within 12 hours.

Smoke artifacts: `/root/data/cys/baseline/logs/ml_recovery_smoke_20260925`.

## Remaining ML fixes

| Issue | Applied in this campaign | Proposed follow-up |
| --- | --- | --- |
| RUN-01: different task coverage | User-selected old 132-pair manifest copied and hashed; missing LOLBO/RS coverage queued | Require matching manifest hashes and exact task-ID sets when computing comparisons. Keep the earlier 19-dataset results under their own protocol label. |
| RUN-02: training/retrieval boundary | This chosen split has zero exact-pair overlap with the inspected retrieval bank | Audit the checkpoint's SFT/ORPO/GRPO sources, retrieval bank, and warm-start presets against these 132 pairs. All 77 datasets occur under other algorithms in old training, so describe this as pair-held-out evaluation. Claiming unseen-dataset generalization would require a different split and compatible training. |
| RUN-03: duplicate RS workers | Old duplicate fleets retired; new owner/slot/task locks and separate history roots | Retain the legacy attempt inventory; select results explicitly by provenance. Add process-liveness checks before any future unattended recovery after host/supervisor failure. |
| RUN-04: repeated fresh restarts | Real sampler recovery exercised for all three methods; retry loops disabled for this campaign | Make checkpoint and trajectory commits transactional; stop on inconsistencies. Record RNG state where full uninterrupted-run reproducibility is required, including LOLBO's external torch RNG. |
| RUN-05: unequal effective limits | Fresh jobs share the 12-hour isolated objective policy and four-CPU cap | Audit ckpt32 and reused runs by their effective limits and hardware. Do not pool their wall-time curves with fresh runs. For a strictly matched resource comparison, collect replacement results for incompatible legacy tasks. |
| RUN-06: repeated TPE failures | Full TPE rerun with longer evaluations; per-evaluation timeout/exception events recorded | Separate timeout, estimator exception, invalid configuration, and non-finite score counts. Diagnose remaining failures before changing the sampler. If time infeasibility must affect TPE proposals, use a documented constraint/failure policy and apply the same feasible domain across methods; do not silently remove `absolute_error` only for TPE. |
| RUN-07: unreliable summaries | Per-algorithm outputs, locked aggregate updates, trajectory-based completion and failure counts | Rebuild a comparison index from all selected trajectories keyed by protocol/method/task/seed. Keep completion and successful-evaluation counts separate. Mark legacy summaries as non-authoritative. |
| Environment drift, newly confirmed | All fresh jobs share one Python/package environment and code snapshot | Old RS used NumPy 1.26.4, PyTorch 2.5.1+cu124, and XGBoost 3.2.0. Audit reused results before pooling; the reused `cls/airlines/xgboost` run is a concrete candidate for replacement under XGBoost 2.1.4. Preserve source environment metadata for every reused artifact. |
| Extra final refit | Existing behavior retained | With validation-only collection, skip the unnecessary final incumbent fit, or isolate/time it and record its cost separately. It currently runs after the 100 attempted objectives and outside their timeout wrapper. |
| Missing final test scores | Existing validation-only protocol retained | After selecting an incumbent solely on validation results, evaluate it once on the untouched row-level test partition. Store that result separately and never feed it back to the optimizer. |
| One random seed | Existing seed retained | After coverage and failure handling are stable, run at least three common seeds and report paired uncertainty across the same task set. |

Reused trajectories are evidence of clean completion on the selected tasks, not proof of a uniform resource or software protocol. This recovery makes coverage possible while retaining the distinction. No retraining, checkpoint rerun, new multi-seed campaign, or held-out-row test evaluation was launched as part of these proposals.

## Files and operations

Within the campaign directory:

- `protocol.json`, `tasks_132.json`, `queues/*.json`: experiment definition and fresh work lists.
- `code/`, `code_sha256.json`, `data_manifest.json`: launch source and data provenance.
- `reused/*.json`, `reused/<method>/.../trajectory.csv`: explicit selected legacy results and copied trajectories.
- `legacy_inventory.json`, `retirement/*.json`: preserved attempt and process inventories.
- `outputs/<method>/...`: new trajectories, sampler checkpoints, and per-run `evaluation_events.jsonl` files.
- `state/<method>/...`, `workers/*.json`, `logs/`: task outcomes, supervisor identities, and logs.

Read current trajectory progress from the controller:

```bash
/root/miniconda3/bin/python /root/data/cys/llmoptimizer_benchmark/benchmark_data/ml_test_20_fixed_recovery_20260925/code/examples/benchmark/ml_fixed_recovery.py status
```

The status command reports recorded worker state; confirm process liveness on the owning host when diagnosing a stalled task. Jobs run as detached processes and survive the terminal/SSH session ending. They do not automatically restart after a machine reboot.

To resume a stopped queue after inspecting its state, use the same `ml_fixed_recovery.py start --method lolbo` or `--method tpe` on the controller. For random search run `start --method random_search` on LLMOptimizer4 using the shared Python environment. Active worker slots are skipped. Completed task trajectories are skipped, and incomplete tasks require readable matching sampler checkpoints.

## Dataset coverage

The exact algorithms for each dataset are in `tasks_132.json`; this is a selected pair set, not all algorithms on each dataset.

Classification, 53 datasets:

`Amazon_employee_access`, `Click_prediction_small`, `Diabetes130US`, `Fashion-MNIST`, `GesturePhaseSegmentationProcessed`, `Internet-Advertisements`, `KDDCup09_appetency`, `PhishingWebsites`, `Satellite`, `ada`, `adult`, `airlines`, `albert`, `amazon-commerce-reviews`, `bank-marketing`, `blood-transfusion-service-center`, `car`, `churn`, `cnae-9`, `covertype`, `credit-g`, `dilbert`, `dionis`, `dna`, `eucalyptus`, `fabert`, `gina`, `jannis`, `jasmine`, `jungle_chess_2pcs_raw_endgame_complete`, `kc1`, `kick`, `kr-vs-kp`, `madeline`, `mfeat-factors`, `micro-mass`, `nomao`, `numerai28.6`, `okcupid-stem`, `ozone-level-8hr`, `pc4`, `phoneme`, `porto-seguro`, `qsar-biodeg`, `segment`, `sf-police-incidents`, `shuttle`, `steel-plates-fault`, `sylvine`, `volkert`, `wilt`, `wine-quality-white`, `yeast`.

Regression, 24 datasets:

`Allstate_Claims_Severity`, `Mercedes_Benz_Greener_Manufacturing`, `Moneyball`, `QSAR-TID-11`, `SAT11-HAND-runtime-regression`, `Santander_transaction_value`, `abalone`, `black_friday`, `boston`, `colleges`, `diamonds`, `elevators`, `house_16H`, `house_sales`, `nyc-taxi-green-dec-2016`, `pol`, `quake`, `sensory`, `socmob`, `tecator`, `topo_2_1`, `us_crime`, `wine_quality`, `yprop_4_1`.
