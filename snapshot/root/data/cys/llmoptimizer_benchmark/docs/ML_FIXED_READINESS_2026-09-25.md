# ML fixed-split policy and remaining work - 2026-09-25

This follow-up records the user's decisions after the [initial recovery launch](ML_FIXED_RECOVERY_2026-09-25.md). Inventory snapshot: 2026-09-25 16:51:57 Asia/Shanghai.

## Approved policy

| Setting | Value |
| --- | --- |
| Scope | ckpt32, TPE, LOLBO, and random search on all 132 ML tasks |
| Per-evaluation wall-clock limit | 172,800 seconds (48 hours), enforced in an isolated forkserver process |
| Overall search timeout | Disabled: `time_limit = null` |
| Search budget | 100 attempted evaluations; failures and timeouts consume attempts |
| Existing result reuse | Reuse a single completed trajectory with exactly 100 finite evaluations, seed 1, and matching task/metric, including results collected under older limits |
| Other workloads | TaskSetBO and BOLT retain their independent policies |

Canonical policy: `/root/data/cys/blockdata_archive/llmoptimizer_benchmark/configs/ml_runtime_policy.json` (`ml_48h_100_attempts_v1`). A copy is saved with the inventory. A run that uses all 100 attempts but has failed evaluations still needs work under the clean-result criterion. Longer limits do not repair estimator/data exceptions or guarantee that every future evaluation finishes.

The updated controller runner accepts `--runtime-policy /root/data/cys/blockdata_archive/llmoptimizer_benchmark/configs/ml_runtime_policy.json`. This overrides its trial and timing arguments and enables isolated evaluation. The overall limit can also be disabled explicitly using `--time-limit none`.

The updated source is `/root/data/cys/blockdata_archive/llmoptimizer_benchmark/examples/benchmark/run_toy_categorical_optimizers.py`. Future managed launches must include the policy argument and retain `--max-rounds 1 --dataset-retries 1`. Remote launches require a new shared code snapshot containing these changes and a reachable policy path. The frozen recovery snapshot and its launcher still contain the previous timing settings; their existing start commands do not apply the new policy.

## Worker state

At the user's request, LOLBO and random-search workers and supervisors were stopped on the controller and LLMOptimizer4. None were found on LLMOptimizer3. Stop audits recorded no surviving selected processes. Existing trajectories and sampler checkpoints were preserved, and eight active task records were marked `stopped_by_user`.

TPE continues with its existing 43,200-second evaluation limit and 4,406,400-second overall limit. Four TPE tasks were active at the inventory snapshot: three still need a clean result, while the fourth already has a reusable clean legacy result. No new production runs were launched during this policy/inventory update.

## Exact remaining tasks

The authoritative manifest is `/root/data/cys/blockdata_archive/llmoptimizer_benchmark/benchmark_data/ml_test_20_fixed/tasks_132.json`, with SHA256 `498029faf93894031bf612c67255be63c73f4b92e0a4d4c31e625febf41f5eec`. It contains 132 dataset/algorithm pairs across 77 datasets. The identically named shared-tree manifest contains a different 19-dataset split.

| Method | Clean results reused | Still needs work | Active within remaining work | Remaining work not active |
| --- | ---: | ---: | ---: | ---: |
| ckpt32 | 115 | 17 | 0 | 17 |
| TPE | 121 | 11 | 3 | 8 |
| LOLBO | 35 | 97 | 0 | 97 |
| Random search | 26 | 106 | 0 | 106 |
| Total | 297 | 231 | 3 | 228 |

The remaining work comprises 17 ckpt32 trajectories with failures; 10 TPE trajectories with failures and one incomplete task; four incomplete and 93 unstarted LOLBO tasks; and one random-search trajectory with failures, five incomplete tasks, and 100 tasks without trajectories. Failure counts reflect the best inspected attempt for each task.

Inventory directory: `../benchmark_data/ml_test_20_fixed_readiness_20260925/`.

- [Report](../benchmark_data/ml_test_20_fixed_readiness_20260925/REPORT.md): counts and completed tasks containing failures.
- [All task records](../benchmark_data/ml_test_20_fixed_readiness_20260925/all_tasks.csv): all 528 combinations, selected evidence, counts, and recommended actions.
- [Full inventory](../benchmark_data/ml_test_20_fixed_readiness_20260925/inventory.json): inspected attempts, original time limits, failed-evaluation durations, and active-task evidence.
- [ckpt32 remaining](../benchmark_data/ml_test_20_fixed_readiness_20260925/queues/reuse_clean/ckpt32.json): 17 tasks.
- [TPE remaining](../benchmark_data/ml_test_20_fixed_readiness_20260925/queues/reuse_clean/tpe.json): 11 tasks.
- [LOLBO remaining](../benchmark_data/ml_test_20_fixed_readiness_20260925/queues/reuse_clean/lolbo.json): 97 tasks.
- [Random-search remaining](../benchmark_data/ml_test_20_fixed_readiness_20260925/queues/reuse_clean/random_search.json): 106 tasks.

`queues/ready_without_active/` excludes active work. Recheck process liveness before scheduling. The secondary `queues/strict_policy/` files describe a different requirement, collecting every result under the new policy; the user chose clean-result reuse instead. None of these inventory lists has been submitted for execution.

To refresh from the controller:

```bash
/root/miniconda3/bin/python /root/data/cys/blockdata_archive/llmoptimizer_benchmark/examples/benchmark/inventory_ml_fixed_tasks.py \
  --output-dir /root/data/cys/llmoptimizer_benchmark/benchmark_data/ml_test_20_fixed_readiness_20260925
```

## Verification and limits

The three focused policy tests passed, covering 100 attempts with simulated failures and no overall cap, continued enforcement of finite overall limits, and application of the 48-hour isolated policy. A real three-evaluation TPE smoke test completed with three finite scores, `time_limit: null`, and forkserver event records. That smoke test used a 60-second evaluation limit to validate the execution path.

Inventory checks verified the manifest hash, all 528 unique method/task combinations, queue totals, and 100 distinct finite evaluations in every reused trajectory. No inspection errors were recorded. Verification artifacts are in the inventory's `validation/` and `stopped/` directories.

Reused runs retain their original software, hardware, and timing provenance, including the previously documented random-search package differences. Their timing curves are not evidence of an identical resource protocol. The existing final incumbent refit remains outside the per-evaluation timeout, and these trajectories report validation scores; final held-out-row test evaluation is still skipped. Further ML issues and proposed fixes remain in the [initial recovery report](ML_FIXED_RECOVERY_2026-09-25.md#remaining-ml-fixes).
