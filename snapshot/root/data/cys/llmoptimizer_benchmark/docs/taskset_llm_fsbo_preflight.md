TaskSet LLM + FSBO min-max — current setup and online checks

The active proposal arm is **FSBO min-max**, replacing RGPE for adaptive proposals, cooldown turns and the five-duplicate fallback. The campaign is exactly the frozen **200-task** held-out split at `benchmark_data/taskset_s1b_s2g_train838_20261029_v2/test200.json`. Existing archived 210-task baselines remain historical results; filter them to these same 200 tasks when comparing.

The model service loads ORPO once on LLMOptimizer3 GPU 0. CPU workers on LLMOptimizer4 own separate TensorFlow oracle runtimes, histories, pending proposals and FSBO subprocesses. The persistent FSBO subprocess caches imports and the original checkpoint tensors, but constructs a fresh target model and optimizer for every request. Target adaptation does not leak from one task or round to the next. This avoids Python/model import overhead while retaining the frozen reference algorithm. FSBO stays in its original FP32 implementation; BF16 base computation plus FP32 LoRA applies to Qwen inference.

The trained FSBO checkpoint is pinned to SHA-256 `9cbf4555f2f1b9816f1e9f73a8a73c97597ad449905365247c9158aee9715d72`. Its settings remain: conditional 43-dimensional encoding, RBF kernel, two 128-unit feature layers, target min-max scaling, up to 100 fitting epochs with the reference early stopping, 256 distinct unseen candidates and expected improvement. Every successful observation, including BOLT warm starts, enters subsequent target fitting. Historical source GPs are still used to rank similar-task prompt cards during rounds 4–8; they are not the adaptive proposal arm.

The schedule retains the ML policy with the BO arm renamed to FSBO:

| Rounds | Selection |
| --- | --- |
| 1–3 | Three frozen BOLT full-SFT warm-start presets |
| 4–8 | LLM proposals |
| 9 | PUCB selects LLM or FSBO |
| 10–20 | PUCB, forcing FSBO immediately after an LLM-selected round |
| 21–100 | PUCB without the early cooldown |

Each LLM turn allows up to five proposals. A duplicate is checked against **all** attempted configurations, including failures. Its existing outcome is appended as a temporary recent reminder **in addition to** the unchanged Recent15 and Top/Bottom history. Repeated reminders accumulate during that retry sequence; they do not become new evaluations or replace the original history window. After five duplicates, FSBO supplies the configuration. The selector records an LLM attempt and credits the FSBO evaluation. There is no extra global model-call cap.

From round 50, LLM prompts use the existing deterministic 50/50 exploration/exploitation policy. Exploration lists counts for every optimizer family, including zero-count families and failures. Exploitation lists supported Good/Bad regions from successful observations. The ORPO adapter was trained with `p5_h0`; those prompt blocks are runtime additions. A successful wiring test does not establish their optimization quality.

The handoff now saves the selector state after selection inside `pending.json`. Restarting consumes the same pending configuration without making another proposal and restores the selector's selection count/RNG state. A stale pending file left after a committed evaluation is removed instead of repeating that evaluation. The existing separate history/state commit can still fail closed on a torn write; automatic recovery for that unrelated failure is not claimed.

The executable runner is `examples/benchmark/domains/run_taskset_llm_fsbo.py`. The dispatcher `queue_taskset_llm_rgpe.py` now accepts `bo_arm: fsbo` and emits FSBO commands. The prepared eight-worker config is `/root/data/cys/baseline/artifacts/taskset_llm_fsbo_preflight_20261031/workers.fsbo.disabled.json`; it remains disabled. It uses `/root/data/cys/baseline/envs/bl_taskset_lolbo/bin/python`, which supplies the existing codec's Torch/GPyTorch dependencies. The original HPO environment failed that import, so no libraries were installed into it; the checked existing FSBO environment resolved the startup failure.

Online checks are isolated under `/root/data/cys/baseline/artifacts/taskset_llm_fsbo_preflight_20261031`. They are preflight evidence, not production benchmark results. Full campaign settings remain 100 rounds; `--check --stop-after 12` bounds real objective evaluations without shortening the selector's 100-round schedule. The two selected tasks are `quadratic_family_seed0` and `mlp_family_seed1`, both members of the frozen test200. Each objective evaluation uses the real oracle, 1,000 training steps, VALID_OUTER at steps 0 and 1,000, and one replica.

The checks cover:

1. BOLT full-model inference: three valid presets per selected task, six configurations total, with no parser repairs. Full training is reused.
2. Complete ORPO loading: all 496 adapter tensors verified exactly against saved FP32 values; BF16 base inference on the idle A40 GPU 0.
3. Real FSBO inference: correct checkpoint, encoding probe, min-max scaling and 256 EI candidates.
4. Persistent implementation parity: posterior means, standard deviations and chosen EI index agree exactly with the frozen one-shot worker on the tested request.
5. Task isolation: process A, then a changed-label B request, then A again; A's posterior and index are identical and the subprocess PID stays the same.
6. Duplicate exhaustion: inject five repeated replies, then use real FSBO; verify one objective evaluation, five attempts, LLM-attempt accounting and FSBO credit.
7. Recent15 expansion: an old duplicate outside the normal selected history is appended while the entire original prompt and all 15 recent records remain intact.
8. Pending handoff recovery: resume the saved FSBO configuration with no new model call and preserve selector state.
9. Two real 12-evaluation target runs, including actual BOLT warm starts, ORPO rounds, similar-task retrieval, FSBO proposals and cooldown.
10. Real restart: stop the quadratic run after eight evaluations with round 9 persisted, then restart and consume that exact configuration.
11. Real round-50 exploration and round-52 exploitation prompts on checksum-verified archived target prefixes. These are shadow LLM proposals; no new oracle evaluations are added for them.
12. Inject one old duplicate, then send its retry prompt to the real ORPO model; verify it returns a valid unseen configuration with Recent15 preserved. This tests an injected failure, not the frequency of naturally occurring duplicates.
13. Offline regression checks for 100-round policy parity, HTTP concurrency, model identity, outage handling, prompt boundaries and pending-file recovery.
14. A disabled plan with exactly 200 unique tasks; no production queue launched. Live-training manifest source hashes remain unchanged.

Read `validation.json`, `fsbo_checks/checks.json`, `prompt_checks_final/checks.json` and `REPORT.md` in that directory for the final statuses and evidence. The FSBO parity test measured about 0.52 seconds for a warm request, versus about 4.52 seconds for the original one-shot invocation on the same small request; first persistent startup took about 16.43 seconds. These figures include different startup/cache effects and do not establish a full-campaign speedup.

Before a full launch, generate and freeze the remaining BOLT presets for the full test200, recheck CPU and memory availability, enable a reviewed copy of the disabled config and start the shared service. The completed online tests do not authorize a full 200-task campaign or prove final optimization performance.
