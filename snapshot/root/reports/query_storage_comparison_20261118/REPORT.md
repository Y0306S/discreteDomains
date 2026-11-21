# Query storage investigation and timing comparison

Generated 2026-11-18T13:46:36.343370+08:00

Previous runs did use RAM. In the matched eighteen-plan replay, RAM warm runtimes were **9.0% lower** than Lustre runtimes overall (geometric mean of per-plan median ratios), so the two backends did not give identical runtime. The cache-advice and historical comparisons are detailed below. This is a bounded sample, not a full test200 rerun.

## Verified historical storage

| Run / record | Database storage | Temporary-file storage / qualification |
|---|---|---|
| Oct 4 train800 restart | Five local block-storage copies; one remote RAM copy | Contemporary audit explicitly warned against pooling absolute runtimes across storage backends. |
| Oct 12 LOLBO/TPE test200 allocation | Six local block-storage copies and eight remote RAM copies | Saved pre-sequential plans establish this allocation; LOLBO completed Oct 13 and TPE Oct 22. Later config edits do not retroactively make every timing RAM-backed. |
| Oct 15 Random Search recovery | Eight RAM database copies on LLMOptimizer4 | SQL spill moved to Lustre after RAM DiskFull errors. |
| Oct 27 baseline storage migration record | Six local copies changed from block storage to RAM | Before/after plan snapshots retained; prior measured results retain their original conditions. |
| Nov 6 FSBO pilot and original test200 pool | Fourteen RAM copies and two warmed Lustre copies | Original test200 workers subsequently configured Lustre spill tablespaces. |
| Nov 7 FSBO test200 expansion | Four additional local RAM copies; total eighteen RAM and two Lustre copies across twenty workers | The additional RAM databases also used disk-backed spill tablespaces. |
| SemanticOPT query evaluation setup | Its provisioner creates private databases under the Lustre campaign directory; retained data directories resolve to Lustre | This setup differs from the RAM-backed FSBO pool. Directory inventory is preparation evidence, not a simultaneous-worker count. |
| Current paused ORPO+FSBO campaign, live verification | Fourteen RAM copies and ten Lustre copies | Six Lustre copies on LLMOptimizer2; fourteen RAM plus four Lustre copies on LLMOptimizer4. RAM workers still use Lustre spill tablespaces. |

Evidence: [historical inventory](historical_inventory.json), [live endpoints](live_inventory.json), [sample endpoint provenance](historical_sample_storage.json). The ten LOLBO/TPE sample endpoints map to five local block-storage copies and five remote RAM copies in the contemporaneous plan; individual FSBO sample endpoints are not recorded in their history metadata. Historical source documents are listed below.

## Matched replay design

- Eighteen saved plans: fifteen held-out test queries spanning fifteen CEB families, five plans each from completed FSBO, LOLBO and TPE evaluations; plus three saved warm-start plans for failure-affected tasks CEB_11A121/122/123.
- The fifteen baseline plans were distinct actual successful SQL calls, excluding Bao imports and cache hits. Selection used the median distinct plan among historical runtimes from 0.05 through 8 seconds. This deliberately bounded sample is not the full 200-query evaluation or a random estimate of campaign performance.
- One verified physical PostgreSQL base backup cloned into private RAM and Lustre directories on LLMOptimizer2. Both copies passed pg_verifybackup with the same manifest. Production database instances and optimization journals were not migrated or resumed.
- Same PostgreSQL 16.15, data, statistics, executable SQL/hints, three CPU cores (0–2), 1 GiB shared_buffers, 64 MiB work_mem, up to two parallel workers, JIT on, 8 GiB temporary-file limit. Full pg_settings comparison differed only in paths, ports and sockets. Results are wall-clock seconds, using the production RuntimeOracle with read-only sessions.
- Test clusters use the default empty temp_tablespaces setting: spill follows the database directory (RAM for the RAM test, Lustre for the Lustre test). Historical RAM-plus-Lustre-spill arrangements are therefore a separate condition for spill-heavy plans.
- One first pass per storage, then five timed repetitions per plan with shuffled query order and alternating storage order. No optimizer or LLM inference work is included. Timeouts are capped at 60 seconds; failed/censored outcomes are never substituted into successful-runtime averages.
- Separately, both private test servers were restarted per plan to clear shared_buffers, and POSIX_FADV_DONTNEED was applied only to the private Lustre database relation files. mincore checked resulting OS-cache residency. This is an advisory cache test, not proof that remote Lustre/server/controller caches are cold. No global cache dropping or production-server restart occurred.
- EXPLAIN ANALYZE/BUFFERS and separate result-hash checks were run outside the timed repetitions. Data counts for title/name/movie_info/cast_info matched.
- An additional matched test covers every sampled plan that spilled in EXPLAIN: RAM tables with a private Lustre temporary tablespace versus Lustre tables with Lustre spill, one warm-up and three paired repetitions, plus separate hybrid EXPLAIN and result checks. Only the test clone catalog was changed to create/drop this temporary tablespace.

## Results

- Warm repetitions: **180 calls**; geometric mean of per-plan RAM/Lustre median ratios **0.910**, or **-9.0%** RAM runtime relative to Lustre.
- Mean SQL runtime over these equally weighted plans: RAM **2.964 s**, Lustre **3.266 s**. This is a sample average, not the campaign's evaluation-time average.
- **5/18** plans within ±5%; **14/18** within ±10%. Per-plan warm ratios range **0.752–0.975**.
- Private-file cache-advice replay: **18/18 successful pairs**, geometric mean RAM/Lustre ratio **0.185** (-81.5%); per-plan range **0.091–0.453**. Only one replay per storage per plan in this diagnostic. Non-completed calls: **0**, retained separately in summary.json and excluded from the successful-pair ratio.
- Lustre client-cache residency after advice: **0.00%–0.00%** of examined relation pages. Files already warmed by prior reads explain why an ordinary replay can hide storage-read costs.
- EXPLAIN checks: **18/18** matching plan shapes; **18/18** matching unordered query-result hashes. **5** sampled plans wrote temporary blocks in EXPLAIN, so spill-heavy behavior is only covered to that extent.
- Matching the historical RAM-table/Lustre-spill arrangement on **5 spilling plans**, three paired repetitions each: RAM/Lustre geometric mean ratio **0.915** (-8.5%). Both conditions use Lustre for spill. Hybrid validation matched **5/5** query results and **5/5** plan shapes. This subgroup ratio is not the full eighteen-query ratio.
- Recorded timing calls: **292**, non-completed **0**; EXPLAIN errors **0**. These short tests cannot certify that intermittent storage faults are repaired.

## Comparison with the saved earlier test timings

For the fifteen completed baseline plans, new RAM medians / saved historical timings have geometric mean **0.954** (-4.6%); new Lustre medians / historical timings **1.053** (+5.3%). The RAM/historical per-plan range is **0.763–1.051**.

This historical comparison is descriptive: the earlier measurements were single executions under different machine, CPU and concurrent-load conditions, and some used different spill placement. It does not attribute those differences solely to RAM. Full per-query values are in comparison.csv.

## Interpretation and recommendation

RAM use is established historically. Runtime equality must be assessed from the controlled ratios above, not inferred from database folder names or one aggregate average. Warm-cache agreement does not establish equivalence for initial reads, spill-heavy queries or a heavily contended full campaign.

For a paper comparison, keep base-data and spill placement, CPU allocation, cache policy and timing protocol consistent across methods. Preserve historical storage boundaries; do not relabel or silently merge old and new seconds. A production RAM migration still needs its own recorded decision, and this investigation did not perform one.

A RAM-table migration would bypass Lustre for base-table reads. If spill files and optimization journals remain on Lustre, those operations still depend on the same storage service. RAM placement is a workaround for selected I/O paths, not a repair of the shared filesystem.

The campaign remains paused with all saved scores intact. Successful isolated reads do not resolve the previously observed I/O errors/short reads on the production data paths. The RAM probe establishes a feasible alternative to investigate, not proof of storage repair.

## Per-query timings

| Query | Saved method | Historical s | RAM median s | Lustre median s | RAM/Lustre | Cache-advice RAM/Lustre | Hybrid RAM/Lustre |
|---|---|---:|---:|---:|---:|---:|---:|
| CEB_10A32 | FSBO | 2.799 | 2.944 | 3.189 | 0.923 | 0.128 | not sampled (no observed spill) |
| CEB_11A100 | LOLBO | 2.240 | 2.281 | 2.345 | 0.973 | 0.174 | not sampled (no observed spill) |
| CEB_11B147 | TPE | 4.069 | 4.053 | 4.407 | 0.920 | 0.233 | 0.944 |
| CEB_1A1020 | FSBO | 2.858 | 2.922 | 3.759 | 0.777 | 0.293 | 0.764 |
| CEB_2A19 | LOLBO | 2.993 | 2.956 | 3.159 | 0.936 | 0.185 | not sampled (no observed spill) |
| CEB_2B170 | TPE | 3.352 | 2.557 | 3.015 | 0.848 | 0.198 | not sampled (no observed spill) |
| CEB_2C202 | FSBO | 0.263 | 0.210 | 0.225 | 0.932 | 0.091 | not sampled (no observed spill) |
| CEB_3A1351 | LOLBO | 2.442 | 2.439 | 2.501 | 0.975 | 0.209 | not sampled (no observed spill) |
| CEB_3B23 | TPE | 5.572 | 5.546 | 5.781 | 0.959 | 0.276 | 0.986 |
| CEB_5A109 | FSBO | 1.920 | 1.679 | 1.826 | 0.919 | 0.113 | not sampled (no observed spill) |
| CEB_6A10 | LOLBO | 1.182 | 1.193 | 1.296 | 0.921 | 0.103 | not sampled (no observed spill) |
| CEB_7A102 | TPE | 4.831 | 4.235 | 4.577 | 0.925 | 0.231 | 0.929 |
| CEB_8A120 | FSBO | 4.039 | 3.852 | 5.119 | 0.752 | 0.259 | not sampled (no observed spill) |
| CEB_9A112 | LOLBO | 2.217 | 2.227 | 2.426 | 0.918 | 0.148 | 0.969 |
| CEB_9B37 | TPE | 1.729 | 1.769 | 1.858 | 0.952 | 0.139 | not sampled (no observed spill) |
| CEB_11A121 | BOLT30_warm_start | 13.590 | 8.312 | 9.551 | 0.870 | 0.453 | not sampled (no observed spill) |
| CEB_11A122 | BOLT30_warm_start | 1.487 | 1.342 | 1.386 | 0.968 | 0.190 | not sampled (no observed spill) |
| CEB_11A123 | BOLT30_warm_start | 2.990 | 2.283 | 2.413 | 0.946 | 0.178 | not sampled (no observed spill) |

## Artifacts and source documents

- [Timing plot](comparison.png), [PDF plot](comparison.pdf), [per-query CSV](comparison.csv), [machine-readable summary](summary.json).
- samples.json records exact executable SQL, plans, source hashes and original timings. timings.jsonl contains every timed call, including unsuccessful calls if any. explain.jsonl contains execution plans and result hashes.
- provision.log and clusters.json record backup verification and isolated server settings. environment.json and setting_differences.json record PostgreSQL configuration checks. cache_probe.jsonl records cache-residency checks.
- /root/reports/queryopt_health_20261005/README.md
- /root/data/cys/logs/query_test200_baselines_20261012/plan.before_sequential_unique.json and stage.lolbo.complete.json / stage.tpe.complete.json
- /root/data/cys/logs/query_random_storage_20261015/README.md
- /root/data/logs/query_fsbo_pilot_20261106_v1/README.md
- /root/data/logs/query_fsbo_test200_50s_20261106_v1/storage_configuration.json
- /root/data/logs/query_fsbo_local_expansion_20261107_v1/README.md and plan.json
- /root/data/cys/baseline/campaigns/semanticopt_eval200_20261110/code/host.py:98 and historical_semanticopt_storage.json
- /root/data/cys/baseline/campaigns/query_our_fsbo_vae80_test200_20261117/resources.json

Temporary test cluster cleanup is recorded separately in cleanup.json. The scripts and measurement evidence are retained for review.
