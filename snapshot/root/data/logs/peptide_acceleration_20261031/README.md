Acceleration applied on 31 October 2026 to the running peptide FSBO and RGPE campaigns.

| Workload | Previous allocation | Current allocation |
| --- | --- | --- |
| FSBO target fitting | Four services sharing LLMOptimizer2 GPU 3 | One service on each LLMOptimizer2 A800 GPU, 0–3 |
| RGPE source fitting | 11 CPU workers on LLMOptimizer4 | Those 11 workers plus eight on LLMOptimizer3 CPUs 0–7 |
| TaskSet SFT, followed by ORPO | LLMOptimizer3 A40 GPU 1, CPUs 8–11 | Continues with the existing configuration |

FSBO services retain ports 18931–18934, authentication, the existing reverse SSH tunnel, and the validated numerical implementation. Their launch threads now use otherwise unassigned CPUs 45, 42, 43 and 44 respectively. Three services had been sharing CPU cores with VAE servers; all existing service threads were moved without interrupting their requests. The default service manager uses the new GPU and CPU allocations when a service needs restarting. Existing VAE/oracle services continue serving the same weights. The health endpoint's `cpu` field records the startup argument; `services.json` and `fsbo_cpu_affinity.json` record the actual current affinity.

Both FSBO dispatchers stopped, and all 48 active evaluators finished and saved their current batch before the services moved. Every prior observation field was compared using SHA-256 prefixes. Incomplete evaluators resumed through their original tasks and CPU slots. The new services passed separate checks with 5,100 observations, 5,000 candidates and 50 correlated draws, including selected-loss recovery, task isolation, snapshot restart and CPU checkpoint recovery. The four production numerical source files retain their previously validated hashes.

RGPE uses the original fitter, exact likelihood implementation, float64 data, normalization, priors, seeds and ten optimizer restarts. The 11 original in-progress fits were adopted without interruption. A fixed partition assigns 78 remaining tasks to machine 3; those tasks cannot be dispatched on machine 4. Queues use separate leases and status files, preserve memory reservations, and wait for all 800 source models before publishing global completion. The 604 model hashes recorded before rollout remain unchanged.

The source queue supports `--name`, `--settings` and `--assignments`. Launch commands and process IDs are saved in `rgpe_remote_launch.json` and `rgpe_machine3_launch.json`. The eight additional workers use the same shared Python environment and write standard upstream SMAC model pickles into the original bank under its existing bank lock. Separate task allocations and existing per-task locks prevent duplicate fits.

An experimental RGPE CUDA backend was implemented and tested separately. At the matrix level, float64 likelihood and gradient checks passed: approximately 0.16 seconds per GPU calculation versus 12 seconds on one CPU core after startup. A complete ten-restart fit on the same 5,100-observation source took 80.6 seconds versus 4,110.3 seconds for its saved CPU reference. However, its posterior failed the required agreement check: maximum absolute mean and covariance differences were 0.00478 and 0.12410. Consequently this backend is disabled in production. Its fit and report are confined to `rgpe_cuda_full_fit/`; no experimental model was written to the production bank.

Validation evidence is in `audit.json`, `fsbo_gpu0_validation/validation.json` through `fsbo_gpu3_validation/validation.json`, the two drain records, and the RGPE CUDA reports. Queue expansion provides 19/11 = 1.73 times as many RGPE fit slots; actual throughput depends on the CPUs and task lengths. FSBO performance samples are saved in `fsbo_performance_before.json` and the final audit. These are fitting-stage measurements, not an end-to-end campaign speedup guarantee.

To recheck the live rollout while its queues are running:

```bash
cd /root/data/cys/llmoptimizer_benchmark
python3 tools/audit_peptide_acceleration.py --require-production-progress
```

Operational rollout code is in `tools/accelerate_peptide_20261031.py`. Its drain actions refuse to overwrite an existing handoff record. Do not rerun a drain or redistribute services while these queues are active; the saved commands and records document this completed transition.

The final live audit passed after 32 production fits committed across all four GPUs. In this early sample, median FSBO queue time fell from 208.5 to 31.4 seconds, and median service time fell from 30.0 to 15.8 seconds. All 48 saved task prefixes and 604 existing RGPE bank hashes were verified again. Both FSBO queues had zero halted slots, and 19 distinct RGPE fits were live.
