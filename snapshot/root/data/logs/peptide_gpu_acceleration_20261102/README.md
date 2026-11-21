# Peptide FSBO batch-50 rerun and RGPE GPU continuation

The new FSBO campaign is `/root/data/logs/peptide_batch50_gpu_test200_20261102_v1/fsbo`.
It starts all 200 held-out tasks from the common 100-point initialization, using
50 recommendations per acquisition, 5,000 acquisition proposals, a shared
5,000-fresh-evaluation acquisition/recenter budget, seed 1, and the frozen VAE.
It uses the existing trained FSBO deep-kernel checkpoint and validated GPyTorch
CUDA services. The new campaign uses `peptide_local_bo.py` locality/recenter
settings, like the current RGPE campaign. Four local coordinators use CPUs
64–67 and CUDA FSBO services on A800 GPUs 0, 2 and 3. The completed original
FSBO campaign is preserved. Its archive also records batch 50; this rerun does
not rely on the claim that the original used batch one.

The existing RGPE campaign remains at
`/root/data/logs/peptide_frozen_local_recenter_test200_20261101_v1/rgpe`.
Its recipe, data, evaluation history, source models, optimizer, initialization,
BO budgets, seeds, candidate pool, batch size 50, joint posterior sampling and
weight dilution are preserved. Its original SMAC/SciPy target-hyperparameter
optimizer (all ten restarts plus the initial start) remains on CPU. New FP64
PyTorch CUDA services accelerate source posterior predictions/ranking and
joint leave-one-out target ranking. This is a hybrid implementation, not a
fully GPU optimizer or a substitution of a different GP model.

The fully GPU likelihood/optimizer experiment did not pass strict numerical
comparison (`validation.json`), and is not enabled in production. The hybrid
implementation passed `hybrid_validation.json`, covering three actual 5,100-row
source GPs against 1,150 real target locations, three leave-one-out folds at
1,700 real target observations, complete target-fit/weight calculation,
correlated batch-50 draws with a nonzero source weight, and snapshot restore.
Target optimizer hyperparameters and the tested complete weight pipeline were
identical. Actual source posterior maximum absolute errors were below 1.1e-10
for covariance and 7.4e-12 for means. One source had a maximum ranking-loss
difference of 2 out of 1,321,350 ordered pairs, due to floating point differences.
The declared ranking tolerance is at most 0.001% of ordered pairs; bitwise
identity of every future stochastic trajectory is not claimed.

Cached source ranking benchmarks took about 0.06–0.07 seconds versus 3.7–8.9
seconds on CPU. Three real leave-one-out folds took about 0.05 seconds versus
6 seconds on CPU. These are component measurements, not an end-to-end campaign
speedup. Cold loading and concurrent worker queueing still cost time. Each of
three CUDA services caches up to twelve verified fixed source GPs; CPU task
workers defer source loading instead of retaining their large model payloads.
Services are loopback only, token authenticated, single-request-at-a-time, with
an 8-GiB CUDA allocator cap each. Remote LLMOptimizer4 workers use SSH reverse
forwards on ports 19941–19943; local endpoints are 18941–18943. GPU 1 is excluded
from these new services for the existing RL training.

All 53 CPU RGPE tasks had no pending evaluation event at migration. Each durable
SQLite checkpoint was backed up, and its state was verified byte for byte before
resuming (`checkpoint_preservation.json`, `before_migration/`). Only incomplete
model computation was discarded. No committed oracle evaluations were removed.
There are 24 RGPE coordinators on LLMOptimizer2 and 29 on LLMOptimizer4. The
latest allocation plan is `dispatch/gpu_ranking_plan_20261102_v2.json` in the
RGPE campaign, using a six-GiB reservation floor with a 1.75× measured peak
memory growth guard, 32-GiB local/28-GiB remote memory headroom, and the existing
three/four CPU core reserves. CPU task claims and CPU slot checks are retained.

Run `python /root/data/logs/peptide_gpu_acceleration_20261102/code/live_audit.py`
for current worker identities, stage telemetry, GPU service health, batch sizes,
tracebacks, allocation overlap and latest proposal counts. Each task writes
`rgpe_gpu_stage.json` during CPU fitting/GPU ranking and `rgpe_gpu_fit.json`
after the full fit. A newly resumed task must finish its CPU target fit before
its first production CUDA ranking request. `gpu_config.json` and its verified
numerical-validation hashes gate both dispatchers and model workers.

Controllers and services were launched in detached sessions with logs in this
directory. They are not registered for automatic restart after a host reboot.
