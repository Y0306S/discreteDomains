RGPE acquisition merge validation and rollout — 2 November 2026

Implemented the exact independent-GP merge with weighted means and squared
weights on full covariances. The previous implementation already merged and
sampled once. The new implementation adds each source covariance in blocks
directly to the merged covariance, then factors that covariance once for all
batch Thompson samples. It avoids full source candidate covariance arrays.

Validation passed before deployment:

| Check | Scale | Result |
| --- | --- | --- |
| Math, source streaming, correlations, numerical handling and sampling | 14 tests | Passed |
| Existing frozen local wrapper, recovery and budget accounting | 17 tests | Passed |
| Amendment validation, CPU/memory admission and task claims | 9 tests | Passed |
| All actual source models, independently checked against original predictions | 800 contributors, 32 candidates | Passed |
| Dense real-source ensemble | 100 contributors, 512 candidates | Passed |
| Production candidate count and batch | One real target GP, 5,000 candidates, batch 50 | Passed |
| Framed model subprocess, compact snapshots and restart | Two worker restarts, 5,000 candidates, batch 50 | Identical seeded draws |
| Actual frozen wrapper launcher and existing pending proposal | Production task pep_235, stop before oracle | State and pending bytes unchanged |

The old/new production-size comparison used the same stored fitted target,
candidate set and seed in separate processes. Peak RSS fell from 0.8640 GiB
to 0.5864 GiB, a **32.1% reduction**. Acquisition took 4.33 versus 4.03 seconds
in the controlled comparison. This single timing comparison does not establish
an overall BO speedup. The maximum sampled-score difference was about 1.2e-9.
The 800-source check used about 1.03 GiB peak RSS and matched the independent
covariance reference to within 6.7e-16. Diagnostic ensemble weights were used
only inside tests. Tests made no new oracle calls and ran no LOLBO optimization.

`validation.json` binds these checks to exact code hashes. It was finalized
before rollout and is referenced by the campaign acquisition amendment.

Both original dispatchers and all original model workers drained gracefully;
none required forced termination. Backups of 28 task SQLite checkpoints and the
prior dispatch plan/status/launch manifests were saved under the campaign's
`dispatch/rollout_20261102_merged/` directory. The original optimization recipe,
checkpoint identities, source bank, stored scores and pending request nonces
were preserved. The new dispatcher plan selects the versioned launcher through
the validated amendment.

The first live audit found **42 new RGPE workers: 20 on LLMOptimizer2 and 22 on
LLMOptimizer4**, all with the expected code marker. Both dispatchers are running,
with no task duplicates, CPU overlaps, legacy RGPE models or failure pause.
The original quota and memory guards remain in place, with concurrency ceilings
of 20 and 26. Actual admission follows available memory and CPU capacity.
BORA has reached its planned retained count of 56 coordinators (26/30), with
all eight healthy LLM replicas busy at the audit.

The VAE remains frozen. Source prediction work, target fitting, source ranking,
joint target LOOCV, dilution and learned weights are unchanged. In particular,
this acquisition change does not replace the per-observation LOOCV ranking
models or make training cost equivalent to one GP. Dense candidate covariance
and component training-data memory still grow with problem size.

Key artifacts:

- `validation.json`: immutable pre-rollout validation and code hashes.
- `rollout.json`: checkpoint backups, both dry-runs, plan identity and launches.
- `live_rollout.json`: live worker version and handover verification.
- `unit.log`, `wrapper.log`, `rollout_tests.log`: all 40 test results.
- `bank800.json`, `dense100.json`, `old5000.json`, `new5000.json`, `wire.json`,
  `launcher.json`: detailed scale and recovery checks.

Campaign:
`/root/data/logs/peptide_frozen_local_recenter_test200_20261101_v1/rgpe`.
Current progress can still be checked using
`/root/reports/bora_rgpe_rebalance_20261101/status.py`, which now recognizes
both legacy and versioned worker entry points.
