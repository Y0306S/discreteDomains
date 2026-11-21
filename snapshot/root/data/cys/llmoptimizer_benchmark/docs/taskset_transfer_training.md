# TaskSet Train838: FSBO and RGPE

The `taskset-transfer-5m` loop checks the eleven TaskSet backfills every 300
seconds and continues until explicitly stopped. Its detached Python process
writes reports; it does not provide automatic chat notifications.

The production directory is
`/root/data/logs/taskset_transfer_train838_20261002`.
Loop state and reports are in `/root/.codex/loops/taskset-transfer-5m`.

## Sources and Training

Only the pinned 838-task training split is permitted. The 210 test tasks are
excluded. Sources are real TPE histories with 100 evaluated configurations per
task, using 1,000 training steps, validation every 1,000 steps, one replica and
`min_normalized_valid_outer`. Loose histories and verified completed-run archives
are supported. No BO evaluations are rerun to build these training inputs.

Failed evaluations remain in the original histories but their failure-penalty
values are excluded from training labels. Each exported task records original
row indices, excluded row indices and the original history SHA-256. Labels keep
their original direction: lower normalized validation loss is better; FSBO uses
its negative for the maximizing backend.

FSBO uses the existing TaskSet conditional encoder, with categorical one-hot
coordinates and canonical inactive parameters. Its input has 43 dimensions.
The feature network is 128/128 ReLU with an RBF GP, Adam at 0.001, batches of 50
observations and the paper's label augmentation. Production performs 100,000
updates, with validation every 100 updates. A deterministic, task-family
stratified split of train838 gives 670 meta-training and 168 validation tasks.
The selected checkpoint minimizes held-out predictive negative log likelihood.
This is the established paper-based benchmark recipe, not a claim that the
upstream HPO-B variant's 100,000 full-dataset epochs are identical.

RGPE fits one GP for each of all 838 source tasks, using the upstream SMAC/RGPE
mixed continuous/categorical kernel and native conditional ConfigSpace encoding.
Source posteriors and serialized-model reloads are checked. Target-task ranking
weights are computed during later evaluation, not by this source-training job.

OptFormer is not part of this automatic launch.

## CPU Handoff

The final backfill is `word_rnn_language_model_family_seed94`, currently owned by
`LLMOptimizer4.w06` on CPUs 28-31. The handoff pauses only its queue supervisor.
Its existing evaluator and TensorFlow subprocess continue running on those CPUs.
This prevents the supervisor from claiming a new test job after the backfill.

When the evaluator exits, the watcher observes its actual exit status while the
paused supervisor still owns it. Only exit code zero and a 100-observation history
permit handoff. The completed supervisor is then retired; a controller acquires
the existing worker, job and CPU lock files, verifies/archives the completed run,
exports the complete training corpus and starts:

| Method | Host | CPUs | Device |
| --- | --- | --- | --- |
| FSBO | LLMOptimizer4 | 28,29 | CPU |
| RGPE | LLMOptimizer4 | 30,31 | CPU |

The controller and training subprocesses retain the existing CPU locks. The
shared campaign plan is not rewritten while other workers may be archiving it;
`cpu_reservation.json` records the allocation override. The other 15 TaskSet
workers continue their existing jobs. Failed checks or uncertain process identity
hold the launch for inspection rather than starting another training copy.

## Verification and Operations

Component checks used two real completed training histories, 200 observations:
FSBO completed 20 updates, reloaded, resumed to 40 and reloaded again; RGPE fitted
two source GPs and reused them on resume. Seven handoff tests covered continued
evaluator execution while its supervisor is paused, observed exit status, PID
identity, launch readiness, disjoint core allocation and occupied-core refusal.
Three source tests covered test-set exclusion, incomplete-history refusal and
failed-label filtering with provenance checks.

`launch_ready.json` pins the prepared scripts and records successful preflight.
The full production corpus is exported only after all source histories complete.
`handoff.json`, `controller_state.json`, and each method's `status.json` record
the subsequent phases. The watcher continues after training completes.

Stop the recurrence with:

```bash
python /root/data/cys/llmoptimizer_benchmark/examples/benchmark/domains/monitor_taskset_transfer.py stop
```

Stopping monitoring during backfill drain resumes its original queue supervisor.
Stopping monitoring after training starts does not stop the training workloads.
Both trainers accept `--resume`, validate source/checkpoint identity, and preserve
previous work; failed launches are not automatically replayed.
