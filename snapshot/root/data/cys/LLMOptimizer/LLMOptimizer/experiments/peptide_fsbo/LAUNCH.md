# Launch:20-round no-improvement retraining, maximum2 attempts

Recorded2026-11-19T08:57:25.034307+00:00. User requested the same10-peptide rollback experiment with
20-round training patience and maximum2 retraining attempts per peptide.

Controller and four evaluators are launched on LLMOptimizer3. See
pipeline_process.json and workers/slot*.json for current PID/start-tick identity.
Slots: GPUs1,2,4,6; CPU pairs34–41, two threads each,65% GPU allocator cap.
Other workloads and reservations were preserved.

The previous partial10-round-gate experiment was stopped and preserved at
/root/data/logs/peptide_target_loss_gate_test10_20261119;
STOPPED_FOR_POLICY_CHANGE.json records verified process shutdown. It already had
10-round updates and is not mixed into these fresh production histories.
Offline assets and exact initialization100 pools are reused.

Implemented in code/retrain_schedule.py and integrated in code/evaluate.py.
Counter increments only on acquisition rounds without feasible MIC improvement,
resets on improvement in any phase and on every attempt, cap2 across accepted,
rejected or interrupted attempts. Native search recentering is unchanged.
VAE2 epochs, accepted-update scratch FSBO source800+current100000 steps, batch50,
proposal/fresh caps4000 (max80 acquisition rounds) and loss gates unchanged.

Validation:6 scheduling behavior tests and11 target gate tests passed.
New actual-A16 integrated smoke passed10 checks with a forced20-count trigger,
real VAE update, source4+current scratch50-step FSBO rebuild and next acquisition.
96 validation-only oracle evaluations are excluded from production. Inherited
full gate/degraded-checkpoint evidence has explicit provenance. Production uses
source800/100000, not component budgets. All63 pinned inputs were verified on
launch host before workers started. launch_verification.json verifies live
controller and four evaluator identities.

The existing peptide-vae-training-10m loop was retargeted with the same lock,
every600seconds; no duplicate monitor. It writes monitor/latest.json/history.jsonl
and loop state, without automatic chat notifications. Completion controller
writes four-method MIC comparison. Do not claim improved MIC before measurement.

Previous completed parent had67 attempts:66 accepted and1 rollback,6–7/task.
Exact per-task counts and trigger rounds are in previous_retraining_counts.json.
New cap permits at most20 attempts across10 tasks,70.1% fewer attempts than67;
wall-time reduction also depends on search, validation, rejection and hardware.
