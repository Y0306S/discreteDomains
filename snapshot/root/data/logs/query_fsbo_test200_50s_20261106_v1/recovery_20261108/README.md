# Query FSBO recovery, 8 November 2026

CEB_7A102 stopped while waiting for VAE decoding, before a new decoded batch or SQL call was committed. The task checkpoint contains 220 completed proposals and no pending database proposal. The inspection saved all artifact hashes before changing the readable task status to paused.

The campaign-local wrapper uses the unchanged, checksum-verified evaluator and recipe. It increases the VAE HTTP wait from 300 to 1800 seconds. The SQL plan cap remains 50 seconds. It isolates recorded failed tasks and continues to other tasks; failures without a durable task status still stop the worker. It never automatically replays an unknown SQL execution. Two queue failure-injection tests passed.

Seventeen stopped workers were restarted, preserving the three healthy evaluators. Placements remain 16 workers on LLMOptimizer4 and four on LLMOptimizer2, sharing the same ten VAE services. Database locks, PostgreSQL health, process identities and memory admission were checked before restart. Current process records are in the campaign's current_workers.json; before/after records and logs are in this directory.

The first w19 launch runs one bounded batch of CEB_7A102. Its previous results must pass a hash comparison before w19 rejoins the full queue. The remaining restarted workers are already allowed to process their queues.

ML RL continues independently in its existing supervisor. Its phase-one final checkpoint contains 38 optimizer updates; phase two is finishing CatBoost rollouts. No ML recipe or active process was changed by this recovery.

Live recovery evidence: CEB_7A102 committed all ten new proposal records (220 to 230), including a new measured SQL result. All pre-recovery artifact hashes match. The one-shot finish_recovery.py process waits for the bounded batch to publish its complete checkpoint, validates its status and prior results, then returns w19 to the full queue. handoff_status.json and bounded_check_validation.json record that final handoff.

Second incident: w03 received PostgreSQL SQLSTATE XX000 (`Interrupted system call`) on proposal 109 of CEB_8A220. The exact plan, pending marker, failed DB attempt, absent result, worker identity, and zero-active-session boundary were checked before authorizing one replay. PostgreSQL returned the ordinary 50-second censored timeout. Both attempts are retained, the proposal was charged once, all 109 pre-existing result hashes still match, and the task checkpoint now contains 120 proposals with no unresolved call. See CEB_8A220_recovery.json and CEB_8A220_recovery_verified.json. w03 was returned to the full queue after the bounded replay batch.
