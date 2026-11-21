# Query campaign interruption and recovery

Earlier stops included a connection timeout and a PostgreSQL temporary-file EINTR. These escaped the old oracle wrapper and paused every worker; bounded same-proposal retries were previously installed.

This incident involved local execution-environment restart/replacement, not an established SQL-specific failure: all local campaign processes disappeared, PID 1 began after the last recorded campaign commits, and all six local databases required unclean-shutdown recovery. All fifteen remote databases survived. The platform trigger is unknown; the current cgroup counters cannot establish historical OOM, and kernel logs are inaccessible.

Recovery preserves 2,309 existing measurements and the agreed recipe/policy. Fixes verify process start times, recover stale inference-stage records while paused, retry identical durable LLM requests on transient transport errors, restore existing databases, and add minute-level diagnostics on both hosts. Sixteen regression tests passed.

Exactly one unscored SQL operation was audited before remeasurement: CEB_2A498 warm_fallback_measurement/2. Its original journal is backed up here. Its old database process was gone and the replacement server had no active SQL. The exact frozen read-only plan was remeasured; it hit the existing 50-second timeout and is censored. No runtime was inferred from the interruption.

All 21 evaluators were verified alive with correct CPU affinity, both hosts committed fresh SQL, both BOLT replicas and all ten VAEs were ready, and no active task errors remained. See recovery_verified.json and live_audit.json for the timestamped counts.

Hourly read-only monitoring was restored as a persistent local Python process. It does not survive a full execution-environment restart or send chat alerts. Independent minute-level remote diagnostics retain evidence if the local host disappears; they do not automatically replay SQL.
