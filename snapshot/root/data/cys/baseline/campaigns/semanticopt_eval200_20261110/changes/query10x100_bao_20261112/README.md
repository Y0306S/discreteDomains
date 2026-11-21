SemanticOPT query configuration amendment — 2026-11-12

Query uses batch size 10 and 100 rounds, totaling 1,000 new proposals per task.
Every query batch prompt supplies all 49 canonical Bao initialization observations,
including measured runtimes and timeout/censoring flags, separately from recent
optimization feedback. Bao observations do not consume the 1,000-proposal budget.
Original committed proposals count toward the budget and are preserved verbatim.
A resumed partial round requests only its remaining slots before full batches of 10.

Peptide retains batches of 50 for 20 rounds (1,000 new proposals). Its batch and
budget policy hashes are unchanged. The frozen original recipe is unchanged.
Other methods and inference/training services are outside this amendment.
The query stage remains queued after peptide. Smaller batches do not reduce the
number of SQL evaluations; an ETA remains unavailable until query throughput is measured.

Validation: 26 unit tests passed (23 existing regression tests and three query
policy/resume/telemetry tests). The mocked query resume test starts at 15 committed
proposals, requests 5 followed by 98 batches of 10, preserves the original records
and Bao initializations, and finishes at 1,000 proposals / 100 rounds without repeats.
The prompt audit covers all 200 test queries and 9,800 Bao observations, checks
first and subsequent batch prompts, validates the constraints, and rejects 1,000
invalid/duplicate cases. Maximum first prompt: 11,297 tokens; subsequent prompt
with adaptive recent feedback: 14,425; input limit: 15,360. Largest estimated
10-plan output: 1,764 tokens. No held-out SQL evaluations were executed by this audit.

The query live batch preflight is invalidated by the new query policy hash and
will run with 10 items on the training smoke task when the queue loads the query
adapter. Its state lives in a policy-specific directory, preserving the old probe.
The existing 50-item probe is historical evidence and does not validate this amendment.

Evidence: prompt_constraint_audit.json, query_checkpoint_audit.json,
publication.json, completion.json; before/ contains configuration and query DB backups.
