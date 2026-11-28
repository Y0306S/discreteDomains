# Same-ten FSBO feasible-search rerun

User authorized launching the new ten-task rerun on 2026-11-22.
Host: LLMOptimizer3. Four worker slots: GPUs 4, 6, 10, 13;
CPU pairs 34–35, 36–37, 38–39, 40–41. Two threads per worker,
40% per-process GPU allocator limit. Existing workloads are preserved.

Tasks: pep_117, pep_219, pep_221, pep_575, pep_673, pep_921, pep_925,
pep_967, pep_980, pep_999. Same retained offline checkpoints, initial100,
seed1, batch50, 5000 candidates, acquisition proposal cap4000 and shared fresh
cap4000. Online update patience20 rounds, maximum2 attempts; accepted updates
rebuild FSBO from scratch with source800 plus current task for100000 steps.
The updated recenter, feasibility-based length, candidate selection, and disabled
restart policies are recorded in config.json and README.md.

Detached controller: code/pipeline.py. PID/start-tick identity is recorded in
pipeline_process.json. Workers write individual logs and checkpoint/pending files.
The controller queues all ten tasks, uses an exclusive campaign lock, can adopt
existing evaluators when resumed, and reports failures without silently retrying.

During execution the controller refreshes RESULTS.md, summary.json and
 task_results.csv every60 seconds. Each task is compared to the prior rollback
run at the same number of acquisition proposals. Final results include mean MIC,
wins/losses/ties, per-task feasibility and runtime, and convergence figures.
Raw task output is evaluation/adapted_online/<task>/. Results use fresh histories;
no validation observations are included in production budgets.

The search passed32 regression tests and a real5000-candidate GPU check.
Reporting was checked with identical completed-run fixtures yielding ten ties;
those fixtures are not new optimization results. Full sequential efficacy remains
to be measured by this run. Four A16 workers replace the A800 validation host;
concurrent workloads affect runtime, so runtime differences are descriptive.
