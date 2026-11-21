# Same-ten rollback experiment launched

Recorded 2026-11-19T16:34:24.158147+08:00.

Host: **LLMOptimizer3**. Four parallel workers on GPUs1,2,4,6, each with two
CPU cores (34–41) and65% GPU allocator limit. Existing services remain active.
Task order prioritizes pep_999; the ten-task cohort is unchanged:
pep_117, pep_219, pep_221, pep_575, pep_673, pep_921, pep_925, pep_967, pep_980, pep_999.

Each task starts fresh from the same offline VAE/FSBO and exact initial100
sequence/score pool, seed1, batch50, proposal/fresh caps4000. The original
source800 scratch-FSBO retraining budget100000 is retained for accepted VAE
updates. No completed prior optimization history or validation measurements
are reused as production observations.

Validation passed on the actual A16 environment: real two-step VAE update,
nonworsening target loss, four-source+current scratch FSBO component rebuild,
post-update decoding/scoring, coordinate consistency, unique oracle accounting,
and real degraded-checkpoint rejection with exact tensor restoration.
The component test used99 extra oracle evaluations, saved only under
validation_fullscale, separate from production budgets. It used50 FSBO steps
on four sources to test integration; production uses800/100000 unchanged.

The controller advances queued tasks and automatically writes the four-method
MIC comparison after all ten finish. Comparators are the original frozen FSBO,
previous online VAE/FSBO and SemanticOPT at the same4000 proposal cap.
The comparison uses the same previous curves, copied into shared
comparison_inputs to make the remote completion report self-contained.
The first controller attempt detected the missing remote report path before
launching any evaluator; that attempt is preserved in validation. The path
was corrected and the manifest reverified before the successful launch.

Monitoring reuses loop **peptide-vae-training-10m**, every600seconds, with
exclusive flock and PID/start-tick verification. It checks controller/workers,
progress, completion, rollback decisions and accepted-loss invariants.
The scheduler writes files and does not send automatic chat notifications.

Live evidence:
- pipeline_process.json and pipeline_status.json: controller identity/status.
- workers/slot0.json through slot3.json: current assignments and worker identity.
- evaluation/adapted_online/<task>/status.json: committed task progress.
- evaluation/adapted_online/<task>/vae_updates: pre-update snapshots and decisions.
- monitor/latest.json: latest scheduled check; monitor/history.jsonl: history.
- /root/.codex/loops/peptide-vae-training-10m/state.json: monitoring identity/cadence.
- validation/live_gpu.json and validation/saved_pep999_gate.json: validation evidence.

On completion: RESULTS.md, comparison.png/pdf/svg, task_results.csv,
paired_comparison.csv, capped_convergence.csv and summary.json.
Runtime comparisons are descriptive: this A16/shared-hardware allocation differs
from the previous A800 run. Improvement in MIC is an outcome to measure.
