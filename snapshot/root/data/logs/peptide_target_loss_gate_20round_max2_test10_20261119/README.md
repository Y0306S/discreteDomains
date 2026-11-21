# Peptide target-loss rollback: 20-round patience, maximum two attempts

Same ten held-out peptides; fresh optimization histories, exact initial100, seed1,
batch50, acquisition proposal cap4000 and fresh-evaluation cap4000 (at most80
acquisition rounds). Offline VAE/FSBO assets are reused from the earlier campaign.

Online retraining has an independent persistent counter. Each acquisition round
without a strictly better feasible incumbent increments it; a feasible incumbent
improvement in any phase resets it. At20 such rounds, attempt a two-epoch VAE
update and reset the counter. At most TWO attempts per peptide across the entire
run; accepted, rejected and interrupted attempts all consume the limit. With no
improvement, attempts occur after acquisition rounds20 and40, provided the run
has not exhausted its evaluation budget. Improvements can delay or prevent them.
20 rounds are1000 acquisition proposals, not necessarily1000 distinct oracle calls.
The native search recentering policy is unchanged, including its10-round setting;
recenter batches do not increment the new training counter.

The existing fixed-target and source retention gates remain active. A target
loss deterioration rejects the VAE and skips FSBO retraining. Each accepted VAE
update rebuilds FSBO FROM SCRATCH with source800 plus current observed task,
100000 steps. This is not20 training epochs or a20-round total run.

A consumed-attempt reservation is atomically checkpointed with OLD model states
before training starts. If interrupted before publication, restart retains those
old states, records an interrupted rollback and consumes that attempt. A successful
publication checkpoints retained VAE, FSBO and re-encoded coordinates together.
The completed parent results and the partial earlier10-round-gate run are preserved.

Previous completed parent:67 attempts,66 accepted,1 rejected,6–7 attempts/task.
This run permits at most20 attempts across all10 tasks, about70% fewer attempts
than67. Overall runtime cannot be inferred from that reduction alone.

Execution uses LLMOptimizer3 four slots on GPUs1,2,4,6, CPU pairs34–41.
The existing peptide-vae-training-10m monitor is retargeted, not duplicated.
Read pipeline_status.json, workers/slot*.json, evaluation/adapted_online/*/status.json,
and monitor/latest.json for live evidence. Gate/GPU component evidence is inherited
with explicit provenance; scheduling tests and new integrated smoke evidence are
recorded separately under validation. Validation-only observations are excluded
from production results. Component FSBO smoke uses4 sources/50 steps;
production uses800 sources/100000 steps.

The controller writes the four-method MIC comparison after all ten tasks finish.
MIC improvement is unproven until measured; previous A800 versus current A16
runtimes do not isolate the effect of this policy.
