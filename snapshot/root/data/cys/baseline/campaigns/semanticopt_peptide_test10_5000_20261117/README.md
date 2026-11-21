# SemanticOPT: continue the same ten peptide tasks to 5,000 evaluations

Authorized on November 17, 2026. This isolated campaign continues the completed
1,000-evaluation SemanticOPT trajectories for the same ten tasks selected for
the VAE/FSBO adaptation pilot:

`pep_117, pep_219, pep_221, pep_575, pep_673, pep_921, pep_925, pep_967, pep_980, pep_999`.

Each task retains its 100 measured initialization peptides and all 1,000
completed optimization observations, proposal history, inference RNG, and
completed batch state. The only optimizer policy amendment increases the cap
from 20 to 100 batches of 50, ending at **5,000 distinct scored proposals**.
The initialization observations are excluded from this cap. This requires
**4,000 additional evaluations per task, 40,000 in total**. No original campaign
file or checkpoint is modified.

The frozen source recipe, trained SemanticOPT peptide LoRA adapter, base model,
temperature 0.7, top-p 0.95, top-k 50, seed 42, prompt window, reference similarity
constraint, duplicate rejection, four-request batch limit, and labeled
constrained random fallback policy are retained. Later prompts truthfully show
the new 5,000-proposal budget; the earlier 1,000-budget decisions are historical
observations rather than a rerun with hindsight. This is the existing SemanticOPT
configuration, including its fallback component. Exports separate fallback and
LLM proposals.

## Resources and execution

The finite job runs on **LLMOptimizer3**. Two inference replicas each shard the
same BF16 model across two otherwise free A16 GPUs: physical pairs **0,3** and
**14,15**. The replicas use CPU cores **6 and 7**. Ten CPU APEX workers use cores
**8–17**, one task per worker. The broker shares core 6. This uses 12 distinct CPU
cores in total. Existing inference services are left running.

Each replica reserves model-loading headroom with a 9/12 GiB device placement
budget and a 14 GiB per-device allocator cap. No quantization or model change is
introduced. Inference replicas, broker, and workers are stopped when the finite
campaign completes or the controller detects a failure.

The previous inference hardware differed. Saved RNG and model identity are
retained, but bitwise continuation across GPU architectures is not claimed.
Existing baseline FSBO seed 1 differs from SemanticOPT seed 42. Cross-method
runtime comparisons therefore require these device/seed qualifications.

## Verification and recovery

`evidence/import_audit.json` records the hash of each source checkpoint and of
the original 1,000 observation rows. All source SQLite state was copied exactly
before an explicit, recorded budget-policy amendment in the copy. No task had
an unresolved pending oracle call. `evidence/fsbo_initialization_agreement.json`
checks the 100 initial sequences and scores against the FSBO pilot for all ten
tasks. Original external report files are not execution dependencies; the
actual source initialization JSON remains pinned.

The 24 existing batch, oracle transaction, fallback, and budget tests passed.
Remote APEX preflight reproduced three saved SemanticOPT CPU scores exactly.
Agreement with the imported initialization scores was within the original
relative tolerance of 0.001 (maximum observed relative error 0.0003971). Nine
diagnostic oracle scores were computed during preflight and are excluded from
optimization results. See `evidence/preflight.json`.

Code, service settings, original dependencies, and policy files are pinned in
`identity.json`. Saved inference responses and SQLite oracle transactions retain
the native evaluator's restart behavior. An unresolved in-flight oracle call
requires inspection rather than silently rescoring it. To restart a stopped
controller, first inspect `error.json`, process liveness, and pending states,
then run on LLMOptimizer3:

```
/usr/bin/python3 -u control/controller.py
```

An exclusive controller lock and per-task locks prevent duplicate execution.
The controller verifies that assigned GPUs are free before loading replicas.

## Progress and results

`status.json` tracks service startup, committed evaluations per task, and total
new evaluations. `processes.json` and `controller_process.json` identify the
processes. Logs are under `logs/`; per-task status and SQLite checkpoints are
under `peptide/runs/`.

After all ten tasks reach 5,000, the controller runs `control/report.py`, checks
that the original observations are unchanged and all new sequences are unique
and feasible, and produces `RESULTS.md`, `summary.json`, `paired_results.csv`,
`convergence.csv`, and `comparison.{png,pdf,svg}`. The report compares measured
MIC at 1,000 and 5,000 and reports additional fallback counts. It also exports
`fsbo_comparison.csv` for available completed runs of the three FSBO pilot arms,
using each task's actual common fresh-evaluation budget. Missing FSBO results
are omitted explicitly; rerunning the report incorporates later completions.

Lower predicted MIC is better. These are computational oracle results, ten
tasks, and one optimizer seed; experimental antimicrobial activity is not
measured by this continuation.
