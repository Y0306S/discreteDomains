# Peptide: BOLT warm50, then ORPO + FSBO, then ORPO + RL + FSBO

Prepared only. No queue, GPU replica, CPU optimization worker, generation, or APEX call has been started. Existing query jobs continue under their current controllers. GPU IDs, ports and CPU assignments are deliberately unset in the service templates.

| Setting | Configuration |
|---|---|
| Evaluation scope | The fixed 200 held-out peptide tasks; train800 remains disjoint |
| Seed | 1 |
| Warm-start model | Completed full SFT, `peptide_warmstart_full800_20261111/model/checkpoint-1500`, 3 epochs |
| Initializations | First **50 distinct feasible model-generated peptides per task**, measured with APEX bacteria_0 |
| First full pass | **ORPO-only + FSBO**, all 200 tasks |
| Second full pass | **ORPO + RL + FSBO**, all 200 tasks, after the entire first pass finishes |
| Initial pool for both passes | Exactly the same frozen 50 sequences, measured MIC values and order |
| Optimization budget | **80 additional attempts per task per pass**, following the earlier ML evaluation design |
| Optimization batch size | **One selected candidate per attempt**; warm50 is an initialization count, not a JSON batch of50 |
| LLM generation | Temperature1.0, top_p0.9, top_k50, max128 new tokens; one sequence JSON per response |
| Warm generation bound | Up to500 samples/task; no score-based cherry-picking or random replacement; underfilled tasks remain incomplete |
| LLM optimization retries | Up to5 invalid/duplicate responses, then FSBO fallback; transport failure pauses the task |
| Arm selection | PUCB, arm-best reward, c2=1.4, independent exploration weights: LLM0.9→0.3, FSBO0.3→0.9 on the original100-attempt schedule |
| Initial optimization phase | First8 post-initialization attempts select LLM; attempts9+ use PUCB |
| Cooldown | Attempts10–20 select FSBO after a preceding LLM selection, including LLM→FSBO fallbacks |
| FSBO source | Existing verified train800 FSBO checkpoint; target negative-MIC minmax scaling |
| FSBO target adaptation | At most1,000 updates, patience16, min_delta0.0001, learning rate0.001; matches the effective existing peptide target-fit amendment |
| FSBO acquisition | 5,000 sparse local latent candidates; native constrained Thompson sampling selects one |
| Frozen representation | Original author VAE, canonical256-dimensional posterior means; no VAE or LLM training |
| Local search | Region length0.8, bounds center±3.2, perturbation probability20/256; region reset at current best for each FSBO selection |
| Constraint model | Native target GP, refit for20 epochs on all current measurements each FSBO selection |
| Recenter | Disabled for this explicitly named single-candidate hybrid variant |
| Resource plan | One CPU thread/task; separate shared GPU services assigned at future launch |

The two model passes are separate evaluations. ORPO+RL does **not** inherit ORPO search history or observations beyond the common warm50 pool. Each target begins its second pass from that same pool with a fresh selector and target optimization state.

## Peptide constraints and objective

All accepted sequences must use uppercase `ACDEFGHIKLMNPQRSTVWY`, have length1–30, have at most one cysteine, and satisfy `1 - Levenshtein(candidate, reference)/len(reference) >= 0.75`. The unchanged reference and previously measured sequences are rejected. Real APEX bacteria_0 MIC is minimized; FSBO maximizes negative MIC. Generation text is never used as a measured score.

After an LLM fallback, FSBO may decode up to5 constrained posterior selections to find a feasible novel sequence. If all5 fail, the attempt records `NO_VALID_PROPOSAL`, consumes one optimization attempt, and retains the incumbent without a fabricated MIC. Infrastructure failure does not consume an attempt. An interrupted APEX call with no durably saved result is marked unresolved and stops that task for an audit; committed measurements replay without a new call.

ORPO-only uses its exact training prompt: measured good5, bad5 and Recent50. ORPO+RL uses its training-compatible similar-task retrieval in optimization attempts1–10 and the normal prompt thereafter. Retrieval uses only train800 source cards and source-GP prediction rank agreement on already measured target observations. The shortlist has16 references, with up to3 source tasks and a Spearman threshold0.5; weak agreement is reported explicitly. There are no Good/Bad Region summaries.

The generator is a full SFT model. Each optimization adapter loads on its original clean Qwen3.5-9B base, **not** on the generator's altered weights. Every checkpoint uses the LLaMA-Factory `qwen3_5_nothink` answer boundary. Prompt caches are bounded to1.5GiB/replica, configured for prompts of at least128 tokens, and isolated by checkpoint and exact prompt. An unchanged warm prompt can reuse prefill; changing observation/retry context creates a new cache key.

## Accounting and comparison

The warm pool requires10,000 accepted measurements. Each optimization pass has16,000 attempts, so the setup permits **up to42,000 measurements total** (fewer when optimization attempts have no valid proposal). Warm measurements are made once and imported into both passes.

The 80-attempt cutoff and PUCB100-attempt schedule carry over the earlier ML design. For adaptive attempts, f = clip((attempt - 9) / 91, 0, 1), omega_LLM = 0.9 - 0.6*f and omega_FSBO = 0.3 + 0.6*f. These are independent exploration multipliers, used as Q_arm + 1.4 * omega_arm * sqrt(total_attempts / (1 + arm_attempts)); they are not normalized selection probabilities. At attempt9 the weights are0.9/0.3; at the attempt80 cutoff they are approximately0.4319/0.7681; the0.3/0.9 endpoints occur at attempt100. The50 warm initializations are external to that schedule and do not update the selector. The first8 *post-initialization* LLM selections are explicit here; historical ML included three presets within its first8 trials. The peptide FSBO acquisition remains latent constrained Thompson sampling, rather than ML's discrete hyperparameter acquisition.

This is a new method variant, `peptide-our-warm50-fsbo-v1`. Historical peptide FSBO used100 mutation-pool initializations, batch50 and5,000 proposals with different local/recenter behavior. Comparisons with those older curves must identify both the initialization and budget differences.

## Files and offline verification

- `recipe.json`: full pinned configuration, model identities, task manifest and dependency hashes.
- `stages/{warm_start,orpo,orpo_rl}.json`: stage order and prerequisites; all remain unqueued.
- `service_templates/`: unassigned replica configurations; no services started.
- `code/method.py`: filtering, retry/fallback, selection, durable journals and stage checks.
- `code/backends.py`: existing Qwen, APEX, frozen-VAE and FSBO integrations.
- `code/run.py`: prepare/validate/config inspection, pool freezing and explicit single-task execution. There is no queue command.
- `evidence/config_validation.json`: pinned dependency and split checks.
- `evidence/offline_tests.log`: behavior/recovery tests with synthetic callbacks, no real oracle calls.

Offline tests cover50 unique warm initializations, malformed/infeasible/duplicate rejection, underfilled pools, five-retry FSBO fallback, exact committed-result replay, unresolved oracle interruption, deterministic resumed selection, selector cooldown, stage identity/budgets and the later normal RL prompt. They do not establish live generation acceptance or throughput.

Live end-to-end preflight and warm-checkpoint shard rehash remain launch-time checks. No runtime ETA is claimed before a real operational smoke test. Full warm weight hashes from the completed training verification are pinned now, with file size/mtime checked; both optimization adapter files are rehashed during validation.

Inspect or revalidate without starting a task:

```bash
PY=/root/data/cys/envs/bolt_clean_venv/bin/python
CAMPAIGN=/root/data/cys/baseline/campaigns/peptide_our_fsbo_test200_20261114
$PY "$CAMPAIGN/code/run.py" show-config
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  $PY "$CAMPAIGN/code/run.py" validate
$PY "$CAMPAIGN/code/test_method.py"
```

Future manual operation uses a durable Qwen broker for exactly the selected model, a dedicated frozen-VAE service and an explicitly chosen CPU. The order is warm generation for all200 -> `freeze-pool` -> ORPO for all200 -> ORPO+RL for all200. No completion monitor has been armed to start this automatically.
