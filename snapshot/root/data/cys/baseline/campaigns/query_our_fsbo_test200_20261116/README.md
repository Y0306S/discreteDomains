# Query BOLT → ORPO + FSBO, batch10 revision

The active implementation is `batch_code/`, with `batch_recipe.json` and `batch_policy.json`. Original `code/`, `recipe.json`, journals, preflight and pins remain historical evidence. Peptide is not queued; ORPO+RL is not enabled.

## Evaluation schedule

200 held-out CEB SQL tasks; 80 total rounds × 10 measured unique plans = 800 evaluations per task. Rounds 1–3 supply 30 BOLT warm starts. Rounds 4–8 use ORPO with training-task retrieval. Rounds 9–20 use the original ML PUCB selector, with no consecutive selected LLM in rounds 10–20. Rounds 21–80 use normal PUCB, cooldown disabled. No ML region summaries. Global selector index is round−9, denominator91, p1=.7, q=arm_best, c2=1.4.

Freeze all ten proposals before evaluation; update feedback only after all ten are measured. Strict plan validation checks exact SQL aliases, N−1 component merges, join operators1/2/3, and no implicit codec repair. Deduplicate against all measurements in this comparison and accepted current proposals. Timeout50s; censored measurements are excluded from exact GP targets and incumbents. Full SQL, aliases and constraints are present in the unchanged BOLT SFT prompt.

## Warm starts and reuse

Reuse each query’s first30 valid saved BOLT measurements in generation order, never fastest30. The audit retains265 and archives89 from ten started queries, with zero invalid saved plans. Archive later measurements outside this comparison. Preserve original temperature1.0 provenance on reused samples.

New BOLT calls return one plan each, temperature0.9, fresh RNG with an unchanged cached prompt prefix. Maximum50 generation calls per query, counting prior legacy calls for partial pools. Explicit legacy exception: already measured first30 are retained even when their old sampling required more than50 calls. After exhaustion, fill only missing slots by randomly sampling valid plans from the previous BO run for that exact SQL, without using old runtime ranks, and measure them fresh under50s. Label `warm_random_previous_BO`.

## Optimization and fallbacks

ORPO returns one plan per call, at most20 calls per round to collect10 unique valid plans. Keep accepted plans across calls; change the prompt to include accepted unmeasured plans and the last five rejected texts/reasons. These never become measured runtime history. ORPO weights remain loaded; prompt caching is disabled.

FSBO makes a joint Thompson batch of10, or only the missing slots when topping up ORPO. Apply the original native codec canonicalization to internal raw VAE tokens, then strictly validate the final executed plan. Keep raw-token/canonical-plan provenance for accepted decodes. Frozen64D author VAE, frozen source weights reset before each target fit, one fit per round using all completed uncensored measurements. Up to five acquisition/decode batches of the remaining count, 5000 latent candidates per acquisition, preserving valid decoded plans. Native local TuRBO region resets at the incumbent for each FSBO selection; the historical standalone FSBO carried region state. No VAE training, recentering, or end-to-end updates.

After FSBO underfill, use fresh constrained random plans for missing slots only, at most1000 draws per round. Label `random` and exclude its reward from PUCB attribution. If still underfilled, pause without starting SQL for that batch. Each actual LLM/FSBO arm gets one reward per round: its best uncensored measured runtime; mixed batches credit their actual proposing arms. All-censored arms get no exact success. Known terminal infrastructure failures have bounded retries of the same proposal. Ambiguous interrupted SQL requires an audit before replay.

Retrieval follows the training method: table Jaccard / alias-count shortlist16, source-GP Spearman agreement on the current causal observations, top3 sources, agreement≥.5 preferred and weak evidence explicitly flagged. Source cards cover train800; verified GP exports exist for788 of those tasks, matching the original training bank. No future held-out measurements enter retrieval. Complete prompts are never truncated. ORPO input allowance32768 tokens; long-context capacity is separately smoke tested.

## Resources and operation

The expanded resource layout uses 12 GPU service slots: two LLM replicas on GPUs 0 and 3, and ten frozen VAE replicas spread 2/3/3/2 across GPUs 0–3. Both LLM replicas serve BOLT during warm30, then automatically switch to ORPO after every warm pool completes and is frozen. BOLT retains prompt-prefix caching; ORPO caching remains disabled. There are 21 CPU evaluators with private PostgreSQL instances: six on LLMOptimizer2 (24 evaluation cores plus 13 service cores within its 40-core quota), and fifteen on LLMOptimizer4 (60 evaluation cores within its 64-core quota). Each evaluator has one optimizer core plus three SQL cores. Seven cores across both hosts cover controller, SSH, filesystem and system overhead. Existing databases are reused without copying. Remote PostgreSQL temporary files use dedicated shared-disk tablespaces, avoiding the nearly full /dev/shm volume; the 8 GB per-process temporary-file limit is retained. Resources are specified separately in resources.json so recipe, policy, measurement budgets and existing journal identities are unchanged. Shared task locks were verified across hosts. The controller pauses at durable boundaries on service/transport failure. VAE replicas are ready during warm30 and used when FSBO starts. Controller: `batch_code/run.py controller`. Status: `status.json`; stage journals under `warm30/runs` and `orpo_batch10/runs`. Touch `PAUSE` to stop at durable boundaries. Completions/pools are immutable and identified by recipe, policy and pool hashes. Final warm pool is shared for a later ORPO+RL comparison.

Validation evidence is in `evidence/batch_*`; launch requires passing revised tests, real train-only preflight, original dependency pins and revised implementation pins. Detailed logs are in `logs/batch*`.
