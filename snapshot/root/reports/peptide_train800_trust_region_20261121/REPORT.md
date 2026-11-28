# Peptide trust regions: all 800 training-task diagnostic —21 November 2026

## Findings

The completed diagnostic supports testing a three-edit local mode alongside a two-edit conservative control. It does not support a universal two-edit cap, and it does not establish that online training caused the pep_999 gap. Smaller latent neighborhoods produce much more useful sequence-space coverage than the original starting length 0.8. A local cap alone is insufficient: use several feasible centers and retain an explicitly global-feasible exploration arm.

These are fixed-pool decoder and predicted-MIC diagnostics, **not 800 completed sequential BO runs**. No production model, running optimization policy or historical result was changed. The held-out 200 tasks were excluded. New isolated code is available for review, not deployed.

## Why frozen FSBO could succeed with the same wide region

Frozen and online used the same nominal recipe: length 0.8, minimum 0.0078125, maximum 1.6; approximately 20/256 perturbed coordinates;5000 Sobol candidates;batch 50;seed 1. Online evaluate.py reads the frozen recipe directly. Their 100 initialization sequences, measured scores and order match. Both online runs fit the target GP before the first acquisition, so missing initial target data is not the explanation.

The models differ. Frozen uses the author VAE, whereas online uses an adapted VAE and updates its models. The same latent length therefore need not imply the same sequence-edit distances, centers or acquisition ranking. A read-only replay of saved acquisition batches through the native update rule gives a more precise answer: frozen and latest rollback both used length 0.8 in rounds 1–32,0.4 in rounds 33–64,and 0.2 in rounds 65–80. Frozen found its strong candidates at rounds 69 and 72 **after shrinking to 0.2**, and the latest online improvement at round 67 also used 0.2. The previous online arm improved at round 52 at length 0.4, resetting its failure counter, and stayed at 0.4 through round 80. These are reconstructed states, not original radius telemetry; radius_replay.json records the trace. The rule matches the frozen CPU source snapshot. Thus frozen did not find its winner while still using the initial 0.8 length. The latent bound is center±4×length in each perturbed coordinate, not a fixed number of amino-acid edits. Length 0.2 does not mean a two-edit sequence constraint. Starting smaller could avoid waiting 64 rounds (3200 proposals) for these two trajectories to contract, but improvement must still be tested sequentially.

Frozen also explored inefficiently on pep_999:17/3561 fresh acquisitions were feasible within the 4000 proposal comparison, versus 25/3729 for the latest rollback arm. It found KLYWRSFAITALAL at round 69, then KLYWRSFAITKAL at round 72, reaching 43.3079 µM. The latest online run reached 101.0068 µM and never generated the latter sequence. More feasible candidates alone did not ensure a better optimum. This is one trajectory per method; neither luck nor a specific training failure is established as the sole cause.

## What was tested

- All 800 configured training tasks, using the exact initialization manifest, task IDs and pinned model weights. IDs are read from the campaign, not assumed to be 000–799.
- Adapted VAE at lengths 0,0.1,0.125,0.15,0.175,0.2,0.3,0.8; author VAE at 0.15,0.2,0.8.
- Three paired seeds,32 decodes per seed,96 draws per task/radius; **844,800 saved complete decodes**. Uncommitted partial decoding during the two-to-four-shard transition was not quantified and is additional compute.
- Fixed centers: reference and first three initialization sequences, chosen without generated-candidate outcomes. These are not adaptive incumbent trajectories.
- Exact original reference constraint: Levenshtein distance≤floor(reference length / 4), plus peptide syntax, length≤30 and at most one C. Local 1,2,3 and global-only gates were compared. Local edits are measured from the generating center; global edits always from the task reference.
- Deduplicated, initially unseen globally feasible candidates were scored with the original eight-model APEX ensemble, bacteria 0, native MIC transformation and averaging. Initial 100 were rescored on the same device and precision, preserving the stored historical scores separately.
- Predicted-MIC summaries take the first 5,10 or 20 available accepted candidates in generation order, and separately the entire finite pool. There is no GP ranking, sequential feedback or offline selection of the best candidates before scoring.

## Full 800 results

Each decoder column uses the same 96 raw draws per task/radius. The MIC column uses at most 20 accepted unique candidates and includes the initial incumbent. A ratio below 1 means improved predicted MIC; lower is better. Underfilled pools consume fewer actual candidate predictions and must not be interpreted as matched 20 call runs.

|Latent length|Local edits|Feasible draws|Mean unique new feasible /96|Tasks with zero new|Geometric best MIC / initial|Tasks improved|Mean candidates used ≤20|Tasks filling 20|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|0.15|2|82.100%|19.18|0.0%|0.9648|76/800|16.6|46.1%|
|0.15|3|84.383%|21.36|0.0%|0.9611|81/800|17.3|54.1%|
|0.15|global only|85.151%|22.10|0.0%|0.9588|83/800|17.4|56.0%|
|0.175|2|66.823%|23.17|0.0%|0.9570|102/800|18.4|68.2%|
|0.175|3|70.882%|27.06|0.0%|0.9458|125/800|19.0|79.8%|
|0.175|global only|72.684%|28.79|0.0%|0.9457|123/800|19.1|81.0%|
|0.2|2|50.043%|22.03|0.0%|0.9586|102/800|18.3|65.1%|
|0.2|3|55.409%|27.18|0.0%|0.9504|120/800|19.2|82.9%|
|0.2|global only|58.336%|29.99|0.0%|0.9461|122/800|19.3|84.8%|
|0.3|2|11.522%|6.69|0.9%|0.9715|79/800|6.7|0.2%|
|0.3|3|15.092%|10.12|0.6%|0.9507|109/800|10.1|3.2%|
|0.3|global only|18.052%|12.96|0.4%|0.9413|135/800|12.6|11.4%|
|0.8|2|0.009%|0.01|99.1%|0.9993|1/800|0.0|0.0%|
|0.8|3|0.016%|0.01|98.6%|0.9993|1/800|0.0|0.0%|
|0.8|global only|0.018%|0.02|98.5%|0.9993|1/800|0.0|0.0%|

At length 0.175, the paired three-edit/two-edit MIC ratio is **0.9883**, with a 95% task-bootstrap interval[0.9809,0.9947]. Three edits performs better on 30 tasks, two edits on 5, and 765 tie. This averages over available pools, including underfill. Restricting to the 546 tasks where both gates fill 20, the ratio is 0.9869, interval[0.9777,0.9949]. The both-filled subset is selected for candidate availability and is not the whole population.

### Radius selection and internal check

For each mode, radius selection minimized the 640-task geometric MIC ratio at capped 20 available candidates over 0.125,0.15,0.175,0.2,0.3. The remaining 160 form an internal development check. Both groups were previously used in representation/source training; this is **not an unseen-task benchmark**, and aggregate progress previews were inspected during development. The bootstrap intervals describe task variability within this source cohort, not post-selection generalization guarantees.

|Local edits|Selected length|640 selection MIC ratio|160 internal-check ratio|160 tasks improved|160 tasks filling 20|
|---:|---:|---:|---:|---:|---:|
|2|0.2|0.9563|0.9676|18/160|62.5%|
|3|0.175|0.9453|0.9474|22/160|80.0%|
|global only|0.3|0.9384|0.9530|22/160|10.0%|

### Author versus adapted decoder

|Length|Local edits|Adapted mean unique new /96|Author mean unique new /96|Adapted feasible draws|Author feasible draws|
|---:|---:|---:|---:|---:|---:|
|0.15|2|19.18|18.89|82.100%|55.642%|
|0.15|3|21.36|22.75|84.383%|59.671%|
|0.2|2|22.03|13.00|50.043%|25.919%|
|0.2|3|27.18|17.31|55.409%|30.404%|
|0.8|2|0.01|0.00|0.009%|0.000%|
|0.8|3|0.01|0.00|0.016%|0.001%|

This compares decoder coverage, not frozen-versus-online BO convergence. No author-pool MIC experiment was run. Training effects cannot be inferred from this table alone.

## Historical evidence across all 800

The pinned source corpus contains 3,973,113 distinct initially novel sequences (counted within tasks), of which 499,793(12.58%) meet the reference constraint. Across 1621 recorded incumbent improvements,857(52.87%) were more than three edits from the preceding best peptide. Source global-edit-budget task counts are{'5': 147, '2': 9, '3': 382, '4': 187, '6': 75}.

This argues for diversity, multiple centers and bounded expansion. It does not prove smaller-step routes cannot reach the same sequences. These moves follow stored source-row order and reflect the historical sampling policy. A cap around the original reference is different from a cap around an adaptive center; the original global reference constraint must never be relaxed.

## Proposed method changes

1. **Decode, exact-filter, deduplicate, then rank actual sequences.** Apply the cheap original global constraint and optional local 2/3 gate before costly MIC prediction. Record raw decoded counts, reasons for rejection, cached candidates, requested batch size, actual fill and physical MIC calls separately. Never rewrite previous 4000 proposal results or silently pad batches.
2. **Use canonical sequence features for acquisition.** Re-encode each accepted decoded peptide with the current VAE, then apply the matching FSBO feature transform/acquisition. The existing policy ranks raw proposed latent coordinates although objective observations are stored at re-encoded canonical coordinates. That mismatch is a plausible issue, not yet a proven cause. After accepted VAE updates, re-encode observations consistently before GP fitting, as the current code already does.
3. **Compare a conservative 2-edit arm and a broader 3-edit arm**, using the same length 0.175 in the main two-versus-three-edit ablation to isolate the edit cap. Separately test the source-selected per-mode radii in policy_plan.json. Use incumbent, reference and diverse promising already measured feasible centers; retain a separately logged global-feasible exploration stream. Explore an initial 80/20 local/exploration allocation only as a preregistered pilot setting, not as a measured optimum. Global-feasible exploration still obeys the original reference constraint.
4. **Replace unproductive recenter replay with new candidate generation.** The latest pep_999 run replayed 2194 cached recenter proposals with zero new sequences. Stop repeated decode/rescore loops when they add no novel candidates; diversify centers or expand generation within a bounded allowance. Prefer a short explicitly recorded batch over unbounded resampling. The current fixed-pool test does not calibrate a full adaptive expansion schedule.
5. **Keep the existing loss rollback, add paired decoder-health diagnostics.** Loss on known initial sequences checks retention, not useful exploration. Measure local feasible yield, novel yield and displacement on fixed anchors before/after each VAE retrain, without using held-out target outcomes. Calibrate a rejection threshold on source-task online updates before turning it into an automatic veto. Do not claim the present loss gate would have rejected pep_999: both latest updates passed and no rollback occurred.

The isolated policy_prototype.py implements center selection, exact sequence gates, canonical-feature acquisition callbacks, deduplication and index-preserving ranking. check_policy.py passed the boundary/alignment checks in policy_checks.log. This prototype does not integrate the live controller, train a GP or launch optimization.

## Controlled next optimization experiment

Use the existing training cohort for policy development, then lock the policy before a held-out-task benchmark. The previously examined 10 peptides (including pep_999) are exploratory targets, not pristine test cases; report them separately from the remaining 190 within the 200-task set, and audit other prior target exposure before describing any subset as untouched. Start with source-stratified matched-seed sequential arms: unchanged geometry; sequence-first canonical ranking with global-only gate; the same pipeline with local 2; and with local 3. Hold VAE/source-FSBO weights and online-update policy fixed initially to isolate search changes. Track identical initial pools, feasible best MIC versus actual new MIC calls, all decoded/rejected/cached counts, and wall-time components. Then separately ablate multiple centers/global-exploration and re-enable online updates with the loss gate. A full 800×4000 BO campaign was not launched by this diagnostic. Sequential optimization is still required before claiming convergence or runtime improvements at that budget; the final held-out benchmark need not itself contain 800 tasks.

## Oracle validation and compute accounting

The new scorer matched the unmodified native predict_APEX function exactly on 32 sequences on the same device (64 sequence predictions across both implementations). Native source/model SHA256 hashes are retained. Historical initialization predictions differed by at most 0.1681% with this hardware/precision/batching; this check alone does not identify the source of that variation. Freshly rescored 100-sequence baselines avoid mixing those values with newly scored candidates. Historical scores are untouched.

Physical APEX sequence predictions: 61,012 candidate predictions +80,000 initialization predictions +64 parity predictions +93 predictions in the preserved first attempt that failed its strict historical-score tolerance check =**141,169**. Each sequence prediction uses all eight ensemble members. Scores are shared across overlapping radius/local-gate analyses within each task; summing table rows would overcount actual compute. Oracle calls here are predictions, not laboratory experiments.

Artifacts: radius_replay.json; trust_region_800.png/pdf/svg; frozen_and_history.png/pdf/svg; summary.json; decoder_summary.csv; mic_summary.csv; paired_2_vs_3.csv; per-task CSVs; strata_summary.csv; policy_plan.json. Shared raw draws, scored outputs, ledger and scripts are in/root/data/cys/tmp/peptide_train800_trust_region_20261121.
