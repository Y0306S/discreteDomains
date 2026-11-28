# Peptide trust-region calibration — 21 November 2026

Provisional recommendation: preserve the exact global similarity constraint, enforce at most2 local Levenshtein edits from a known feasible center, start a separate pilot at latent trust-region length0.15, and compare0.2 as a bounded expansion. Very small latent regions repeat known peptides; the current0.8 region decodes far outside the feasible neighborhood. These results establish decoder yield, not MIC gains.

## Completed diagnostic

Two source tasks per allowed-reference-edit budget2,3,4,5,6 were sampled with seed20261121 from the training800, without inspecting outcomes. Ten development tasks,2VAEs (author and adapted),8latent lengths,3paired seeds and64draws/seed; plus3saved VAE checkpoints for pep999. Total35328decoder draws. No training, MIC predictor calls or production changes. Source tasks are development data, not independent test results.

Each task uses the reference and first3initialization sequences as centers. Native Sobol/sparse-mask geometry perturbs about20/256coordinates by at most4×length. Seeds, centers and perturbations are paired across lengths/models. Decoder validity also enforces sequence syntax, maximum30residues and at most1C. Novelty excludes all100 initialization sequences and the reference; future production caches may lower yield.

## Development results: adapted VAE, two local edits

| Latent length | Pass exact global + local constraints | Mean distinct unseen feasible peptides /192draws | Minimum across10tasks |
|---|---:|---:|---:|
|0.025|97.5%|0.0|0|
|0.05|97.5%|0.0|0|
|0.075|97.4%|0.6|0|
|0.1|96.7%|4.4|0|
|0.15|86.0%|33.7|12|
|0.2|56.7%|43.6|24|
|0.4|6.8%|8.0|0|
|0.8|0.5%|0.6|0|

At length0.15, one local edit gives79.9%valid and22.2unique novel peptides/192draws, compared with86.0%and33.7for two edits. Two edits retain more candidate coverage and are the recommended pilot default. Three edits remain a diagnostic comparator in summary.csv; they do not replace the global reference limit.

The0.15 recommendation balances decoder acceptance and novelty, chosen from development data. Length0.2 maximizes observed mean novel yield among tested lengths for the adapted VAE but rejects more decodes. Length0.1 has high retention but yields very little novel exploration. A fixed loss gate cannot substitute for these measurements. These small task-clustered probes do not identify the optimum for MIC or a universal radius for every peptide.

## pep999: inspected diagnostic only

At two local edits, the final round40checkpoint produces27distinct unseen feasible peptides/192draws at length0.15 and38at0.2, versus2at0.8. The pre-online checkpoint gives28,40and1respectively. These are decode-only probes around reference/initialization centers, not the historical incumbent, GP-selected acquisition or a replay of optimization. They were not used to choose global settings, and none were MIC-scored.

## Proposed implementation for a separate pilot

1. Keep the global limit distance(sequence,reference) <= floor(reference_length/4). For pep999 that is3edits. Independently enforce distance(sequence,center) <=2. An edit includes insertion, deletion or substitution. Candidate centers must themselves satisfy the global constraint.
2. Use length0.15 for the conservative arm; compare0.2 and a one-edit arm. Keep the current wide-search policy as a control. Initializers, target fitting, model weights and seeds must match across arms. Hold VAE/source-prior updates fixed during this first search ablation.
3. Decode a bounded pool, enforce exact constraints before MIC scoring, remove cached and within-batch duplicates, and retain original latent indices. Use several already-observed feasible centers when yield is low. Do not relax the reference constraint or silently fill with invalid samples.
4. Replace repeated recenter replay with bounded generation of novel constrained candidates. If latent resampling still yields no novel candidates, an explicit local mutation pool (then VAE encoding and FSBO ranking) is a separate fallback arm to evaluate. Do not add it silently to the radius comparison.
5. Record all rejected decoder draws, cache hits, accepted sequences, physical MIC calls and wall time. New screened runs use a different proposal policy from historical runs; compare equal physical MIC calls and separately report all generation attempts.
6. Run the4arms on the10source development tasks with3seeds and500new physical MIC calls per task/arm/seed (maximum60000calls), then freeze the policy and evaluate untouched held-out queries. This optimization pilot is specified in pilot_plan.json and has not been launched. No MIC benefit is claimed from decoder-only results.

The isolated edit_region.py filter passed checks for global/local boundaries, insertions/deletions, duplicate rejection and original-index alignment. It is not imported by production. The upper radius0.2 and lower radius0.1 are provisional pilot bounds, not a deployed adaptive schedule.

Artifacts: calibration.png/pdf/svg, results.json, summary.json/csv, draws.csv, calibrate.py, edit_region.py, check_edit_region.py, pilot_plan.json.
