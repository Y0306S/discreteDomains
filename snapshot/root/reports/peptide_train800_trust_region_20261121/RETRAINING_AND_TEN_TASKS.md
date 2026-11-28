# Retraining checks and feasibility across the ten peptides

A full radius search is unnecessary after every FSBO target fit. FSBO-only training changes the objective surrogate and ranking, while the VAE determines the mapping from latent coordinates to peptide sequences. The target GP already refits every acquisition round. With a fixed VAE, track selected-candidate feasibility and novelty and keep normal trust-region adaptation.

After each accepted VAE update, a small paired decoder check should be part of validation: current measured feasible centers; identical seeds/observation set before and after; a short radius sweep around 0.125–0.2; exact reference feasibility; unique unobserved feasible sequences; and posterior-reconstruction novelty used by the recenter stage. This probe needs no MIC calls. Re-encode centers and observations after a VAE change; keep the original reference edit constraint unchanged. The current run has loss rollback but does not yet automatically perform this radius/decoder-health check. Thresholds for rejecting an update need source-task calibration.

The scalar length is not a sequence edit budget or Euclidean diameter. The implemented bounds are center±4×length in perturbed coordinates, so length 0.175 gives coordinate width 1.4. A local two/three-edit gate is an additional exact sequence-space check.

## Measured acquisition feasibility

First 80 acquisition rounds (4000 proposals). Denominators are fresh acquisition candidates actually evaluated; initialization, cached repetitions and recenter candidates are excluded. Every saved feasibility flag was rechecked against exact reference Levenshtein distance. Percentages measure constraint satisfaction, not MIC improvement.

|Task|Latest rollback: feasible/fresh|Latest rollback rate|Frozen FSBO rate|
|---|---:|---:|---:|
|pep_117|20/3969|0.50%|0.05%|
|pep_219|37/3944|0.94%|0.71%|
|pep_221|9/3913|0.23%|1.55%|
|pep_575|4/3992|0.10%|0.85%|
|pep_673|10/3906|0.26%|0.05%|
|pep_921|22/3751|0.59%|0.09%|
|pep_925|39/3847|1.01%|0.34%|
|pep_967|9/3748|0.24%|0.19%|
|pep_980|18/3883|0.46%|0.17%|
|pep_999|25/3729|0.67%|0.48%|

The other nine tasks jointly have 168/34953 feasible fresh acquisitions (0.481%). Across all ten, the rollback run has 193/38682(0.499%); frozen FSBO 167/36715(0.455%); previous online 185/38646(0.479%). The problem is shared across tasks and methods, rather than confined to pep_999.

Before the first online VAE update (after round 20 in all ten), only 2/9703 fresh acquisitions were feasible (0.0206%). Thus poor feasibility already existed before online updates. These observations do not establish that offline VAE adaptation or GP ranking has no effect.

Recenter accounting adds a separate difference: frozen generated 324 feasible new sequences from 1840 fresh recenter candidates across the ten; latest rollback generated zero feasible sequences from only 3 fresh recenter candidates. Most rollback recenter proposals were cached repeats. Therefore acquisition-only percentages do not describe every source of useful frozen observations. Improve or bypass replay that produces no novelty, and measure recenter novelty alongside local-radius health.

## What the completed 800-task check measured

It tested author and adapted decoders, paired seeds, several latent lengths, exact local 1/2/3/global gates, novel-candidate counts and APEX-predicted MICs on fixed pools. It did not repeat sequential GP training or 4000 evaluation optimization for every task. Four decoder shards took approximately 12.8–15.1 minutes after shard startup (imports/earlier transition additional); scoring took 5.3 minutes and overlapped decoding. These timings do not establish production overhead per VAE update.

Reviewable prototype changes are exact filtering before MIC, deduplication, several feasible centers and ranking the canonical re-encoding of actual decoded sequences. They remain isolated from production. The recommended next sequential ablation holds VAE/source-FSBO fixed while comparing the original policy and sequence-first variants, then separately tests online retraining. Keep two-edit and three-edit local modes as controlled arms, and retain a globally feasible exploration comparison.
