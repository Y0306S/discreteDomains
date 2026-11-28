# Peptide RGPE fixed top80 / bootstrap100 pilot

100 initialization observations → source mean-only ordered-pair ranking → fixed top80 → bootstrap100 weights recomputed every round.

Four tasks, two rounds each, 50 proposals per round. Original frozen VAE, APEX, 1500→200 target-EI shortlist and 4000-step dilution horizon preserved.

One-time shared source screen: 21.92 min; four-task parallel pilot wall time: 1.05 min.

| Task | Init best feasible MIC | Round1 MIC | Round2 MIC | Acquisition round1 / round2 (s) | Fresh / fresh feasible |
|---|---:|---:|---:|---:|---:|
| pep_104 | 8.4417 | 8.4417 | 8.4417 | 16.50 / 24.59 | 99 / 0 |
| pep_109 | 8.2728 | 8.2728 | 8.2728 | 17.34 / 18.25 | 96 / 0 |
| pep_111 | 7.6157 | 7.6157 | 7.6157 | 17.04 / 23.09 | 96 / 0 |
| pep_132 | 20.9472 | 20.9472 | 20.9472 | 18.28 / 22.29 | 86 / 0 |

Lower MIC is better. Feasibility uses the original peptide similarity constraint.

Each prefilter/<task>/selection.json contains the fixed 80 source IDs and all 800 ranking losses. telemetry/<task>/round{1,2}.json records all 81 weights and per-stage timing.

All four first rounds assign target weight 1. Round2 only pep_104 retains sources: pep_166=0.10, pep_712=0.05, target=0.85; all others target=1. Posterior merge mostly skips historical models, so the large speedup cannot be attributed solely to top80 filtering.

Acquisition timing includes candidate generation, worker initialization (first call), GP fit, bootstrap weighting, EI shortlist, merged-posterior sampling and constraint sampling; excludes the preceding constraint GP fit and subsequent peptide evaluation. All four tasks yielded zero fresh feasible peptides; pep_132 repeated one already-cached feasible sequence.

summary.json gives the earlier experiment’s first-round timing for reference. It is not a controlled benchmark: hardware, cache/load, source count and ranking mode differ. Two rounds do not establish long-run optimization quality.

Validation: upstream ranking parity including ties; full-GP vs compact mean-only bootstrap weight parity on three real GPs; identical initial sequences/scores and reused canonical coordinates; all four tasks committed exactly two rounds. Full-horizon history status remains running because this finite pilot stops at 2 events.
