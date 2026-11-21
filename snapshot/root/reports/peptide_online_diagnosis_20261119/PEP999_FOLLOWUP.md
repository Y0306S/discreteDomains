# Focused follow-up: pep_999 online tuning

The measured failure is inefficient feasible exploration. The current logs do not prove that the online updates caused the gap to frozen FSBO.

| Event | Online VAE + FSBO | Frozen FSBO |
|---|---:|---:|
| Initial best predicted MIC (µM) | 112.650 | 112.650 |
| First discovery of KLYWRSFAITALAL | Round 52; MIC 109.458 | Round 69; MIC 109.463 |
| Discovery of KLYWRSFAITKAL | Never generated | Round 72; MIC 43.308 |
| Fresh feasible candidates in rounds 61–80 | 3 / 966 (0.31%) | 16 / 935 (1.71%) |
| Final measured MIC under the established cap | 109.458 | 43.308 |

The online method found the same intermediate candidate earlier, then failed to discover the next useful sequence. It encountered its incumbent sequence 21 times, including cached observations, versus four encounters in the frozen run through round 80. The better frozen candidate never appeared in the online observation history, so it was not discarded through an incorrect score comparison or feasibility flag. Both sequences are three edits from the 13-residue reference and satisfy the 75% constraint.

All seven online updates were accepted. Each completed its 100,000-step scratch FSBO rebuild with a verified reload. Each retraining manifest includes pep_999 and its current observations; this is not a case where the target task was omitted. Task episodes are sampled uniformly among 800 sources plus the current target after the initial full pass. The subsequent target-specific FSBO fitting still occurs each acquisition round.

There are two relevant limitations in what the training optimizes and validates:

1. The VAE update uses sequence reconstruction plus KL loss; MIC and constraint GP losses do not backpropagate through the VAE. It does not directly train the decoder to produce lower-MIC, constraint-valid candidates around the current target.
2. The VAE acceptance gate checks 128 reconstructions from 32 source tasks. The scratch FSBO checkpoint is selected using source validation tasks; pep_999 is not one of them. Passing these checks therefore does not establish improved target acquisition feasibility. This source-only validation is a design choice, not evidence of an implementation error or a reason to select on held-out future target outcomes.

The target admits at most three edits. Online acquisition candidates commonly decoded eight to twelve edits away (medians by ten-round window), so most of the physical evaluation budget went outside the allowed neighborhood. The constraint model ranks latent candidates before decoding; exact post-decode infeasibility is recorded, but those candidates are still scored and charged to the budget. Existing results must preserve that accounting.

Recenter attempts produced 2,138 decoded proposals, only four fresh sequences, and no fresh feasible sequences. This provided little new feasible neighborhood coverage. Of the 5.02 recorded task wall hours, approximately 3.48 were spent on source re-encoding and full scratch FSBO rebuilding, and 1.48 on target refitting.

All saved online feasibility flags were recomputed against the reference and agreed. This rules out a wrong saved feasibility flag as an explanation for the observed gap. The failure existed before the first online update: zero fresh feasible acquisitions in rounds 1–30. With one optimizer seed, different initial VAE/FSBO weights between methods, and no matched adapted-frozen arm in this campaign, a causal statement that online tuning itself degraded quality would exceed the evidence.

The next experiment needed to separate effects is an adapted-frozen control starting from exactly the online arm's initial VAE and FSBO weights, with the same initialization, optimizer seeds and oracle accounting. A separate diagnostic arm can test target-observed reconstruction retention and smaller latent perturbations; exact feasibility filtering must report physical oracle calls consistently. No extra experiment or budget extension was launched during this read-only follow-up.

[Diagnostic graph](pep999_diagnostics.png) · [Machine-readable evidence](pep999_followup.json) · [Cohort report](REPORT.md)
