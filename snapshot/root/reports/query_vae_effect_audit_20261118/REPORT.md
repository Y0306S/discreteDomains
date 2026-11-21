# Query VAE and FSBO effect audit

The retrained variant has **no demonstrated optimization improvement so far**. It does not show the severe reconstruction collapse seen in the earlier peptide adaptation. Better reconstruction has not translated into better latent-space acquisition.

## Matched production results

Read-only snapshots of both campaigns contain 29 retrained-run tasks and 21 original-run tasks with committed optimization state. Compare the 21 overlapping queries, truncating both histories to the same number of complete ten-plan rounds within each query. This retains **7,030 optimization measurements per arm**, plus identical saved BOLT30 starts. Timeouts remain censored and do not become exact incumbents. All matched retrained-run observations precede the recorded RAM migration/resume boundary; the saved later measurements are not used for this comparison. Queries have unequal matched budgets, so this is a descriptive paired snapshot, not a fixed-budget final benchmark.

| Metric | Result |
|---|---:|
| Geometric mean of retrained/original best runtime | 1.0026 (+0.26%) |
| Median ratio | 0.9971 |
| Task bootstrap 95% interval for geometric mean ratio | 0.9794–1.0261 |
| Retrained wins / losses / within 5% | 2 / 3 / 16 |
| Current best plans proposed by LLM | 21/21 |
| Random-filled slots: original / retrained | 226 / 276 |
| FSBO requested slots filled: original / retrained | 94.30% / 93.18% |
| Accepted among returned FSBO decodes: original / retrained | 47.99% / 44.78% |

The returned-decode rate is measured after the VAE service's internal crossjoin retries; it is not first-draw validity. Rejected returns combine exhausted decoder fallbacks and repeated canonical plans. The FSBO slot-fill rate includes up to five acquisition batches. Matched selected-arm schedules each contain 577 LLM-selected and 126 FSBO-selected rounds, although LLM-selected rounds can call FSBO to fill missing slots. Current winners being LLM plans does not rule out earlier FSBO feedback influencing later LLM proposals.

There are 912 common successful non-warm plan measurements; their geometric mean retrained/original timing ratio is 0.9943. This provides context for timing variability, not a correction factor. The 21 queries span only three templates (10A, 11A, 11B); these results do not establish performance across all 200 tasks.

## Direct VAE checks

The selected epoch-8 VAE improves training-side validation exact-string reconstruction from **75.35% to 92.90%** and total validation loss from **0.06614 to 0.03560**. This metric uses the native teacher-forced reconstruction path, so a separate autoregressive probe was necessary.

The offline probe uses the first held-out query from each of 15 templates and its first two saved BOLT plans in generation order, selected without looking at runtimes. For each example it makes two stochastic draws in each setting. No SQL is executed and no training or checkpoint selection uses this diagnostic. All final executed plans are canonicalized and strictly validated using the same policy as production; raw-token strict validity is saved separately.

| Probe setting | Original valid non-crossjoin | Retrained valid non-crossjoin |
|---|---:|---:|
| Decode encoded plan mean | 59/60 | 60/60 |
| Add Gaussian noise, SD 5% of each source coordinate span | 37/60 | 50/60 |
| Add Gaussian noise, SD 20% of each source coordinate span | 7/60 | 4/60 |
| Unranked candidates from native initial trust-region geometry | 8/60 | 8/60 |

The perturbations use each model's own source normalization span, representing its search setup; they are not a decoder-only comparison at identical absolute latent vectors. These are small diagnostic samples, not direct acquisition quality or query runtime measurements. At the encoded mean, the retrained model returns the exact input plan in 59/60 draws versus 53/60 originally: strong reconstruction also means little novelty there. At the small perturbation, valid plans novel against BOLT30 are 24/60 retrained versus 22/60 original.

Across all 450 BOLT plans for those 15 queries, the original encoding has 1/28,800 coordinates outside its source minmax bounds; the retrained encoding has 22/28,800. This does not indicate a large general encoding-range failure. Both decoders struggle with broader latent perturbations, consistent with search leaving regions that decode to useful plans. The probe does not prove that this is the cause of all production underfilling.

The acquisition wrapper resets native TuRBO to length 0.8 each call. Without absolute bounds, it constructs candidates with offsets up to **±3.2 raw latent units**, perturbing roughly 20/64 coordinates. That half-width is about **47% of the original source coordinate span and 54% of the retrained span at the median coordinate**. Source minmax normalization is applied when scoring candidates with the GP, but it does not constrain this raw candidate box. Thus the retrained decoder is being searched with a relatively wider box. The fourth diagnostic uses this sparse uniform candidate geometry before GP ranking, not selected Thompson proposals; it supports checking search scale but cannot substitute for evaluating selected proposals. This behavior is shared by both compared hybrid wrappers, rather than a newly introduced VAE training bug.

## Implementation and comparison differences

- Original FSBO used **788 usable historical tasks: 631 training, 157 validation**. The retrained variant uses **80 selected tasks: 64 training, 16 validation**, containing 2,663 unique uncensored source plans. Consequently, this is a combined VAE, FSBO-corpus, embedding and normalization change; it cannot isolate a VAE effect.
- The new FSBO completed 100,000 training steps and serves the selected step-3,600 weights, with verified reload. Its final training-step validation was worse than the selected checkpoint; the final checkpoint was not accidentally substituted.
- Source plans were re-encoded with the new VAE and new bounds; saved encoder/corpus equivalence checks passed. Retrieval deliberately uses a separate original encoder and its matching original source GPs. No evidence of mixing the two coordinate systems was found.
- The active VAE checkpoint hash is `eebbe1ee70209aec6b8236d3c01d43c0d3591e52f9bbc23e311c610712e0204d`. All 16 configured local VAE processes have the matching checkpoint arguments. Twelve health requests returned ready with the matching loaded model and separate original retrieval encoder; four timed out even after a longer retry. Their processes were still present. Because health uses the same mutex as decoding, this check cannot distinguish a long decoding request from a stalled service. Live evidence is saved in `live_services.json`.
- The new VAE is frozen during held-out Query optimization; peptide's repeated online VAE updates are not occurring in this Query run.

## Recommended next experiment

Keep the current results and running configuration intact. A controlled ablation should compare the original and retrained VAEs with the **same source task split, measured plans, FSBO training budget, acquisition seeds and SQL storage/cache protocol**. To compare with the original full-source method, re-encode the same 788 usable source tasks under the retrained VAE and train FSBO on the same 631/157 split. Compare decoder novelty/validity as well as BO incumbents. Replay shortlisted plans under a common SQL measurement setup before attributing a small runtime difference to model quality. No production retraining, rollout restart or policy change was performed by this audit.

Artifacts: `comparison.json`, per-query CSVs, `decoder_probe.json`, `live_services.json`, and `comparison.png/pdf`. Audit code and probe code are stored alongside these files.
