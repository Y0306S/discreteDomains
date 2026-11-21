# Peptide constraint pass rates: changes grounded in the existing LOLBO code

Status: investigation and proposal only, 2026-11-01. No optimizer, pinned recipe,
production history, running worker, model, or source bank was changed. The new
audit reads saved histories; it makes zero decoder or APEX calls.

## The main finding changes the comparison

The previous combined LOLBO rate mixed acquisition with recentering. A new audit
reads all 200 completed LOLBO histories and checks each history SHA256 against
the previous full archive audit.

| Method / phase | Fresh measurements | Fresh similarity-feasible | Rate |
| --- | ---: | ---: | ---: |
| TPE acquisition | 888,995 | 171 | 0.01924% |
| FSBO acquisition | 887,925 | 7,025 | 0.79117% |
| LOLBO acquisition | 582,140 | 4,133 | 0.70997% |
| LOLBO recenter | 406,512 | 128,346 | 31.57250% |
| LOLBO combined | 988,652 | 132,479 | 13.39996% |

Thus FSBO's acquisition pass rate is slightly higher than LOLBO's acquisition
pass rate. 96.88% of LOLBO's fresh feasible measurements occur in recentering.
This is phase attribution, not a causal ablation proving how much comes from
joint representation updates versus reconstruction-based sampling.

The existing convergence audit also reports that FSBO improves 196/200 tasks
with mean per-task MIC improvement of 59.06%, versus LOLBO's 167/200 and 39.51%.
Higher feasibility alone is not evidence of better optimization.

RGPE has no completed archives in the inspected campaign. Its available 48
partial histories contain 21,806 fresh measurements and one fresh feasible
measurement (0.00459%). This is an incomplete snapshot, not a matched final
comparison. See phase_audit.json for the exact counts and limitations.

All 1,442,184 post-initialization LOLBO history rows and all inspected RGPE rows
pass alphabet, nonempty length <=30, and at-most-one-cysteine checks. The native
wrappers validate these structural rules before scoring. Their learned black-box
constraint is similarity: c(x) = 0.75 - similarity(x, reference), feasible at
c <= 0. Consequently, among recorded structurally valid scored candidates,
passing similarity also passes the implemented constraints. This does not
describe raw decoder tokens before native postprocessing or unimplemented
biochemical requirements.

## What FSBO and RGPE already share with LOLBO

The transfer adapter already uses a best measured feasible center, native raw
latent TuRBO bounds, a 5,000-candidate Sobol pool, sparse perturbation probability
20/256, the target similarity GP, and constrained batch Thompson sampling.
Both methods propose batches of 50. Recommending these as entirely missing
features would be incorrect.

Native default region length is 0.8; raw-coordinate weights are 8, giving initial
half-width 3.2 in perturbed coordinates. Native success/failure tolerances are
10/32, length bounds are 0.5**7 and 1.6. Constraint-aware state updates already
shrink after sustained lack of feasible objective improvement.

## Proposed change list

1. **Port the existing recenter proposal stage to the other peptide adapters as
   a separately measured variant.** Use the recent acquisition batch plus the
   unique feasible top-k archive (native k=1000). Encode these seed sequences
   using the VAE posterior sampling already used by native bo.encode, then
   decode, score, and record the resulting sequences. Use the existing finite
   two-epoch recenter schedule and ten consecutive non-improving acquisition
   rounds as the initial reference policy. Recenter events do not advance the
   acquisition round or update TuRBO as if they were acquisitions, matching
   LOLBO's acquisition=False handling. Method batch sizes remain explicit:
   ten rounds of batch-one TPE is a different number of acquisition proposals
   from ten batch-50 rounds. For frozen transfer/TPE variants, first test this
   stage with the author VAE unchanged; its yield is not established by the
   updated-VAE LOLBO results. Seeded latent sampling and stochastic native
   decoding may generate fresh nearby sequences; cache hits remain cache hits.

2. **Train FSBO/RGPE's target constraint GP on the vectors actually decoded.**
   Currently the adapter stores proposed_z, but appends canonical_z (the frozen
   posterior mean of the output peptide) to features and uses features to train
   the constraint GP. Native LOLBO instead appends z_next and its measured
   constraint, and fits its constraint models on that train_z. Preserve the
   existing canonical frozen features for FSBO/RGPE objective transfer, and
   maintain a separate target constraint training array containing actual
   proposed/recenter z. Initialize that array with the measured pool's known
   encodings. Repeated sequences still provide constraint observations at each
   actual latent input, while their oracle score is cached. This ports native
   LOLBO's feedback association without changing the source bank. Its effect
   on pass rate needs an ablation; code inspection establishes the difference,
   not its causal contribution.

3. **For TPE, port feasible-centered local search and the sparse mask.** Keep
   TPE's density-based proposal logic and explicit constraint feedback, but
   restrict proposed changes to the native local region and perturb a subset
   of coordinates using the native 20/d mask. Keep all other coordinates at
   the feasible center. Use LOLBO's constrained success/failure updates to
   adapt the region. This is a local sparse TPE variant, not plain global TPE.
   Changing masked coordinates must be part of candidate construction and
   accurately represented in ask/tell records; simply editing a suggested
   vector after ask would misreport what was evaluated. Reuse the existing
   native state/bound calculations rather than the legacy TrustRegionTPESampler
   unchanged: that helper expects z_* names and minimization with penalty-based
   feasibility, unlike the test200 z000 names, maximization, explicit constraints.
   No multivariate-TPE change is included: that is not a LOLBO mechanism.

4. **Give other peptide BO adapters the same local feasible search policy and
   explicit constraint observations.** For GP adapters that support posterior
   sampling, reuse the existing native Sobol/mask/constraint-TS machinery with
   their objective posterior. HEBO/SMAC-style adapters retaining their own
   acquisition need that acquisition restricted to the same local domain and
   a deliberate constraint integration; replacing it with TS creates a named
   variant. The common part can include recenter proposal events and exact
   similarity scoring without replacing the method's objective model. There
   are no matched final peptide rates established here for those other methods.

5. **Match native trust-region initialization/restart bookkeeping.** Transfer
   currently initializes best_constraint_values to zero; LOLBO initializes it
   to the actual constraint at the best feasible observation. At restart,
   transfer selects the incumbent from saved rows before appending the latest
   batch; native LOLBO reinitializes after update_next. Use the actual best
   feasible score/constraint over the committed observations, including the
   latest batch. This is a small parity correction, not an established main
   explanation for the feasibility gap.

6. **If needed, tune only the existing locality parameters on training tasks.**
   Smaller initial length, shorter failure tolerance, and fewer perturbed
   coordinates are existing LOLBO knobs/formulas. Compare these as declared
   parameter variants with the current native defaults; no new adaptive
   feasibility-rate controller is proposed. First isolate the recenter and
   constraint-coordinate changes so a combined result is interpretable.

7. **Use native LOLBO for any peptide run labeled LOLBO.** Its audited V5 path
   already performs joint VAE/objective/constraint updates after stagnation,
   all-history re-encoding, surrogate refresh, and bounded recentering. The
   generic legacy run_domain_search path selects CmaEsSampler for non-TaskSet
   optimizer='lolbo'; that is not the audited peptide implementation. Other
   legacy paths can repair sequences toward the reference or resample
   duplicates. Route new comparisons through the validated native decoder,
   initialization, scoring and accounting behavior instead. This finding does
   not invalidate the completed native V5 LOLBO campaign.

## Full end-to-end updating is not a universal switch

Native LOLBO's joint loss combines VAE loss, objective surrogate loss and
constraint surrogate loss. A TPE density model or SMAC random forest cannot
participate in that same differentiable GP loss. Enabling the existing legacy
VAE-only helper would not reproduce the native joint update.

FSBO could be investigated as a separate differentiable end-to-end variant,
with target observations re-encoded and models refit after each VAE change.
Its pretrained features were learned in the frozen representation, so this
changes their input distribution and requires validation.

RGPE's 800 source GPs explicitly depend on the frozen canonical coordinates.
Turning on target VAE updates while directly reusing their old-coordinate
predictions is inconsistent. Making that coherent requires representation
alignment or source-model rebuilding, which is beyond a direct LOLBO-only
port. The recommended first RGPE variant therefore keeps the VAE and source
bank frozen and tests recenter proposals plus actual-z constraint feedback.
No new coordinate-alignment model is proposed.

For LOLBO itself, the audit gives no basis for replacing its existing
update/recenter mechanism. Measure acquisition and recenter separately, and
test the existing locality parameters if better acquisition feasibility is
the aim. Do not treat its combined 13.4% as an acquisition target already
demonstrated by native LOLBO.

## Evaluation and implementation scope

Prepare new versioned adapters/recipes, preserving completed runs. Keep the
same 100 initial measurements, reference-specific 0.75 threshold, native
decoder, objective and cached-score behavior. Preserve the shared maximum of
5,000 fresh post-initialization measurements across acquisition and recenter,
and at most 5,000 acquisition proposals. Record recenter proposals separately,
as LOLBO already does. Persist the finite recenter plan, actual constraint
coordinates, counters, RNG and pending scores for reproducible recovery.

Run a small paired train800 pilot before a new full campaign. Report fresh
feasible rate separately for acquisition and recenter, their combined rate,
unique fresh feasible sequences, duplicates, feasible MIC improvement and
runtime. Do not count extra decode-and-reject attempts, sequence repair, or
reference mutation as improvements to these methods: none is part of the
native audited LOLBO search policy being copied.

## Code evidence

- Transfer coordinate handling and native acquisition:
  /root/data/cys/llmoptimizer_benchmark/tools/peptide_transfer_eval.py:375,
  :435, :460, :472.
- Native actual-z feedback and recenter orchestration:
  /root/data/logs/peptide_bo_release_20261027_v5/tools/peptide_bo_v5.py:299,
  :386, :407.
- Feasible centers and unique feasible top-k:
  /root/data/logs/peptide_bo_release_20261027_v5/mindware/domains/protein/verified_bo.py:172,
  :223.
- Native local bounds, mask and constraint-aware state:
  /root/data/logs/peptide_bo_train800_20261027_v5b/upstream_cpu/lolbo/utils/bo_utils/turbo.py:9,
  :35, :105.
- Native joint loss:
  /root/data/logs/peptide_bo_train800_20261027_v5b/upstream_cpu/lolbo/utils/utils.py:35.
- Plain TPE setup/ask/tell:
  /root/data/cys/llmoptimizer_benchmark/tools/peptide_test200.py:216, :265.
- Legacy non-TaskSet LOLBO dispatch:
  /root/data/cys/llmoptimizer_benchmark/mindware/domains/search.py:559, :630.
- Earlier completed-run feasibility audit:
  /root/reports/peptide_tpe_diagnosis_20261101/audit.json.
- Earlier objective comparison:
  /root/reports/peptide_convergence_20261101/README.md.
- New phase audit: audit_phases.py and phase_audit.json in this directory.
