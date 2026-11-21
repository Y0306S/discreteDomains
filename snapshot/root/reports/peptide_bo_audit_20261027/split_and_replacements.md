# Peptide split and replacement audit

## Applied V5 Policy

The initialization and recenter fixes passed live validation and the user
authorized the fresh production campaign:
`/root/data/logs/peptide_bo_train800_20261027_v5b`.
The original split and histories remain preserved. This campaign replaces
`pep_049`, `pep_073`, `pep_082` with new IDs `pep_1000`, `pep_1001`, `pep_1002`
using the three APEXGo references listed below. All 200 test references remain
unchanged. Maximum cross-partition similarity remains 0.736842105263158.

All 100,000 selected initialization entries now satisfy decoder residue rules.
Filtering is by original file order, not score; 774 shortage/replacement mutants
were freshly scored, with exact sequence/score provenance preserved. V5b reuses
that measured pool without rescoring.

Recenter uses native decoding/scoring and shares the 5,000-fresh-call budget with
acquisition. Acquisition is separately capped at 5,000 proposals, in batches of
50. Hence a task can finish before 100 acquisition rounds. The ten-round update
trigger counts consecutive acquisition failures, including cached batches.

The remainder is the historical analysis preceding implementation; its statements
that launch was prohibited describe that earlier stage. Current validation and
worker details are in `README.md` and the campaign's `workers/` directory.

## Agreed settings

- Use a constraint GP and constrained Thompson sampling.
- Retain trust-region contraction after 32 consecutive unsuccessful batches.
- Trigger VAE/GP updates after 10 consecutive non-improving acquisition rounds.
  An improvement resets the counter; this is not a periodic every-ten-round update.
- Retain v4 initialization and acquisition settings: 100 precomputed initialization
  observations and 100 acquisition rounds of 50 proposals (5,000 proposals).
  Cached duplicates count as proposals. The selected initialization rows are the
  original first 100 aligned sequence/score pairs for each reference.

V4 used encode-only re-encoding and made no additional recenter oracle calls.
The user subsequently delegated recenter accounting. The chosen bounded policy
is recorded below. The existing v4 launch hold remains in place.

## Current 800/200 split

Source: `/root/blockdata/llmoptimizer_benchmark/docs/peptide_split_800_200.json`.
SHA-256: `25cd85c12a30d46e5312e09cb5c4770418c9e04096f20d6dd5ae08a17a649138`.

The library contains exactly 1,000 distinct references. All are already assigned:
800 training and 200 test, with no overlapping or unassigned IDs.

The compatibility lower bound is the number of non-standard residues plus the
number of cysteines beyond one. A reference permits at most floor(length/4)
Levenshtein edits under the 75% similarity requirement. Replacing non-standard
residues and excess cysteines constructs a compatible sequence within this bound
when sufficient edits are allowed; this does not guarantee VAE generation.

| Partition | Impossible references | References with X | References with multiple C |
| --- | ---: | ---: | ---: |
| Training | 3 | 3 | 17 |
| Test | 0 | 0 | 1 |

The three impossible training references remain `pep_049`, `pep_073`, and `pep_082`.
All references have length at most 30.

All 160,000 train/test reference pairs were checked using
`max(1-distance/len(train), 1-distance/len(test))`. Zero exceed 0.75. The largest
similarity is 0.736842105263158. The test set therefore needs no replacement for
the checked residue, length, or cross-partition similarity criteria.

Test `pep_235`, `CYRAYICRKKFLSLRKASIKLQSL`, has two cysteines. Only one forced edit
is needed and six edits are allowed, so feasible decoder-compatible sequences
exist. It is not one of the impossible references.

## Test initialization audit

The first 100 entries for each of the 200 test references were checked in the
generated initialization and score files under the authors' BOLT checkout.

- 20,000 observations checked; all scores finite and all similarity constraints met.
- Every test task has exactly 100 distinct selected initialization sequences.
- No selected initialization equals its reference.
- 347 selected initializations violate the decoder residue policy, all by having
  multiple cysteines; `pep_235` contributes 78 of these.

These remain measured, similarity-feasible initializations. Initialization and new
decoder proposals use different residue policies upstream. Their scores should
not be silently changed or treated as evidence that the decoder can reconstruct
them unchanged. This mismatch alone does not require replacing the test tasks.

## Replacement options

There are no unused references in the existing BOLT library. Duplicating another
training reference would not create a new task, and transferring a test reference
would change the held-out set.

The authors' APEXGo reference list contains seven distinct additional references
not present in the BOLT library. All seven satisfy the standard alphabet,
at-most-one-cysteine and length requirements. Every candidate also passes the
same bidirectional similarity check against all 200 held-out references.

Source: `/root/data/cys/APEXGo/optimization/constants.py`.
SHA-256: `7970635a55d87bf9857e2e8456b25de2b8c62be49a994281ce0ef3a602661735`.

| Additional reference | Length | Maximum similarity to any test reference |
| --- | ---: | ---: |
| RACLHARSIARLHKRWRPVHQGLGLK | 26 | 0.307692 |
| KTLKIIRLLF | 10 | 0.473684 |
| RMARNLVRYVQGLKKKKVI | 19 | 0.333333 |
| RNLVRYVQGLKKKKVIVIPVGIGPHANIK | 29 | 0.310345 |
| GHLLIHLIGKATLAL | 15 | 0.400000 |
| RQKNHGIHFRVLAKALR | 17 | 0.294118 |
| HWITINTIKLSISLKI | 16 | 0.555556 |

A proposed length-matched replacement set, without selecting on MIC results, is:

| Removed task | Proposed new reference | Old/new length |
| --- | --- | --- |
| pep_049 | KTLKIIRLLF | 11 / 10 |
| pep_073 | RQKNHGIHFRVLAKALR | 18 / 17 |
| pep_082 | RMARNLVRYVQGLKKKKVI | 19 / 19 |

These come from a different paper's reference list; their provenance is not the
BOLT extinct-peptide library. Using them would be a disclosed dataset adaptation.
They need new distinct task IDs, newly generated and scored initialization pools,
and a versioned split manifest. Existing reference IDs and saved histories must
not be overwritten. No claim about optimization quality has been established.

The user subsequently accepted the training-data and reference-replacement
conflicts, delegated the recenter decision, requested correction of the
initialization/decoder residue mismatch, and explicitly prohibited launching.
No replacement, scoring, training, or optimization job is applied or started yet.

## Initialization provenance and revised preparation plan

The original BOLT release ships one 1,000-row random-mutation pool for reference
zero. Our local per-reference pools were generated by
`/root/blockdata/llmoptimizer_benchmark/tools/build_bolt_init_pools.py`, reusing
the authors' `apex_oracle/init_data/create_mutations.py` sampler. The generated
manifest records seed 0, 1,000 mutants for each of the 1,000 references, and no
top-up calls. Each sequence was scored as `-apex_wrapper(sequences)[:, 0]` and
stored with its corresponding measured APEX score. These are random mutants,
not LLM recommendations and not VAE decodes.

The mutation sampler enforces similarity but omits the standard-alphabet and
at-most-one-cysteine requirements. It can leave X and multiple C residues from
the reference untouched or introduce additional C residues. V4 selected the
first 100 measured entries without imposing these decoder requirements.

A complete scan of the 1,000-row pools found:

| Partition | Incompatible selected-prefix rows | Tasks with affected prefixes |
| --- | ---: | ---: |
| Training, original 800 tasks | 2,642 | 407 |
| Test, 200 tasks | 347 | 108 |

Every test task has at least 100 decoder-compatible mutants in its full pool.
Seven feasible original training tasks have fewer than 100 such mutants:

| Task | Compatible existing mutants | New mutants needed |
| --- | ---: | ---: |
| pep_001 | 53 | 47 |
| pep_002 | 8 | 92 |
| pep_018 | 0 | 100 |
| pep_024 | 10 | 90 |
| pep_069 | 47 | 53 |
| pep_098 | 55 | 45 |
| pep_603 | 53 | 47 |

The selected preparation policy is to retain the first 100 compatible entries
in existing file order, without ranking by MIC. Where fewer are available,
generate additional distinct random mutants satisfying all four checks:
standard alphabet, at most one cysteine, length at most 30, and similarity at
least 0.75 against the actual reference. The three replacements also require
new 100-entry initialization pools. This requires 474 new rows for the seven
shortage cases and 300 rows for the replacements, at most 774 new scoring rows.
All other selected rows retain the score for the exact unchanged sequence.

Generation must reserve the mandatory edits needed to remove unknown residues
and excess cysteines, then sample additional substitutions within the remaining
edit allowance. Simply applying the old sampler harder would waste attempts
for cases such as pep_018. A CPU-only construction check generated 100 distinct,
valid mutants for all seven shortage cases and all three proposed replacement
references, in 100-112 attempts per case. It made zero oracle calls, wrote no
production pools, and establishes only combinatorial validity, not VAE
reconstruction or MIC improvement.

Changed or newly generated sequences must never inherit the score of an old
sequence. They need APEX scoring before use. Prepared pools will be versioned
and shared across compared methods. Original histories and scored pools remain
intact. Reference IDs for the three replacements must be new IDs.

## Delegated recenter accounting decision

Retain the authors' decode-and-score recenter behavior after a joint VAE/GP
update triggered by ten consecutive non-improving acquisition rounds. Also
retain the corrected re-encoding of existing observations so GP coordinates
and unchanged measured sequences remain aligned after VAE weights change.

Use two stopping bounds:

- At most 5,000 acquisition proposals, including cached duplicates, in batches
  of at most 50. This permits at most 100 acquisition rounds.
- At most 5,000 fresh post-initialization APEX calls in total, shared by
  acquisition and recentering. Cache hits do not consume fresh-call capacity.

Recenter calls are bounded before scoring and logged separately with the exact
decoded sequence and measured score. The final batch or recenter operation
must be shortened when capacity is insufficient; there must be no overshoot.
If recentering consumes this shared capacity, acquisition can stop before its
100th round. This is a disclosed change from v4's fixed 100 acquisition rounds,
chosen to preserve its maximum fresh-evaluation cost while restoring the
original recenter mechanism. Cached observations must still update the GP and
appropriate counters. An interrupted recenter needs the same durable scored
batch and RNG recovery guarantees as acquisition.

The implementation and GPU verification of this revised campaign are not yet
complete. All production launch activity remains prohibited by the user's
latest instruction.
