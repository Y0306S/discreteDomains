"""Tests for BOLT peptide constraint helpers."""

import random

from mindware.domains.protein.bolt_constraints import (
    bolt_reference_sequence,
    is_feasible,
    propose_bolt_feasible_seq,
    random_feasible_mutant,
    repair_to_feasible,
)


def test_bolt_reference_sequence_seed0():
    ref = bolt_reference_sequence(constraint_index=0)
    assert ref == "RRYYEQLEQASRKGNRGFRR"


def test_repair_makes_infeasible_decode_feasible():
    ref = bolt_reference_sequence(constraint_index=0)
    bad = "WWWWWWWWWWWWWWWWWWWW"
    fixed = repair_to_feasible(bad, ref, min_similarity=0.75)
    assert is_feasible(fixed, ref, min_similarity=0.75)


def test_random_mutant_stays_feasible():
    ref = bolt_reference_sequence(constraint_index=0)
    start = "RRTYFQLEQASRKGNRGFRR"
    for seed in range(5):
        m = random_feasible_mutant(start, ref, min_similarity=0.75, rng=random.Random(seed))
        assert is_feasible(m, ref, min_similarity=0.75)


def test_propose_bolt_feasible_seq_avoids_seen():
    ref = bolt_reference_sequence(constraint_index=0)
    anchor = "RRTYFQLEQASRKGNRGFRR"
    seen = {
        propose_bolt_feasible_seq(
            decoded="MMMMMMMMMMMMMMMMMMMM",
            reference=ref,
            anchor=anchor,
            best=None,
            min_similarity=0.75,
            rng=random.Random(0),
        )
    }
    out = propose_bolt_feasible_seq(
        decoded="MMMMMMMMMMMMMMMMMMMM",
        reference=ref,
        anchor=anchor,
        best=None,
        min_similarity=0.75,
        rng=random.Random(1),
        seen=seen,
        max_novel_attempts=32,
    )
    assert out not in seen
    assert is_feasible(out, ref, min_similarity=0.75)


def test_propose_bolt_feasible_seq_from_random_decode():
    ref = bolt_reference_sequence(constraint_index=0)
    anchor = "RRTYFQLEQASRKGNRGFRR"
    out = propose_bolt_feasible_seq(
        decoded="MMMMMMMMMMMMMMMMMMMM",
        reference=ref,
        anchor=anchor,
        best=None,
        min_similarity=0.75,
        rng=random.Random(0),
    )
    assert is_feasible(out, ref, min_similarity=0.75)
