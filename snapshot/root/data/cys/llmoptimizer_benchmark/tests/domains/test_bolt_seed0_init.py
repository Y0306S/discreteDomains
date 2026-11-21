"""Tests for BOLT seed_0 init pool loading."""

from __future__ import annotations

from mindware.domains.protein.bolt_seed0_init import (
    bolt_maximize_score_to_mic,
    load_seed0_pool,
    total_trial_budget,
)


def test_load_seed0_pool_shape():
    seqs, scores = load_seed0_pool(limit=10)
    assert len(seqs) == 10
    assert len(scores) == 10
    assert all(isinstance(s, str) and s for s in seqs)
    assert all(s < 0 for s in scores)


def test_score_to_mic_conversion():
    assert bolt_maximize_score_to_mic(-102.5) == 102.5


def test_total_trial_budget():
    assert total_trial_budget(n_init_points=1000, n_post_init_trials=500) == 1500
