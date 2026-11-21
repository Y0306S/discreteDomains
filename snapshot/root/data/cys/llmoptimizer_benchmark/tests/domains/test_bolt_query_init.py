"""BayesQO-aligned query init / objective helpers."""

from __future__ import annotations

from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock

from mindware.domains.queryopt.bolt_execution import (
    BAO_CSV,
    bao_censored_csv_paths,
    bao_lolbo_init,
    load_bao_init,
    runtime_to_lolbo_score,
    worst_runtime_from_lolbo_scores,
    _csv_is_real,
)


class TestBoltQueryInit(TestCase):
    def test_runtime_to_lolbo_score_negates_runtime(self):
        self.assertAlmostEqual(runtime_to_lolbo_score(2.5), -2.5)

    def test_worst_runtime_from_lolbo_scores(self):
        ys = [-0.31, -100.0, -5.0]
        self.assertAlmostEqual(worst_runtime_from_lolbo_scores(ys), 100.0)

    def test_ceb_bao_init_loads_censoring(self):
        rows = load_bao_init("CEB_8A94")
        self.assertIsNotNone(rows)
        self.assertEqual(len(rows), 49)

    def test_ceb_bao_init_dedupes_corrupt_duplicate_plan_ids(self):
        # Raw CSV has extra rows for some CEB_11A* tasks from quoting corruption.
        rows = load_bao_init("CEB_11A114")
        self.assertIsNotNone(rows)
        self.assertEqual(len(rows), 49)
        enc, rt, cens = rows[0]
        self.assertIsInstance(enc, list)
        self.assertGreaterEqual(len(enc), 2)
        self.assertGreater(rt, 0)
        self.assertIn(cens, (0, 1))
        self.assertTrue(_csv_is_real(BAO_CSV))

    def test_bao_lolbo_init_preserves_censored_runtime_not_penalty(self):
        adapter = MagicMock()
        adapter.vae.encode_flat.side_effect = lambda enc: [0.1] * 64
        init = bao_lolbo_init(adapter, "CEB_8A94")
        self.assertIsNotNone(init)
        self.assertGreaterEqual(len(init["xs"]), 30)
        self.assertEqual(len(init["ys"]), len(init["xs"]))
        self.assertEqual(len(init["censoring"]), len(init["xs"]))
        self.assertGreater(init["worst_runtime_observed"], 0)
        cens_idx = next(i for i, c in enumerate(init["censoring"]) if c == 1)
        rows = load_bao_init("CEB_8A94")
        self.assertAlmostEqual(
            init["ys"][cens_idx],
            runtime_to_lolbo_score(rows[cens_idx][1]),
        )

    def test_bao_censored_paths_include_bolt_and_bayesqo(self):
        paths = bao_censored_csv_paths()
        self.assertGreaterEqual(len(paths), 2)
        self.assertTrue(any("query_plans" in str(p) for p in paths))
        self.assertTrue(any("BayesQO" in str(p) for p in paths))
        # BOLT copy may be absent; BayesQO copy may be git-lfs pointer only.
        self.assertTrue(any(p.is_file() for p in paths))
