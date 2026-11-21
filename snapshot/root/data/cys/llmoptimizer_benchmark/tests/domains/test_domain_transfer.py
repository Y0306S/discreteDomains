#!/usr/bin/env python3
"""Unit tests for domain adapters, sequence VAE, converters, and search loop."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ConfigSpace import ConfigurationSpace

from mindware.components.optimizers.hybrid_selector import HybridSelector
from mindware.domains import get_adapter
from mindware.domains.base import dummy_result, latent_config_space, z_from_config
from mindware.domains.history import observation, sort_improving, write_openbox_history
from mindware.domains.protein.features import extra_from_seq
from mindware.domains.queryopt.sql_parse import parse_tables, repair_order
from mindware.domains.search import run_domain_search
from mindware.domains.sequence_vae import AA_ALPHABET, SequenceVAE, train_sequence_vae


class TestSequenceVAE(unittest.TestCase):
    def test_encode_decode_alphabet(self):
        seqs = ["ACDEFGHIKLMNPQRSTVWY", "GGGGAAAAKKKK", "RRYYEQLEQASRKGNRGFRR", "KLKLKLKLKL"]
        vae = train_sequence_vae(seqs, list(AA_ALPHABET), latent_dim=16, max_len=24, steps=8, seed=0)
        z = vae.encode(seqs[0])
        self.assertEqual(len(z), 16)
        decoded = vae.decode_string(z)
        self.assertTrue(decoded)
        self.assertTrue(all(c in AA_ALPHABET for c in decoded))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "vae.pkl"
            vae.save(path)
            loaded = SequenceVAE.load(path)
            self.assertEqual(len(loaded.encode(seqs[1])), 16)


class TestAdapters(unittest.TestCase):
    def test_protein_dummy_loop(self):
        adapter = get_adapter("protein", task={"task_id": "toy"}, dummy=True)
        cs = adapter.search_space()
        self.assertIsInstance(cs, ConfigurationSpace)
        cfg = cs.get_default_configuration()
        result = adapter.evaluate_config(cfg)
        self.assertTrue(math_isfinite(result.objective))
        self.assertTrue(str(result.artifact))
        self.assertIn("length_bin", adapter.semantic_keys())

    def test_query_parse_and_repair(self):
        sql = """
        SELECT COUNT(*) FROM title as t, kind_type as kt, cast_info as ci
        WHERE t.id = ci.movie_id AND t.kind_id = kt.id
        """
        pairs = parse_tables(sql)
        aliases = [a for _, a in pairs]
        self.assertGreaterEqual(len(aliases), 3)
        broken = ["ci", "missing", "t"]
        fixed = repair_order(broken, aliases)
        self.assertEqual(sorted(fixed), sorted(aliases))
        adapter = get_adapter("query", task={"task_id": "q1", "sql": sql}, dummy=True)
        res = adapter.evaluate_config(adapter.search_space().get_default_configuration())
        self.assertTrue(math_isfinite(res.objective))
        self.assertEqual(len(res.artifact), len(aliases))

    def test_dl_named_space(self):
        adapter = get_adapter("dl", task={"task_id": "smb-000"}, dummy=True)
        names = {hp.name for hp in adapter.search_space().get_hyperparameters()}
        self.assertIn("embedding_lr", names)
        self.assertIn("weight_decay", names)
        res = adapter.evaluate_config(adapter.search_space().get_default_configuration())
        self.assertTrue(math_isfinite(res.objective))


class TestConvertersAndSearch(unittest.TestCase):
    def test_fake_bo_history_and_search(self):
        seq = "RRYYEQLEQASRKGNRGFRR"
        z = [0.1] * 16
        cfg = {f"z_{i}": z[i] for i in range(16)}
        cfg.update(extra_from_seq(seq))
        rows = [
            observation(cfg, 1.0, extra={"source": "library_pairs"}),
            observation({**cfg, "z_0": 0.2}, 0.8, extra={"source": "library_pairs"}),
            observation({**cfg, "z_0": -0.3}, 1.4, extra={"source": "library_pairs"}),
            observation({**cfg, "z_1": 0.4}, 0.5, extra={"source": "library_pairs"}),
        ]
        ordered = sort_improving(rows, minimize=True)
        self.assertEqual(len(ordered), 4)
        with tempfile.TemporaryDirectory() as td:
            path = write_openbox_history(Path(td) / "h.json", task_id="toy", observations=ordered)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["task_id"], "toy")
            self.assertEqual(len(payload["observations"]), 4)

        adapter = get_adapter("protein", task={"task_id": "toy"}, dummy=True)
        out = run_domain_search(adapter, n_trials=8, seed=1, optimizer="random")
        self.assertEqual(len(out["traj"]), 8)
        self.assertTrue(math_isfinite(out["best_objective"]))

        out2 = run_domain_search(adapter, n_trials=6, seed=1, optimizer="ptllm_cmaes", llm_prob=0.7)
        self.assertEqual(len(out2["traj"]), 6)
        sources = {r["source"] for r in out2["traj"]}
        self.assertTrue(sources & {"llm", "llm_warm_start", "cmaes", "tpe"})

    def test_hybrid_bo_arm_cmaes(self):
        import numpy as np

        sel = HybridSelector(mode="bernoulli", llm_prob=0.0, bo_arm="cmaes", rng=np.random.RandomState(0))
        arm = sel.choose(0, 10, higher_is_better=False)
        self.assertEqual(arm, "cmaes")
        sel.update("cmaes", 1.0, higher_is_better=True, beat_incumbent=True)
        self.assertEqual(sel.n["cmaes"], 1)


class TestBoltHelpers(unittest.TestCase):
    def test_left_deep_flat_and_roundtrip_names(self):
        from mindware.domains.bolt_vae import flat_to_table_names, left_deep_flat
        from mindware.domains.queryopt.ceb import IMDB_TABLES

        flat = left_deep_flat([3, 1, 7])
        self.assertEqual(len(flat) % 5, 0)
        names = flat_to_table_names(flat, IMDB_TABLES)
        self.assertEqual(names[0], IMDB_TABLES[3])
        self.assertEqual(names[-1], IMDB_TABLES[7])

    def test_ceb_sql_path(self):
        from mindware.domains.queryopt.ceb import load_task_ids, sql_path_for_task

        ids = load_task_ids(limit=5)
        if not ids:
            self.skipTest("CEB task list missing")
        path = sql_path_for_task(ids[0])
        self.assertIsNotNone(path)
        self.assertTrue(path.is_file())


def math_isfinite(x: float) -> bool:
    import math

    return math.isfinite(float(x))


if __name__ == "__main__":
    unittest.main()
#!/usr/bin/env python3
"""Unit tests for domain adapters, sequence VAE, converters, and search loop."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ConfigSpace import ConfigurationSpace

from mindware.components.optimizers.hybrid_selector import HybridSelector
from mindware.domains import get_adapter
from mindware.domains.base import dummy_result, latent_config_space, z_from_config
from mindware.domains.history import observation, sort_improving, write_openbox_history
from mindware.domains.protein.features import extra_from_seq
from mindware.domains.queryopt.sql_parse import parse_tables, repair_order
from mindware.domains.search import run_domain_search
from mindware.domains.sequence_vae import AA_ALPHABET, SequenceVAE, train_sequence_vae


class TestSequenceVAE(unittest.TestCase):
    def test_encode_decode_alphabet(self):
        seqs = ["ACDEFGHIKLMNPQRSTVWY", "GGGGAAAAKKKK", "RRYYEQLEQASRKGNRGFRR", "KLKLKLKLKL"]
        vae = train_sequence_vae(seqs, list(AA_ALPHABET), latent_dim=16, max_len=24, steps=8, seed=0)
        z = vae.encode(seqs[0])
        self.assertEqual(len(z), 16)
        decoded = vae.decode_string(z)
        self.assertTrue(decoded)
        self.assertTrue(all(c in AA_ALPHABET for c in decoded))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "vae.pkl"
            vae.save(path)
            loaded = SequenceVAE.load(path)
            self.assertEqual(len(loaded.encode(seqs[1])), 16)


class TestAdapters(unittest.TestCase):
    def test_protein_dummy_loop(self):
        adapter = get_adapter("protein", task={"task_id": "toy"}, dummy=True)
        cs = adapter.search_space()
        self.assertIsInstance(cs, ConfigurationSpace)
        cfg = cs.get_default_configuration()
        result = adapter.evaluate_config(cfg)
        self.assertTrue(math_isfinite(result.objective))
        self.assertTrue(str(result.artifact))
        self.assertIn("length_bin", adapter.semantic_keys())

    def test_query_parse_and_repair(self):
        sql = """
        SELECT COUNT(*) FROM title as t, kind_type as kt, cast_info as ci
        WHERE t.id = ci.movie_id AND t.kind_id = kt.id
        """
        pairs = parse_tables(sql)
        aliases = [a for _, a in pairs]
        self.assertGreaterEqual(len(aliases), 3)
        broken = ["ci", "missing", "t"]
        fixed = repair_order(broken, aliases)
        self.assertEqual(sorted(fixed), sorted(aliases))
        adapter = get_adapter("query", task={"task_id": "q1", "sql": sql}, dummy=True)
        res = adapter.evaluate_config(adapter.search_space().get_default_configuration())
        self.assertTrue(math_isfinite(res.objective))
        self.assertEqual(len(res.artifact), len(aliases))

    def test_dl_named_space(self):
        adapter = get_adapter("dl", task={"task_id": "smb-000"}, dummy=True)
        names = {hp.name for hp in adapter.search_space().get_hyperparameters()}
        self.assertIn("embedding_lr", names)
        self.assertIn("weight_decay", names)
        res = adapter.evaluate_config(adapter.search_space().get_default_configuration())
        self.assertTrue(math_isfinite(res.objective))


class TestConvertersAndSearch(unittest.TestCase):
    def test_fake_bo_history_and_search(self):
        seq = "RRYYEQLEQASRKGNRGFRR"
        z = [0.1] * 16
        cfg = {f"z_{i}": z[i] for i in range(16)}
        cfg.update(extra_from_seq(seq))
        rows = [
            observation(cfg, 1.0, extra={"source": "library_pairs"}),
            observation({**cfg, "z_0": 0.2}, 0.8, extra={"source": "library_pairs"}),
            observation({**cfg, "z_0": -0.3}, 1.4, extra={"source": "library_pairs"}),
            observation({**cfg, "z_1": 0.4}, 0.5, extra={"source": "library_pairs"}),
        ]
        ordered = sort_improving(rows, minimize=True)
        self.assertEqual(len(ordered), 4)
        with tempfile.TemporaryDirectory() as td:
            path = write_openbox_history(Path(td) / "h.json", task_id="toy", observations=ordered)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["task_id"], "toy")
            self.assertEqual(len(payload["observations"]), 4)

        adapter = get_adapter("protein", task={"task_id": "toy"}, dummy=True)
        out = run_domain_search(adapter, n_trials=8, seed=1, optimizer="random")
        self.assertEqual(len(out["traj"]), 8)
        self.assertTrue(math_isfinite(out["best_objective"]))

        out2 = run_domain_search(adapter, n_trials=6, seed=1, optimizer="ptllm_cmaes", llm_prob=0.7)
        self.assertEqual(len(out2["traj"]), 6)
        sources = {r["source"] for r in out2["traj"]}
        self.assertTrue(sources & {"llm", "llm_warm_start", "cmaes", "tpe"})

    def test_hybrid_bo_arm_cmaes(self):
        import numpy as np

        sel = HybridSelector(mode="bernoulli", llm_prob=0.0, bo_arm="cmaes", rng=np.random.RandomState(0))
        arm = sel.choose(0, 10, higher_is_better=False)
        self.assertEqual(arm, "cmaes")
        sel.update("cmaes", 1.0, higher_is_better=True, beat_incumbent=True)
        self.assertEqual(sel.n["cmaes"], 1)


class TestBoltHelpers(unittest.TestCase):
    def test_left_deep_flat_and_roundtrip_names(self):
        from mindware.domains.bolt_vae import flat_to_table_names, left_deep_flat
        from mindware.domains.queryopt.ceb import IMDB_TABLES

        flat = left_deep_flat([3, 1, 7])
        self.assertEqual(len(flat) % 5, 0)
        names = flat_to_table_names(flat, IMDB_TABLES)
        self.assertEqual(names[0], IMDB_TABLES[3])
        self.assertEqual(names[-1], IMDB_TABLES[7])

    def test_ceb_sql_path(self):
        from mindware.domains.queryopt.ceb import load_task_ids, sql_path_for_task

        ids = load_task_ids(limit=5)
        if not ids:
            self.skipTest("CEB task list missing")
        path = sql_path_for_task(ids[0])
        self.assertIsNotNone(path)
        self.assertTrue(path.is_file())


def math_isfinite(x: float) -> bool:
    import math

    return math.isfinite(float(x))


if __name__ == "__main__":
    unittest.main()
