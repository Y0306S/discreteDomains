"""Regression tests for the peptide campaign's accounting and durable state."""

import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mindware.domains.protein import verified_bo as bo

spec = importlib.util.spec_from_file_location("peptide_campaign", ROOT / "tools/peptide_bo_campaign.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def fake_state():
    return SimpleNamespace(
        objective=SimpleNamespace(vae=torch.nn.Linear(2, 2), xs_to_scores_dict={"AAAA": -2.0}, num_calls=0),
        model=torch.nn.Linear(2, 1), c_models=[torch.nn.Linear(2, 1)],
        train_x=["AAAA", "AAAC"], train_y=torch.tensor([[-2.0], [-1.0]]),
        train_z=torch.zeros(2, 256), train_c=torch.tensor([[0.0], [0.1]]),
        top_k_xs=["AAAA"], top_k_scores=[-2.0], top_k_zs=[torch.zeros(1, 256)],
        top_k_cs=[torch.zeros(1, 1)], best_x_seen="AAAA", best_score_seen=-2.0,
        tr_state=SimpleNamespace(length=0.4, failure_counter=3),
        initial_model_training_complete=True, progress_fails_since_last_e2e=2,
        tot_num_e2e_updates=1, new_best_found=False, k=1000, bsz=2, acq_func="ts")


class ProtocolTests(unittest.TestCase):
    def test_duplicates_are_charged_as_proposals_but_scored_once(self):
        queried = []
        def oracle(xs):
            queried.extend(xs)
            return np.array([-3.0] * len(xs))
        obj = SimpleNamespace(xs_to_scores_dict={"AAAA": -2.0}, num_calls=0,
            query_oracle=oracle, compute_constraints=lambda xs: torch.zeros(len(xs), 1))
        scored = bo.score_batch(obj, ["AAAA", "AAAC", "AAAC"])
        self.assertEqual(queried, ["AAAC"])
        self.assertEqual(scored["cache_hits"], [True, False, True])
        self.assertEqual(obj.num_calls, 1)
        self.assertEqual(scored["scores"], [-2, -3, -3])
        bo.score_batch(obj, ["AAAC"])
        self.assertEqual(obj.num_calls, 1)

    def test_nonfinite_scores_do_not_enter_cache(self):
        obj = SimpleNamespace(xs_to_scores_dict={}, num_calls=0,
            query_oracle=lambda xs: np.array([np.nan]),
            compute_constraints=lambda xs: torch.zeros(len(xs), 1))
        with self.assertRaises(ValueError):
            bo.score_batch(obj, ["AAAA"])
        self.assertEqual(obj.xs_to_scores_dict, {})

    def test_partial_final_batch(self):
        self.assertEqual([bo.next_batch_size({"proposals": n}, 11, 4) for n in [0, 4, 8, 11]], [4, 4, 3, 0])

    def test_center_and_incumbent_exclude_infeasible_best(self):
        state = fake_state()
        def generate(**kw):
            self.assertEqual(kw["X"].shape[0], 1)
            self.assertEqual(kw["Y"].item(), -2)
            self.assertIs(kw["model"], state.model)
            return torch.zeros(2, 256)
        state.objective.vae_decode = lambda z: ["AAAA"] * len(z)
        with patch.dict(sys.modules, {"lolbo.utils.bo_utils.turbo": SimpleNamespace(generate_batch=generate)}):
            bo.propose(state, 2)
        bo.unique_top_k(state)
        self.assertEqual(state.best_x_seen, "AAAA")
        self.assertEqual(state.best_score_seen, -2)

    def test_snapshot_restores_models_rng_and_trust_region(self):
        state = fake_state()
        identity = {"task": "pep_000"}
        saved = bo.snapshot(state, identity, {"proposals": 2}, [{"sequence": "AAAA"}])
        expected = [random.random(), np.random.rand(), torch.rand(1).item()]
        state.model.weight.data.fill_(99)
        state.objective.vae.weight.data.fill_(99)
        state.c_models[0].weight.data.fill_(99)
        state.tr_state.failure_counter = 30
        progress, rows = bo.restore(state, saved, identity)
        self.assertEqual(expected, [random.random(), np.random.rand(), torch.rand(1).item()])
        self.assertTrue(torch.equal(state.model.weight, saved["model"]["weight"]))
        self.assertTrue(torch.equal(state.objective.vae.weight, saved["vae"]["weight"]))
        self.assertTrue(torch.equal(state.c_models[0].weight, saved["constraints"][0]["weight"]))
        self.assertEqual(state.tr_state.failure_counter, 3)
        self.assertEqual(progress["proposals"], 2)
        with self.assertRaises(ValueError):
            bo.restore(state, saved, {"task": "pep_001"})

    def test_bad_latent_dimensions_fail(self):
        state = fake_state()
        state.train_z = torch.zeros(2, 255)
        with self.assertRaises(ValueError):
            bo.validate_state(state)

    def test_impossible_reference_bounds(self):
        for reference in ("CYCRRRFCVCV", "GICYCICGRGICRCICGR", "GXXNNXXXXXXNXXNXXGY"):
            self.assertGreater(bo.required_decoder_edits(reference), len(reference) // 4)
        self.assertLessEqual(bo.required_decoder_edits("ILAGFLLSNNIL"), 3)

    def test_rotation_never_removes_latest_before_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            bo.atomic_torch(path, {"round": 1})
            replace = bo.os.replace
            def fail_publish(src, dst):
                if Path(dst) == path:
                    self.assertTrue(path.exists())
                    raise OSError("simulated interruption")
                return replace(src, dst)
            with patch.object(bo.os, "replace", side_effect=fail_publish):
                with self.assertRaises(OSError):
                    bo.atomic_torch(path, {"round": 2}, rotate=True)
            self.assertEqual(torch.load(path, weights_only=False)["round"], 1)

    def test_pool_filtering_keeps_scores_and_reference_blocks_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "xs").write_text("CCAA\nCAAA\nAAAA\nCCCC\nAACD\nAADD\nAADA\nAACC\n")
            (root / "ys").write_text("-1\n-2\n-3\n-4\n-5\n-6\n-7\n-8\n")
            manifest = dict(ref_start=0, n_refs=2, rows=8, num_mutations=4,
                init_path=str(root / "xs"), scores_path=str(root / "ys"),
                per_ref=[dict(ref_index=0, reference="AAAA"), dict(ref_index=1, reference="AADD")])
            bo.atomic_json(root / "manifest.json", manifest)
            recipe = dict(pool_manifest=str(root / "manifest.json"), initialization=2,
                          similarity=0.0, init_policy="first_valid")
            pools, audit = bo.collect_pools(recipe)
            self.assertEqual(pools["pep_000"]["scores"], [-2, -3])
            self.assertEqual(pools["pep_001"]["reference"], "AADD")
            self.assertEqual(pools["pep_001"]["scores"], [-5, -6])
            self.assertEqual(audit[0]["replaced_prefix_rows"], 1)

    def test_archive_recovery_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "pep_000"
            directory.mkdir()
            identity = {"task": "pep_000"}
            history = dict(identity=identity, status="complete", initialization=2,
                proposals=1, proposal_budget=1, fresh_evaluations=1, cache_hits=0,
                observations=[{}, {}, {}])
            bo.atomic_json(directory / "history.json", history)
            bo.atomic_json(directory / "progress.json", {})
            bo.atomic_torch(directory / "checkpoint.pt", {})
            bo.atomic_torch(directory / "checkpoint.prev.pt", {})
            with patch.object(campaign, "cleanup_archived", side_effect=OSError("interrupted cleanup")):
                with self.assertRaises(OSError):
                    campaign.archive_run(directory, identity)
            archive = directory.with_suffix(".tar.gz")
            self.assertTrue(archive.exists())
            campaign.cleanup_archived(directory, archive, identity)
            self.assertFalse(directory.exists())
            campaign.cleanup_archived(directory, archive, identity)
            with self.assertRaises(ValueError):
                campaign.verify_archive(archive, {"task": "pep_001"})

    def test_preflight_must_match_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bo.atomic_json(root / "recipe.json", {"format": bo.FORMAT})
            bo.atomic_json(root / "preflight_passed.json", {"recipe_sha256": "wrong"})
            with self.assertRaises(ValueError):
                campaign.verify_preflight(root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
