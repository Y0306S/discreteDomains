"""Behavioral checks for strict pools, shared budgets and durable recenter events."""

import ast
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from mindware.domains.protein import verified_bo as bo
from mindware.domains.protein import peptide_v5 as v5
import peptide_bo_v5 as runner


class V5Tests(unittest.TestCase):
    def test_mutants_satisfy_all_constraints_even_with_excess_cysteines_or_X(self):
        for ref in ["PTTCTWLPTDYHQCNCHVFW", "WGFCWNVCVYRNGVRVCHRRCN", "KLCGETCFKFKCYTPGCSCSYPFCK",
                    "WLKKLKKXLKKLXKK", "KTLKIIRLLF", "RQKNHGIHFRVLAKALR", "RMARNLVRYVQGLKKKKVI"]:
            sequences = v5.mutants(ref, 100)
            self.assertEqual(len(set(sequences)), 100)
            self.assertEqual(sequences, v5.mutants(ref, 100))
            for sequence in sequences:
                bo.validate_sequence(sequence)
                self.assertGreaterEqual(bo.similarity(sequence, ref), 0.75)
                self.assertNotEqual(sequence, ref)

    def test_impossible_references_are_not_silently_repaired(self):
        for ref in ["CYCRRRFCVCV", "GICYCICGRGICRCICGR", "GXXNNXXXXXXNXXNXXGY"]:
            with self.assertRaises(ValueError):
                v5.mutants(ref, 100)

    def test_shared_budget_counts_distinct_new_sequences_and_keeps_cached_results(self):
        sequences = ["AAAA", "AAAC", "AAAC", "AACA", "AACC"]
        self.assertEqual(v5.bounded_prefix(sequences, {"AAAA": -1}, 1), sequences[:3])
        self.assertEqual(v5.bounded_prefix(sequences, {"AAAA": -1}, 0), ["AAAA"])

    def test_acquisition_and_recenter_totals_are_checked_together(self):
        recipe = dict(initialization=2, proposals=5, fresh_budget=3)
        progress = dict(proposals=2, recenter_proposals=3, fresh_evaluations=3, cache_hits=2,
                        acquisition_fresh=1, recenter_fresh=2)
        v5.check_accounting(recipe, progress, [{}] * 7)
        progress["fresh_evaluations"] = 4
        with self.assertRaises(ValueError):
            v5.check_accounting(recipe, progress, [{}] * 7)

    def test_CPU_adaptation_changes_device_placement_without_detaching_gradients(self):
        tree = v5.CPUTransformer().visit(ast.parse("x = model.cuda(); y = torch.load(path); z = torch.device('cuda'); loss.backward()"))
        ast.fix_missing_locations(tree)
        code = ast.unparse(tree)
        self.assertIn("model.cpu()", code)
        self.assertIn("map_location='cpu'", code)
        self.assertIn("loss.backward()", code)
        self.assertNotIn("detach", code)

    def test_remote_forward_keeps_gradient_update_local(self):
        from mindware.domains.protein.peptide_rpc import RemoteObjective
        weight = torch.tensor(2.0, requires_grad=True)
        obj = SimpleNamespace(vae_forward=lambda xs: (weight * torch.ones(1, 256), weight**2))
        remote = RemoteObjective(obj, "http://unused", "task")
        with patch.object(remote, "request", side_effect=AssertionError("No RPC during training")):
            z, loss = obj.vae_forward(["AAAA"])
            (loss + z.sum()).backward()
        self.assertIsNotNone(weight.grad)
        self.assertNotEqual(weight.grad.item(), 0)

    def test_scored_pending_replay_does_not_call_oracle(self):
        spec = importlib.util.spec_from_file_location("old_tests", ROOT / "tests/domains/test_verified_peptide_bo.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        state = module.fake_state()
        state.tr_state.restart_triggered = False
        def update(z, y, xs, cs, acquisition):
            state.train_x += xs
            state.train_y = torch.cat([state.train_y, y.reshape(-1, 1)])
            state.train_z = torch.cat([state.train_z, z])
            state.train_c = torch.cat([state.train_c, cs])
        state.update_next = update
        progress = dict(rounds=0, proposals=0, recenter_proposals=0, fresh_evaluations=0, cache_hits=0,
            acquisition_fresh=0, recenter_fresh=0, events=0, seconds=0.0)
        scored = dict(sequences=["AACA"], scores=[-3.0], constraints=torch.zeros(1, 1),
            cache_hits=[False], new_scores={"AACA": -3.0}, fresh_evaluations=1, rng=bo.capture_rng())
        identity = dict(task="pep_000")
        pending = dict(phase="acquisition", z=torch.zeros(1, 256), scored=scored)
        recipe = dict(initialization=2, proposals=1, fresh_budget=1, references={"pep_000": "AAAA"}, similarity=0.75)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bo.atomic_torch(root / "pending.pt", pending)
            with patch.object(bo, "score_batch", side_effect=AssertionError("Must use saved score")):
                runner.commit_pending(state, identity, progress, [{}, {}], pending, root, recipe)
        self.assertEqual(state.objective.num_calls, 1)
        self.assertEqual(progress["fresh_evaluations"], 1)
        self.assertEqual(state.train_x[-1], "AACA")


if __name__ == "__main__":
    unittest.main(verbosity=2)
