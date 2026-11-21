#!/usr/bin/env python3
"""TaskSet CASH space + dummy TPE/SMAC BO."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from mindware.domains import get_adapter
from mindware.domains.dlhpo.taskset_space import (
    FAMILY_HP,
    TASKSET_FAMILIES,
    decode_taskset_config,
    space_from_taskset,
)
from mindware.domains.search import run_domain_search

try:
    import openbox  # noqa: F401

    HAS_OPENBOX = True
except ImportError:
    HAS_OPENBOX = False


class TestTaskSetCash(unittest.TestCase):
    def test_conditional_space_samples_one_family(self):
        cs = space_from_taskset()
        names = {hp.name for hp in cs.get_hyperparameters()}
        self.assertIn(FAMILY_HP, names)
        for fam in TASKSET_FAMILIES:
            self.assertTrue(any(n.startswith(f"{fam}:") for n in names), fam)
        cfg = cs.sample_configuration()
        payload = decode_taskset_config(cfg)
        self.assertIn(payload["optimizer_family"], TASKSET_FAMILIES)
        self.assertTrue(payload["hyperparameters"])
        family = payload["optimizer_family"]
        # Inactive families should not leak keys into the train_inner payload.
        for key in payload["hyperparameters"]:
            self.assertFalse(key.startswith("adam") or key.startswith("nadamw"))
        self.assertIn("learning_rate", payload["hyperparameters"])
        if family == "adam1p_wide_grid":
            self.assertEqual(set(payload["hyperparameters"]), {"learning_rate"})

    def test_dummy_tpe_and_smac(self):
        adapter = get_adapter("taskset", task={"task_id": "mlp_family_seed12"}, dummy=True)
        with tempfile.TemporaryDirectory() as td:
            tpe = run_domain_search(adapter, n_trials=8, seed=1, optimizer="tpe", output_dir=Path(td) / "tpe")
            self.assertEqual(tpe["n_trials"], 8)
            self.assertTrue(math.isfinite(float(tpe["best_objective"])))
            fams = {decode_taskset_config(o["config"])["optimizer_family"] for o in tpe["observations"]}
            self.assertGreaterEqual(len(fams), 1)
            if not HAS_OPENBOX:
                self.skipTest("openbox not installed (use hpo_benchmark env for SMAC)")
            smac = run_domain_search(adapter, n_trials=8, seed=1, optimizer="smac", output_dir=Path(td) / "smac")
            self.assertTrue(math.isfinite(float(smac["best_objective"])))

    def test_tpe_resume_matches_uninterrupted_configs(self):
        """Pickled Optuna state + history must continue the same sequence after a stop."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            full_dir = td / "full"
            part_dir = td / "part"
            adapter = get_adapter("taskset", task={"task_id": "mlp_family_seed12"}, dummy=True)
            full = run_domain_search(adapter, n_trials=8, seed=7, optimizer="tpe", output_dir=full_dir)
            adapter = get_adapter("taskset", task={"task_id": "mlp_family_seed12"}, dummy=True)
            part = run_domain_search(adapter, n_trials=4, seed=7, optimizer="tpe", output_dir=part_dir)
            self.assertTrue((part_dir / "optuna_study.pkl").is_file())
            self.assertTrue((part_dir / "history.json").is_file())
            adapter = get_adapter("taskset", task={"task_id": "mlp_family_seed12"}, dummy=True)
            resumed = run_domain_search(
                adapter,
                n_trials=8,
                seed=7,
                optimizer="tpe",
                output_dir=part_dir,
                resume_observations=part["observations"],
            )
            full_cfgs = [o["config"] for o in full["observations"]]
            res_cfgs = [o["config"] for o in resumed["observations"]]
            self.assertEqual(full_cfgs, res_cfgs)
            self.assertEqual(resumed.get("resumed_from"), 4)


if __name__ == "__main__":
    unittest.main()
