#!/usr/bin/env python3
"""TaskSet CASH space + dummy TPE/SMAC BO."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
    def test_cli_rejects_mixed_evaluation_protocols_before_skip(self):
        from examples.benchmark.domains import run_taskset_bo
        from mindware.domains.dlhpo.taskset_oracle import DEFAULT_OBJECTIVE_METRIC, bo_protocol
        from mindware.domains.history import observation, write_openbox_history

        with tempfile.TemporaryDirectory() as td:
            for method in ("random", "tpe", "lolbo"):
                with self.subTest(method=method):
                    out = Path(td) / "mlp_family_seed12" / method / "seed1"
                    info = {
                        "oracle": "train_inner", "optimizer_backend": "lolbo_no_vae_cash",
                        "objective_metric": DEFAULT_OBJECTIVE_METRIC,
                        "protocol": {"training_steps": 100, "eval_every_n": 50, "num_replicas": 1},
                    }
                    obs = observation(dict(space_from_taskset().get_default_configuration()), 0.5,
                                      extra={"info": info})
                    args = ["--optimizer", method, "--task-ids", "mlp_family_seed12",
                            "--n-trials", "1", "--no-dummy", "--output-dir", td]
                    with patch.object(run_taskset_bo, "run_domain_search") as search:
                        write_openbox_history(out / "history.json", task_id="mlp_family_seed12",
                                              observations=[obs])
                        with self.assertRaisesRegex(ValueError, "evaluation protocol"):
                            run_taskset_bo.main(args)
                        info["protocol"] = bo_protocol()
                        info["objective_metric"] = "mean_normalized_valid_outer"
                        write_openbox_history(out / "history.json", task_id="mlp_family_seed12",
                                              observations=[obs])
                        with self.assertRaisesRegex(ValueError, "evaluation protocol"):
                            run_taskset_bo.main(args)
                        info["objective_metric"] = DEFAULT_OBJECTIVE_METRIC
                        write_openbox_history(out / "history.json", task_id="mlp_family_seed12",
                                              observations=[obs])
                        self.assertEqual(run_taskset_bo.main(args), 0)
                        search.assert_not_called()

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
