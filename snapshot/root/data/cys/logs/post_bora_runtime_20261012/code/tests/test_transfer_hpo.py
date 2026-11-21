import copy
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from ConfigSpace import ConfigurationSpace, Constant, UniformIntegerHyperparameter
from openbox.utils.constants import SUCCESS, FAILED

from mindware.components.optimizers import ml_checkpoint
from mindware.components.optimizers import pt4opt_transfer_optimizer as transfer
from mindware.components.optimizers.transfer_hpo.protocol import acquisition_diagnostics, validate_dump
from mindware.components.optimizers.transfer_hpo.schema import cs_from_spec, schema_fingerprint

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmark_data/ml_train_80_transfer_v2"


class MemoryKD:
    def __init__(self):
        self.configs, self.perfs = [], []

    def update_history(self, observation, **kwargs):
        self.configs.append(observation.config)
        self.perfs.append(-observation.objectives[0])


class LedgerOptimizer(transfer._TransferHPOOptimizer):
    def _check_assets(self):
        self._set_assets({"fixture": "immutable"})

    def update_saver(self, *args):
        pass

    def _suggest_model(self, observed):
        raise RuntimeError("Deliberate worker failure")


class TransferTests(unittest.TestCase):
    def test_all_exported_schemas_round_trip(self):
        for group in json.loads((DATA / "coverage.json").read_text())["groups"]:
            dump = json.loads(Path(group["dump"]).read_text())
            with self.subTest(group=group["group"]):
                cs = cs_from_spec(dump["configspace"])
                self.assertEqual(schema_fingerprint(cs), dump["schema_fingerprint"])
                cs.check_configuration(cs.get_default_configuration())

    def test_source_boundary_and_alignment(self):
        dump = json.loads((DATA / "cls_adaboost.json").read_text())
        task = next(iter(dump["tasks"]))
        dump["tasks"] = {task: dump["tasks"][task]}
        validate_dump(dump)
        for mutation in ("task", "path", "labels", "direction"):
            bad = copy.deepcopy(dump)
            record = bad["tasks"][task]
            if mutation == "task":
                bad["tasks"]["cls/not_train80/adaboost"] = bad["tasks"].pop(task)
            elif mutation == "path":
                record["source_path"] = "/tmp/outside_allowed_history.json"
            elif mutation == "labels":
                record["y_minimize"].pop()
            else:
                record["y_maximize"][0] = 12345
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_dump(bad)

    def make_optimizer(self, directory, startup=3):
        space = ConfigurationSpace(seed=1)
        space.add_hyperparameters([Constant("algorithm", "adaboost"),
                                   UniformIntegerHyperparameter("value", 1, 100)])
        kd = MemoryKD()
        with patch.object(transfer.KnowledgeDatabase, "instance", return_value=kd):
            return LedgerOptimizer(SimpleNamespace(), space, "hpo", "holdout", output_dir=directory,
                                   evaluation_limit=2, inner_iter_num_per_iter=5,
                                   optimizer_params={"task_type": "cls", "n_startup_trials": startup},
                                   timestamp=time.time(), logging_enabled=False)

    def test_failures_count_and_durable_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            session = str(Path(directory) / "session")
            with patch.dict(os.environ, {"ML_SESSION_DIR": session, "ML_RUN_ID": "transfer_test"}):
                optimizer = self.make_optimizer(directory)
                ml_checkpoint.attach(optimizer)
                evaluated = []

                def evaluate(config, *args):
                    evaluated.append(dict(config))
                    return SimpleNamespace(config=config, objectives=[0.25], elapsed_time=0.01,
                                           trial_state=FAILED if len(evaluated) == 1 else SUCCESS)

                with patch.object(transfer, "wrapper_evaluater", side_effect=evaluate):
                    optimizer.iterate()
                    optimizer.iterate()
                self.assertEqual(len(evaluated), 2)
                self.assertEqual(optimizer.trial_cnt, 2)
                self.assertEqual(len(optimizer._observed()), 1)
                self.assertEqual(optimizer.perfs, [float("-inf"), -0.25])
                restored = self.make_optimizer(directory)
                self.assertEqual(ml_checkpoint.attach(restored), 2)
                with patch.object(transfer, "wrapper_evaluater") as oracle:
                    restored.iterate()
                    oracle.assert_not_called()

    def test_pending_proposal_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"ML_SESSION_DIR": str(Path(directory) / "session"),
                                         "ML_RUN_ID": "pending_test"}):
                optimizer = self.make_optimizer(directory)
                ml_checkpoint.attach(optimizer)
                config, note = optimizer._suggest()
                evidence = str(Path(directory) / "decisions/trial-000/attempt-original")
                proposal = {"config": dict(config), "note": note, "suggest_time": 0.01,
                            "diagnostics_dir": evidence}
                ml_checkpoint.prepare(optimizer, proposal)
                restored = self.make_optimizer(directory)
                ml_checkpoint.attach(restored)
                restored.inner_iter_num_per_iter = 1
                def evaluate(config, *args):
                    self.assertEqual(dict(config), proposal["config"])
                    return SimpleNamespace(config=config, objectives=[0.5], elapsed_time=0.01, trial_state=SUCCESS)
                with patch.object(LedgerOptimizer, "_suggest", side_effect=AssertionError("must reuse proposal")), \
                     patch.object(transfer, "wrapper_evaluater", side_effect=evaluate):
                    restored.iterate()
                self.assertEqual(restored.trial_cnt, 1)
                self.assertIsNone(ml_checkpoint.pending(restored))
                self.assertEqual(restored.traj_rows[0]["diagnostics_dir"], evidence)

    def test_model_failure_does_not_evaluate_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            optimizer = self.make_optimizer(directory, startup=1)
            optimizer.trial_cnt = 1
            optimizer.configs.append(optimizer.config_space.get_default_configuration())
            optimizer.perfs.append(0.2)
            with patch.object(transfer, "wrapper_evaluater") as oracle:
                with self.assertRaisesRegex(RuntimeError, "worker failure"):
                    optimizer.iterate()
                oracle.assert_not_called()
            self.assertEqual(optimizer.trial_cnt, 1)

    def test_worker_evidence_survives_success_failure_and_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            optimizer = self.make_optimizer(directory)
            optimizer.trial_cnt = 3

            def response(command, **kwargs):
                request = Path(command[command.index("--request") + 1])
                output = Path(command[command.index("--out") + 1])
                self.assertTrue(request.exists())
                kwargs["stdout"].write("worker trace\n")
                kwargs["stderr"].write("worker warning\n")
                output.write_text(json.dumps({"index": 0, "weights": [0.2, 0.8],
                                              "acquisition": {"values": [0.3]}}))
                return subprocess.CompletedProcess(command, 0)

            with patch.object(transfer.subprocess, "run", side_effect=response):
                payload = optimizer._run_worker(sys.executable, "fixture.py", directory, [], {"seed": 1})
                config = optimizer.config_space.get_default_configuration()
                optimizer._selected(payload, [config])
            first = Path(optimizer._proposal_diagnostics)
            self.assertEqual(json.loads((first / "response.json").read_text())["weights"], [0.2, 0.8])
            self.assertIn("worker warning", (first / "stderr.log").read_text())
            record = json.loads((first / "worker.json").read_text())
            self.assertEqual(record["selected_config"], dict(config))
            self.assertEqual(record["status"], "succeeded")
            self.assertGreaterEqual(record["elapsed_seconds"], 0)
            with self.assertRaises(RuntimeError):
                optimizer._selected({"index": 5}, [config])
            self.assertEqual(json.loads((first / "worker.json").read_text())["status"], "invalid_response")

            for label, effect in (
                    ("failed", subprocess.CompletedProcess([], 1)),
                    ("timeout", subprocess.TimeoutExpired("fixture", 0.1))):
                kwargs = {"side_effect": effect} if isinstance(effect, Exception) else {"return_value": effect}
                with patch.object(transfer.subprocess, "run", **kwargs), self.assertRaises(RuntimeError):
                    optimizer._run_worker(sys.executable, "fixture.py", directory, [], {"seed": 1})
                failed = Path(optimizer._proposal_diagnostics)
                self.assertNotEqual(first, failed)
                self.assertEqual(json.loads((failed / "worker.json").read_text())["status"], label)
                self.assertTrue((failed / "request.json").exists())
                self.assertTrue((failed / "stderr.log").exists())
            self.assertTrue((first / "response.json").exists())

    def test_acquisition_diagnostics_expose_zero_scores(self):
        diagnostics = acquisition_diagnostics("EI", [0.0, 0.0, 0.0])
        self.assertTrue(diagnostics["all_zero"])
        self.assertEqual(diagnostics["distinct_count"], 1)
        self.assertEqual(diagnostics["values"], [0.0, 0.0, 0.0])
        self.assertFalse(acquisition_diagnostics("TAF", [0, 0.2])["all_zero"])


if __name__ == "__main__":
    unittest.main()
