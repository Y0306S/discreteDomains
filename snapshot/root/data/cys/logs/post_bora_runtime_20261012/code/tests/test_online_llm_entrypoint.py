"""Offline CLI and real CASH evaluator checks; language models are fixtures."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from sklearn.datasets import make_classification

from mindware import CASH
from mindware.components.optimizers.bora_optimizer import BORAOptimizer
from mindware.components.feature_engineering.transformation_graph import DataNode
from mindware.components.optimizers.transfer_hpo.schema import build_hpo_cs
from mindware.components.utils.constants import CLASSIFICATION, NUMERICAL
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase
from mindware.modules.base import BaseAutoML
import run_toy_categorical_optimizers as runner


class DefaultRuntime:
    def __init__(self, config):
        self.config = config
        self.observed = []

    def propose(self, prompt, step):
        return {"raw_text": json.dumps(self.config), "tokens": [42], "features": [1.0],
                "gradient_step": False, "policy_version": 0}

    def observe(self, features, reward):
        self.observed.append(reward)


class EntryPointTests(unittest.TestCase):
    def test_bolt_cash_dispatch_performs_a_real_evaluation(self):
        from mindware.components.optimizers.bolt_ml_optimizer import BOLTMLOptimizer
        x, y = make_classification(n_samples=80, n_features=6, random_state=3)
        node = DataNode([x, y], feature_type=[NUMERICAL] * 6, task_type=CLASSIFICATION)
        config = dict(build_hpo_cs("cls", "random_forest").get_default_configuration())
        client = SimpleNamespace(complete_messages=Mock(return_value=json.dumps(config)))
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ), \
                patch.object(KnowledgeDatabase, "_instance", None):
            os.environ.pop("ML_SESSION_DIR", None)
            cash = CASH(include_algorithms=["random_forest"], task_type=CLASSIFICATION,
                metric="bal_acc", data_node=node, optimizer="bolt_ml", amount_of_resource=100,
                time_limit=None, per_run_time_limit=2400, output_dir=directory,
                resampling_params={"test_size": .25}, optimizer_params={
                    "llm_client": client, "extractor_client": SimpleNamespace(calls=0),
                    "task_type": "cls", "recommendation_default_fallback": True,
                    "dataset_profile": {"metric": "bal_acc"}, "lolbo_device": "cpu"})
            self.assertIsInstance(cash.optimizer, BOLTMLOptimizer)
            cash.optimizer.iterate()
            self.assertEqual(cash.optimizer.trial_cnt, 1)
            self.assertEqual(cash.optimizer.traj_rows[0]["evaluation_status"], "SUCCESS")
            self.assertEqual(cash.optimizer.bo_state()["observations"], 1)
            self.assertEqual(client.complete_messages.call_count, 1)

    def test_bora_runner_saves_one_evaluation_and_resume_does_not_refit(self):
        x, y = make_classification(n_samples=80, n_features=6, random_state=3)
        node = DataNode([x, y], feature_type=[NUMERICAL] * 6, task_type=CLASSIFICATION)
        config = dict(build_hpo_cs("cls", "random_forest").get_default_configuration())
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ), \
                patch.object(KnowledgeDatabase, "_instance", None):
            with patch("sys.argv", ["run", "--optimizer", "bora", "--dataset", "fixture",
                    "--algorithm", "random_forest", "--n-trials", "1", "--time-limit", "none",
                    "--per-run-time-limit", "30", "--skip-test-eval",
                    "--extractor-max-tokens", "512", "--extractor-max-calls", "2",
                    "--extractor-timeout", "10", "--output-dir", directory + "/outputs",
                    "--history-dir", directory + "/history", "--session-dir", directory + "/session"]):
                args = runner.parse_args()
            with patch.object(BORAOptimizer, "_prompt", return_value=json.dumps(config)) as propose, \
                    patch.object(BaseAutoML, "_refit_config", side_effect=AssertionError("Extra fit")) as refit:
                first = runner.run_one_optimizer(args, "bora", node, node, ["random_forest"], {})
                self.assertTrue(first["progress"]["clean_complete"])
                self.assertFalse(first["model_info"]["fitted_model_saved"])
                self.assertIsNone(first["model_info"]["best"][2])
                history = json.loads(Path(first["histories"]["random_forest"]).read_text())
                self.assertEqual(len(history["observations"]), 1)
                session = Path(first["session_dir"])
                saved_results = {str(p): p.read_bytes() for p in session.glob("evaluations/*/result.pkl")}
                self.assertEqual(len(saved_results), 1)
                args.resume = True
                restored = runner.run_one_optimizer(args, "bora", node, node, ["random_forest"], {})
                self.assertEqual(restored["run_id"], first["run_id"])
                self.assertEqual(restored["progress"], first["progress"])
                history = json.loads(Path(restored["histories"]["random_forest"]).read_text())
                self.assertEqual(len(history["observations"]), 1)
                self.assertEqual(saved_results, {str(p): p.read_bytes() for p in session.glob("evaluations/*/result.pkl")})
                self.assertEqual(propose.call_count, 1)
                refit.assert_not_called()

    def test_both_cli_names_and_explicit_limits(self):
        for name in ("bora", "tosfit"):
            with patch("sys.argv", ["run", "--optimizer", name, "--algorithm", "random_forest",
                                    "--extractor-max-tokens", "128", "--extractor-max-calls", "2",
                                    "--extractor-timeout", "10", "--tosfit-device", "cpu",
                                    "--tosfit-reference-device", "cpu", "--recommendation-fallback", "default"]):
                args = runner.parse_args()
            params = runner.build_optimizer_params(args, name, args.algorithm, {})
            self.assertEqual(params["extractor_max_calls"], 2)
            self.assertTrue(params["recommendation_default_fallback"])
            self.assertIsNotNone(runner.durable_session_directory(args, name))
            self.assertNotIn(name, runner.parse_optimizers(["all"], []))

    def test_both_adapters_call_actual_cash_ml_evaluator(self):
        x, y = make_classification(n_samples=80, n_features=6, random_state=3)
        node = DataNode([x, y], feature_type=[NUMERICAL] * 6, task_type=CLASSIFICATION)
        config = dict(build_hpo_cs("cls", "random_forest").get_default_configuration())
        for name in ("bora", "tosfit"):
            runtime = DefaultRuntime(config)
            extractor = SimpleNamespace(extract=Mock(side_effect=AssertionError("valid JSON must bypass GPT")))
            params = {"llm_client": SimpleNamespace(complete=Mock(return_value=json.dumps(config))),
                      "extractor_client": extractor, "dataset_profile": {"metric": "bal_acc"},
                      "tosfit_runtime": runtime, "logging_enabled": False}
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory, \
                    patch.object(KnowledgeDatabase, "_instance", None), \
                    patch.object(KnowledgeDatabase, "fetch_best_conf_knowledge",
                                 side_effect=AssertionError("No cross-task retrieval")):
                cash = CASH(include_algorithms=["random_forest"], task_type=CLASSIFICATION,
                            metric="bal_acc", data_node=node, optimizer=name, amount_of_resource=1,
                            time_limit=None, per_run_time_limit=30, output_dir=directory,
                            resampling_params={"test_size": 0.25}, optimizer_params=params)
                best, _ = cash.run(refit="partial")
                self.assertTrue(np.isfinite(best))
                self.assertEqual(cash.optimizer.trial_cnt, 1)
                self.assertEqual(len(cash.optimizer.traj_rows), 1)
                self.assertTrue(np.isfinite(cash.optimizer.incumbent_perf))
                extractor.extract.assert_not_called()
                if name == "tosfit":
                    self.assertEqual(len(runtime.observed), 1)


if __name__ == "__main__":
    unittest.main()
