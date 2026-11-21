import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ConfigSpace import ConfigurationSpace, Constant
from mindware.components.optimizers import ml_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/benchmark"))
import run_toy_categorical_optimizers as runner


class TransferRunnerTests(unittest.TestCase):
    def arguments(self, directory):
        with patch.object(sys, "argv", ["runner", "--optimizer", "fsbo", "rgpe",
                                        "--dataset", "car", "--algorithm", "extra_trees"]):
            args = runner.parse_args()
        args.output_dir = directory
        args.session_dir = Path(directory) / "sessions"
        args.runtime_policy_details = None
        args.resume = True
        args.max_rounds = 1
        return args

    def test_completion_needs_the_requested_verified_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "session"
            session.mkdir()
            state = {"attempts": 100, "successful": 100, "failed": 0,
                     "clean_complete": True, "pending": False, "identity": "fsbo_model_A"}
            (session / "progress.json").write_text(json.dumps(state))
            context = {"variant": "fsbo", "session_dir": str(session), "identity": "fsbo_model_A",
                       "output_dir": directory, "algorithm": "extra_trees"}
            with patch.dict(os.environ, {"ML_SESSION_DIR": str(session), "ML_RUN_ID": "rgpe_model_B"}):
                self.assertFalse(runner.variant_is_complete(Path(directory), "rgpe", 100, "extra_trees"))
                self.assertFalse(runner.variant_is_complete(Path(directory), "fsbo", 100, "extra_trees"))
                self.assertTrue(runner.variant_is_complete(Path(directory), "fsbo", 100, "extra_trees",
                                                            run_context=context))
            for changes in ({"variant": "rgpe"}, {"identity": "fsbo_model_B"},
                            {"output_dir": directory + "/another_dataset"}, {"algorithm": "random_forest"}):
                with self.subTest(changes=changes):
                    self.assertFalse(runner.variant_is_complete(Path(directory), "fsbo", 100, "extra_trees",
                                                               run_context={**context, **changes}))
            state.update(successful=99, failed=1, clean_complete=False)
            (session / "progress.json").write_text(json.dumps(state))
            self.assertTrue(runner.variant_is_complete(Path(directory), "fsbo", 100, "extra_trees",
                                                       run_context=context))
            self.assertFalse(runner.variant_is_complete(Path(directory), "fsbo", 100, "extra_trees",
                                                        run_context=context, require_success=True))

    def test_multi_method_run_validates_each_method_and_keeps_separate_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory)
            args.session_dir.mkdir()
            (args.session_dir / "progress.json").write_text(json.dumps(
                {"attempts": args.n_trials, "clean_complete": True, "identity": "unrelated"}))
            calls = []

            def run_optimizer(args, name, *unused):
                calls.append(name)
                session = runner.durable_session_directory(args, name)
                session.mkdir(parents=True)
                state = {"identity": name, "attempts": args.n_trials, "pending": False,
                         "clean_complete": name == "fsbo", "successful": args.n_trials - (name == "rgpe")}
                (session / "progress.json").write_text(json.dumps(state))
                args._run_contexts[name] = {"variant": name, "session_dir": str(session), "identity": name,
                                            "output_dir": directory, "algorithm": "extra_trees"}
                os.environ["ML_SESSION_DIR"] = str(session)
                return {"optimizer": name, "best_perf": 0.5, "test_perf": None, "output_dir": directory,
                        "progress": state, "run_id": name}

            with patch.dict(os.environ, {"ML_SESSION_DIR": str(args.session_dir)}), \
                    patch.object(runner, "load_real_data", return_value=(None, None, None, "data", {})), \
                    patch.object(runner, "run_one_optimizer", side_effect=run_optimizer), \
                    contextlib.redirect_stdout(io.StringIO()):
                path, _ = runner.run_one_dataset(args)
            self.assertEqual(calls, ["fsbo", "rgpe"])
            summary = json.loads(path.read_text())
            self.assertEqual(summary["completed_variants"], ["fsbo"])
            self.assertEqual(set(summary["progress_by_optimizer"]), {"fsbo", "rgpe"})
            self.assertNotIn("progress", summary)
            self.assertNotEqual(runner.durable_session_directory(args, "fsbo"),
                                runner.durable_session_directory(args, "rgpe"))
            previous = runner.durable_session_directory(args, "rgpe")
            args.seed += 1
            self.assertNotEqual(previous, runner.durable_session_directory(args, "rgpe"))
            previous = runner.durable_session_directory(args, "rgpe")
            args.dataset = "other"
            self.assertNotEqual(previous, runner.durable_session_directory(args, "rgpe"))

    def test_completed_resume_restores_outer_best_configuration_without_evaluating(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory)
            args.n_trials = 4
            args.skip_test_eval = True
            args.history_dir = directory
            args.session_dir.mkdir()
            space = ConfigurationSpace()
            space.add_hyperparameter(Constant("algorithm", "extra_trees"))
            inner = SimpleNamespace(asset_identity={"checkpoint_sha256": "model_A"},
                                    schema_fingerprint="schema", output_dir=directory,
                                    topk_saver=SimpleNamespace(identifier="saved"),
                                    incumbent_perf=0.7, incumbent_config={"algorithm": "extra_trees"})
            outer = SimpleNamespace(optimizer=inner, cs=space, incumbent=None,
                                    get_conf=Mock(return_value={}), run=Mock())

            def attach(optimizer):
                (args.session_dir / "progress.json").write_text(json.dumps(
                    {"attempts": 4, "successful": 4, "clean_complete": True,
                     "pending": False, "identity": os.environ["ML_RUN_ID"]}))
                return 4

            def model_info(**kwargs):
                self.assertEqual(outer.incumbent, inner.incumbent_config)
                self.assertEqual(outer.incumbent_perf, inner.incumbent_perf)
                return {"best": outer.incumbent}

            outer.get_model_info = model_info
            with patch.dict(os.environ, {}), patch.object(runner, "CASH", return_value=outer), \
                    patch.object(ml_checkpoint, "attach", side_effect=attach), \
                    patch.object(runner, "save_histories", return_value={}), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = runner.run_one_optimizer(args, "fsbo", None, None, args.algorithm, {})
            outer.run.assert_not_called()
            self.assertEqual(result["progress"]["attempts"], 4)

    def test_mixed_legacy_summary_preserves_its_verified_completed_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory)
            results = [{"optimizer": "fsbo", "progress": {"clean_complete": True}, "run_id": "fsbo"},
                       {"optimizer": "tpe"}]
            path = runner.save_dataset_summary(Path(directory), "car", "data", args.algorithm,
                                               args, results, completed_variants=["fsbo", "tpe"])
            self.assertEqual(json.loads(path.read_text())["completed_variants"], ["fsbo", "tpe"])


if __name__ == "__main__":
    unittest.main()
