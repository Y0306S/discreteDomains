"""BOLT handoff, exact observations and resumable LOLBO state."""

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import optuna
import torch
from ConfigSpace import Configuration
from openbox.utils.constants import SUCCESS, FAILED, TIMEOUT

from test_online_llm_baselines import Extractor, MemoryHistory, space
from mindware.components.optimizers import ml_checkpoint
from mindware.components.optimizers.bolt_ml_optimizer import BOLTMLOptimizer
from mindware.components.optimizers.lolbo_sampler import LOLBOTurboSampler
from mindware.components.optimizers.online_llm_optimizer import OnlineLLMOptimizer


class BoltTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cs = space()
        self.kd = SimpleNamespace(bo_history_mab={"fixture": MemoryHistory()}, update_history=Mock())
        for context in (
            patch("mindware.components.optimizers.online_llm_optimizer.KnowledgeDatabase.instance",
                  return_value=self.kd),
            patch.object(OnlineLLMOptimizer, "update_saver"),
            patch.dict(os.environ, {"ML_RUN_ID": "bolt-handoff-test"}),
            patch("mindware.components.optimizers.bolt_ml_optimizer.LOLBOTurboSampler",
                  side_effect=lambda seed, device: LOLBOTurboSampler(
                      seed=seed, device="cpu", n_candidates=32,
                      init_n_epochs=1, num_update_epochs=1)),
        ):
            context.start()
            self.addCleanup(context.stop)

    def make(self):
        configs = []
        for step in range(10):
            config = dict(self.cs.get_default_configuration())
            config["fixture:depth"] = 3 + step
            configs.append(json.dumps(config))
        return BOLTMLOptimizer(
            evaluator=SimpleNamespace(), config_space=self.cs, evaluation_limit=100,
            per_run_time_limit=2400, time_limit=None, output_dir=self.temp.name,
            task_type="rgs", task_id="rgs/fixture/fixture", logging_enabled=False,
            recommendation_default_fallback=True, extractor_client=Extractor(),
            llm_client=SimpleNamespace(complete_messages=Mock(side_effect=configs)))

    @staticmethod
    def oracle(config, evaluator, limit, **kwargs):
        assert limit == 2400 and kwargs["force_isolation"]
        return SimpleNamespace(config=config, objectives=[float(config["fixture:depth"])],
                               trial_state=SUCCESS, elapsed_time=0.01)

    def test_100_actual_evaluations_ten_llm_calls_and_exact_objective_sign(self):
        opt = self.make()
        with patch("mindware.components.optimizers.online_llm_optimizer.wrapper_evaluater",
                   side_effect=self.oracle) as evaluate:
            opt.run()
            opt.iterate()
        self.assertEqual(evaluate.call_count, 100)
        self.assertEqual(opt.llm_client.complete_messages.call_count, 10)
        self.assertEqual(len(opt.traj_rows), 100)
        trials = opt.study.get_trials(deepcopy=False)
        self.assertEqual(len(trials), 100)
        self.assertEqual([t.user_attrs["phase"] for t in trials],
                         ["learned_initialization"] * 10 + ["bo"] * 90)
        for row, trial in zip(opt.traj_rows, trials):
            config = json.loads(row["config_json"])
            self.assertEqual(trial.params, opt._config_to_optuna_params(config))
            self.assertEqual(trial.value, config["fixture:depth"])
            self.assertEqual(row["perf"], -trial.value)
        self.assertTrue(opt.study.sampler._initial_fit_done)
        self.assertEqual(opt.study.sampler._n_seen, 90)
        self.assertEqual(opt.study.sampler._pending, [])

    def test_failed_starts_use_budget_without_fake_gp_scores_or_extra_llm_calls(self):
        opt = self.make()
        outcomes = iter([TIMEOUT, FAILED] + [SUCCESS] * 12)

        def evaluate(config, evaluator, limit, **kwargs):
            obs = self.oracle(config, evaluator, limit, **kwargs)
            obs.trial_state = next(outcomes)
            if obs.trial_state != SUCCESS:
                obs.objectives = [float("inf")]
            return obs

        with patch("mindware.components.optimizers.online_llm_optimizer.wrapper_evaluater",
                   side_effect=evaluate) as oracle:
            for _ in range(14):
                opt.iterate()
        self.assertEqual(oracle.call_count, 14)
        self.assertEqual(opt.llm_client.complete_messages.call_count, 10)
        trials = opt.study.get_trials(deepcopy=False)
        self.assertEqual([t.state for t in trials[:2]],
                         [optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.FAIL])
        self.assertTrue(all(t.value is None for t in trials[:2]))
        self.assertEqual([r["evaluation_status"] for r in opt.traj_rows[:2]], ["TIMEOUT", "FAILED"])
        self.assertEqual(trials[10].user_attrs["supplier"], "lolbo_startup_random")
        self.assertEqual(trials[12].user_attrs["supplier"], "lolbo")
        self.assertTrue(opt.study.sampler._initial_fit_done)

    def test_pending_bo_resume_restores_gp_and_batch_without_regeneration(self):
        first = self.make()
        first._ml_session = ml_checkpoint.MLSession(first, self.temp.name)
        with patch("mindware.components.optimizers.online_llm_optimizer.wrapper_evaluater",
                   side_effect=self.oracle) as evaluate:
            for _ in range(12):
                first.iterate()
            proposal = first._suggest()
            proposal["suggest_time"] = 0.0
            ml_checkpoint.prepare(first, proposal)
            state = first.bo_state()
            weights = {k: v.clone() for k, v in first.study.sampler._model.state_dict().items()}
            second = self.make()
            second._ml_session = ml_checkpoint.MLSession(second, self.temp.name)
            self.assertEqual(second._ml_session.restore(), 12)
            self.assertEqual(second.bo_state(), state)
            self.assertEqual(ml_checkpoint.pending(second), proposal)
            self.assertTrue(all(torch.equal(weights[k], v)
                                for k, v in second.study.sampler._model.state_dict().items()))
            second.iterate()
            self.assertEqual(evaluate.call_count, 13)
            self.assertEqual(dict(evaluate.call_args.args[0]), proposal["config"])
            self.assertEqual(second.study.sampler._pending, state["queued_candidates"])
            second.llm_client.complete_messages.assert_not_called()
            self.assertIsNone(ml_checkpoint.pending(second))

    def test_invalid_bo_trials_never_receive_recovery_configuration_score(self):
        opt = self.make()
        with patch("mindware.components.optimizers.online_llm_optimizer.wrapper_evaluater",
                   side_effect=self.oracle) as evaluate:
            for _ in range(10):
                opt.iterate()
            original = opt._try_suggest_config_from_trial

            def reject_live(trial):
                return original(trial) if isinstance(trial, optuna.trial.FixedTrial) else None

            with patch.object(opt, "_try_suggest_config_from_trial", side_effect=reject_live):
                opt.iterate()
        self.assertEqual(evaluate.call_count, 11)
        trials = opt.study.get_trials(deepcopy=False)
        self.assertTrue(all(t.state == optuna.trial.TrialState.FAIL for t in trials[10:14]))
        self.assertEqual(trials[14].params,
                         opt._config_to_optuna_params(json.loads(opt.traj_rows[-1]["config_json"])))
        self.assertEqual(trials[14].value, -opt.traj_rows[-1]["perf"])

    def test_mismatched_trial_is_failed_and_actual_config_is_recorded_exactly(self):
        opt = self.make()
        trial = opt.study.ask()
        original = opt._try_suggest_config_from_trial(trial)
        changed = dict(original)
        changed["fixture:depth"] = 15 if changed["fixture:depth"] != 15 else 3
        config = Configuration(self.cs, values=changed)
        opt._tell_config_result(config, 7.0, trial)
        trials = opt.study.get_trials(deepcopy=False)
        self.assertEqual(trials[0].state, optuna.trial.TrialState.FAIL)
        self.assertEqual(trials[1].params, opt._config_to_optuna_params(config))
        self.assertEqual(trials[1].value, 7.0)

    def test_cli_selects_bolt_recipe_durable_identity_and_no_extra_test_fit(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/benchmark"))
        import run_toy_categorical_optimizers as runner

        with patch.object(sys, "argv", ["run", "--optimizer", "bolt_ml", "--algorithm", "lightgbm",
                "--n-trials", "100", "--time-limit", "none", "--per-run-time-limit", "2400",
                "--skip-test-eval", "--extractor-max-tokens", "2048", "--extractor-timeout", "60",
                "--extractor-max-calls", "100", "--session-dir", self.temp.name]):
            args = runner.parse_args()
        params = runner.build_optimizer_params(args, "bolt_ml", args.algorithm, {})
        self.assertTrue(params["recommendation_default_fallback"])
        self.assertEqual(params["task_type"], args.task_type)
        self.assertIsNotNone(runner.durable_session_directory(args, "bolt_ml"))
        self.assertFalse(runner.variant_is_complete(Path(self.temp.name), "bolt_ml", 100))
        self.assertNotIn("bolt_ml", runner.parse_optimizers(["all"], []))
        args.skip_test_eval = False
        with self.assertRaisesRegex(ValueError, "skip-test-eval"):
            runner.build_optimizer_params(args, "bolt_ml", args.algorithm, {})


if __name__ == "__main__":
    unittest.main()
