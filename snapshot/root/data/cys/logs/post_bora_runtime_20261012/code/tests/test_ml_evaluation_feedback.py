"""Regression coverage for configuration attribution, deadlines, and recovery."""

import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


def _slow_with_child(marker):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    Path(marker).write_text(str(child.pid))
    time.sleep(30)


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        import optuna
        from ConfigSpace import ConfigurationSpace
        from ConfigSpace.forbidden import ForbiddenAndConjunction, ForbiddenEqualsClause
        from ConfigSpace.hyperparameters import CategoricalHyperparameter
        from mindware.components.optimizers.pt4opt_llm_tpe_optimizer import PT4OptLLMTPEOptimizer
        from mindware.components.optimizers.ml_evaluation_feedback import runtime_constraints

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        self.optuna = optuna
        self.cs = ConfigurationSpace(seed=0)
        alg = CategoricalHyperparameter("algorithm", ["test"])
        penalty = CategoricalHyperparameter("test:penalty", ["l1", "l2"], default_value="l2")
        loss = CategoricalHyperparameter("test:loss", ["hinge", "squared_hinge"], default_value="squared_hinge")
        self.cs.add_hyperparameters([alg, penalty, loss])
        self.cs.add_forbidden_clause(ForbiddenAndConjunction(
            ForbiddenEqualsClause(penalty, "l1"), ForbiddenEqualsClause(loss, "hinge")))
        self.opt = object.__new__(PT4OptLLMTPEOptimizer)
        self.opt.config_space = self.cs
        self.opt._algorithm_hp = alg
        self.opt._algorithm_choices = ["test"]
        self.opt._single_algorithm = self.opt.fixed_algorithm = "test"
        self.opt.study = optuna.create_study(sampler=optuna.samplers.TPESampler(
            seed=1, constraints_func=runtime_constraints))
        self.opt.logger = logging.getLogger("feedback_test")
        self.opt.tpe_duplicate_retries = 0
        self.opt._is_duplicate = lambda cfg: False
        self.opt._sample_nonduplicate_config = self.opt._sample_fallback_configuration

    def config(self, penalty="l1", loss="squared_hinge"):
        from ConfigSpace import Configuration
        from mindware.components.optimizers.utils import MyConfiguration
        return MyConfiguration(Configuration(self.cs, values={
            "algorithm": "test", "test:penalty": penalty, "test:loss": loss}))

    def test_invalid_trial_never_receives_fallback_score(self):
        self.opt.study.enqueue_trial({"test:penalty": "l1", "test:loss": "hinge"})
        cfg, trial, _, _ = self.opt._suggest_tpe_config()
        self.assertIsNone(trial)
        self.assertEqual(self.opt.study.trials[0].state, self.optuna.trial.TrialState.FAIL)
        self.opt._tell_tpe_external_config(cfg, 0.123)
        stored = self.opt.study.trials[1]
        self.assertTrue(self.opt._trial_matches_config(stored, cfg))
        self.assertEqual(stored.value, 0.123)

    def test_external_feedback_does_not_consume_another_waiting_trial(self):
        self.opt.study.enqueue_trial({"test:penalty": "l2", "test:loss": "hinge"})
        cfg = self.config()
        self.opt._tell_tpe_external_config(cfg, 0.456)
        self.assertEqual(self.opt.study.trials[0].state, self.optuna.trial.TrialState.WAITING)
        self.assertTrue(self.opt._trial_matches_config(self.opt.study.trials[1], cfg))
        self.assertEqual(self.opt.study.trials[1].value, 0.456)

    def test_mismatched_pending_trial_is_failed_and_replaced(self):
        self.opt.study.enqueue_trial({"test:penalty": "l2", "test:loss": "hinge"})
        trial = self.opt.study.ask()
        self.opt._try_suggest_config_from_trial(trial)
        actual = self.config()
        self.opt._tell_tpe_trial(trial, 0.5, actual)
        old, new = self.opt.study.trials
        self.assertEqual(old.state, self.optuna.trial.TrialState.FAIL)
        self.assertIsNone(old.value)
        self.assertTrue(self.opt._trial_matches_config(new, actual))

    def test_unrepresentable_external_config_raises_without_false_score(self):
        self.opt._try_suggest_config_from_trial = lambda trial: self.config("l2", "hinge")
        with self.assertRaisesRegex(RuntimeError, "cannot be recorded exactly"):
            self.opt._tell_tpe_external_config(self.config(), 0.5)
        self.assertEqual(self.opt.study.trials, [])

    def test_float_precision_and_conditional_parameters_match(self):
        from ConfigSpace import ConfigurationSpace
        from ConfigSpace.hyperparameters import UniformFloatHyperparameter
        self.opt.config_space = ConfigurationSpace(seed=2)
        self.opt.config_space.add_hyperparameter(UniformFloatHyperparameter("x", 1e-6, 1.0, log=True))
        self.opt._algorithm_hp = None
        self.opt._single_algorithm = None
        for _ in range(20):
            trial = self.opt.study.ask()
            cfg = self.opt._try_suggest_config_from_trial(trial)
            self.assertTrue(self.opt._trial_matches_config(trial, cfg))
            self.opt._tell_tpe_trial(trial, 1.0, cfg)
        self.assertEqual(len(self.opt.study.trials), 20)

    def test_timeout_is_pruned_infeasible_and_has_no_fake_objective(self):
        from openbox.utils.constants import FAILED, SUCCESS, TIMEOUT
        from openbox.utils.history import Observation
        from mindware.components.optimizers.ml_evaluation_feedback import observation_feedback
        from optuna.samplers._tpe.sampler import _split_trials
        cfg = self.config()
        for status, value in ((SUCCESS, 0.5), (TIMEOUT, float("inf")), (FAILED, float("inf"))):
            obs = Observation(cfg, objectives=[value], trial_state=status, elapsed_time=2400)
            objective, state, attrs = observation_feedback(obs, 2400)
            self.opt._tell_tpe_external_config(cfg, objective, state=state, attrs=attrs)
        complete, timeout, failure = self.opt.study.trials
        self.assertEqual(timeout.state, self.optuna.trial.TrialState.PRUNED)
        self.assertIsNone(timeout.value)
        self.assertEqual(timeout.system_attrs["constraints"], (1.0,))
        self.assertEqual(complete.system_attrs["constraints"], (0.0,))
        self.assertEqual(failure.state, self.optuna.trial.TrialState.FAIL)
        below, above = _split_trials(self.opt.study, [complete, timeout], 1, True)
        self.assertEqual([t.number for t in below], [complete.number])
        self.assertEqual([t.number for t in above], [timeout.number])

    def test_asked_timeout_also_records_constraint(self):
        trial = self.opt.study.ask()
        cfg = self.opt._try_suggest_config_from_trial(trial)
        if cfg is None:
            self.opt.study.tell(trial, state=self.optuna.trial.TrialState.FAIL)
            trial = self.opt.study.ask()
            cfg = self.opt._try_suggest_config_from_trial(trial)
        self.assertIsNotNone(cfg)
        self.opt._tell_tpe_trial(trial, None, cfg, state=self.optuna.trial.TrialState.PRUNED,
                                 attrs={"evaluation_status": "TIMEOUT"})
        self.assertEqual(self.opt.study.trials[-1].system_attrs["constraints"], (1.0,))

    def test_study_upgrade_preserves_trial_ids_and_sampler_rng(self):
        import pickle
        from mindware.components.optimizers.ml_evaluation_feedback import upgrade_runtime_study, runtime_constraints
        old = self.optuna.create_study(sampler=self.optuna.samplers.TPESampler(seed=17))
        trial = old.ask()
        trial.suggest_float("x", 0, 1)
        old.tell(trial, 0.4)
        pending = old.ask()
        pending.suggest_float("x", 0, 1)
        before = pickle.dumps(old.sampler._rng.rng.get_state())
        updated = upgrade_runtime_study(old)
        self.assertEqual(before, pickle.dumps(updated.sampler._rng.rng.get_state()))
        self.assertEqual([(t.number, t._trial_id) for t in old.trials],
                         [(t.number, t._trial_id) for t in updated.trials])
        self.assertIs(updated.sampler._constraints_func, runtime_constraints)
        self.assertEqual(updated.trials[0].system_attrs["constraints"], (0.0,))
        self.assertEqual(updated.trials[1].state, self.optuna.trial.TrialState.RUNNING)

    def test_timeout_shown_in_recent_history_without_entering_best_or_worst(self):
        from mindware.components.optimizers.pt4opt_llm_optimizer import build_history_prompt_block
        cfg = self.config()
        observations = [{"iter_idx": i, "perf": score, "config_json": json.dumps(cfg.get_dictionary()),
                         "evaluation_status": "TIMEOUT" if i == 2 else "SUCCESS"}
                        for i, score in enumerate((0.1, 0.3, float("-inf")))]
        text, meta = build_history_prompt_block([cfg, cfg], [0.1, 0.3], "test", self.cs,
            top_k=1, worst_k=1, recent_k=2, evaluation_observations=observations,
            experience_self_mode="off")
        self.assertIn("trial=3, evaluation_status=TIMEOUT, score=unavailable", text)
        self.assertEqual(meta["selected_recent_trials"], [2, 3])
        self.assertNotIn(3, meta["selected_best_trials"])
        self.assertNotIn(3, meta["selected_worst_trials"])

    def test_hybrid_stops_at_100_attempts_even_when_every_evaluation_times_out(self):
        from types import SimpleNamespace
        from openbox.utils.constants import TIMEOUT
        from openbox.utils.history import Observation
        from mindware.components.optimizers.ml_checkpoint import progress
        opt = self.opt
        cfg = self.config()
        opt.inner_iter_num_per_iter = 1
        opt.evaluation_num_limit = 100
        opt.per_run_time_limit = 2400
        opt.time_limit = None
        opt.trial_cnt = 0
        opt.traj_rows, opt.perfs = [], []
        opt.llm_duplicate_policy = "history_tpe5"
        opt._choose_source = lambda index: ("tpe", True, None)
        opt._suggest_tpe_config = lambda: (cfg, None, 0, "test proposal")
        opt.experience_retrieve_ranker = "off"
        opt.objective_function = lambda config: Observation(config, objectives=[float("inf")],
                                                             trial_state=TIMEOUT, elapsed_time=2400)
        opt.selector = SimpleNamespace(arm_best={}, update=lambda *args, **kwargs: None)
        opt._update_incumbent = lambda: None
        opt.incumbent_perf, opt.incumbent_config = float("-inf"), None
        opt.trial_sources = {}
        opt._record_hybrid_event = lambda **row: opt.traj_rows.append({
            "iter_idx": row["iter_idx"], "perf": row["perf"], **row["meta"]})
        for _ in range(105):
            opt.iterate()
        self.assertEqual(len(opt.traj_rows), 100)
        self.assertEqual(len(opt.study.trials), 100)
        self.assertTrue(all(t.state == self.optuna.trial.TrialState.PRUNED for t in opt.study.trials))
        self.assertTrue(progress(opt.traj_rows, 100)["budget_exhausted"])

    def test_checkpoint_migration_keeps_new_deadline_and_pending_proposal(self):
        from types import SimpleNamespace
        from mindware.components.optimizers.ml_checkpoint import MLSession
        cfg = self.config()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"ML_RUN_ID": "old"}):
            opt = SimpleNamespace(config_space=self.cs, evaluator=SimpleNamespace(output_dir=directory),
                KD=SimpleNamespace(configs=[cfg], perfs=[0.5]), per_run_time_limit=172800,
                time_limit=None, evaluation_num_limit=100, traj_csv_path=str(Path(directory) / "traj.csv"),
                traj_rows=[{"iter_idx": 0, "perf": 0.5}], configs=[cfg], perfs=[0.5])
            old = MLSession(opt, directory)
            old.save({"cfg": cfg, "iter_idx": 1})
            opt.per_run_time_limit = 2400
            with patch.dict(os.environ, {"ML_RUN_ID": "new", "ML_PREVIOUS_RUN_ID": "old"}):
                session = MLSession(opt, directory)
                self.assertEqual(session.restore(), 1)
                self.assertEqual(opt.per_run_time_limit, 2400)
                self.assertEqual(opt.traj_rows[0]["evaluation_limit_seconds"], 172800)
                self.assertEqual(session.pending["iter_idx"], 1)
                self.assertEqual(opt.runtime_policy_revisions[0]["after_attempts"], 1)


class DeadlineTests(unittest.TestCase):
    def test_cuda_flag_never_disables_deadline(self):
        from mindware.components.optimizers.llm_optimizer_utils import wrapper_evaluater
        from openbox.utils.constants import TIMEOUT
        with patch("torch.cuda.is_initialized", return_value=True):
            observation = wrapper_evaluater(30, time.sleep, 0.4)
        self.assertEqual(observation.trial_state, TIMEOUT)
        self.assertLess(observation.elapsed_time, 4)

    def test_child_processes_are_cleaned_up(self):
        from mindware.components.optimizers.isolated_evaluator import run_isolated
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "child.pid"
            result = run_isolated(_slow_with_child, (str(marker),), {}, 10.0)
            self.assertTrue(result["timeout"])
            self.assertTrue(marker.exists(), "the child evaluation never started")
            pid = int(marker.read_text())
            status = Path(f"/proc/{pid}/stat")
            self.assertTrue(not status.exists() or status.read_text().split(") ", 1)[1].startswith("Z"))

    def test_success_and_cumulative_interruption_budget(self):
        from mindware.components.optimizers.isolated_evaluator import run_isolated
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "ML_SESSION_DIR": directory, "ML_EVALUATION_ID": "0"}):
            folder = Path(directory) / "evaluations/0"
            folder.mkdir(parents=True)
            (folder / "execution.json").write_text(json.dumps({
                "status": "interrupted", "elapsed_seconds": 2.0}))
            result = run_isolated(time.sleep, (30,), {}, 1.0)
            self.assertTrue(result["timeout"])
            self.assertEqual(result["elapsed_time"], 2.0)
        result = run_isolated(abs, (-3,), {}, 20.0)
        self.assertFalse(result["timeout"])
        self.assertEqual(result["result"], 3)

    @unittest.skipUnless(os.environ.get("ML_TEST_CUDA") == "1", "separate real-CUDA smoke test")
    def test_deadline_after_actual_cuda_initialization(self):
        import torch
        from mindware.components.optimizers.llm_optimizer_utils import wrapper_evaluater
        from openbox.utils.constants import TIMEOUT
        torch.ones(1, device="cuda")
        self.assertTrue(torch.cuda.is_initialized())
        observation = wrapper_evaluater(30, time.sleep, 0.5)
        self.assertEqual(observation.trial_state, TIMEOUT)
        self.assertLess(observation.elapsed_time, 4)


if __name__ == "__main__":
    unittest.main()
