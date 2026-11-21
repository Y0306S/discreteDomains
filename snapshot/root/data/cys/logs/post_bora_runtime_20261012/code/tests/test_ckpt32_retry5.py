"""The ckpt32 retry policy preserves real observations and evaluation budgets."""

import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import Constant, UniformIntegerHyperparameter
from openbox.utils.constants import FAILED, SUCCESS
from openbox.utils.history import Observation

from mindware.components.optimizers import ml_checkpoint
from mindware.components.optimizers.pt4opt_llm_tpe_optimizer import PT4OptLLMTPEOptimizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/benchmark"))
from experiment_utils import apply_hybrid_settings


class ToyKnowledge:
    def __init__(self, space):
        self.space = space
        self.configs, self.perfs, self.bo_history_mab = [], [], {}
        self.lower_is_better = False

    def _dict_to_cfg(self, values):
        return Configuration(self.space, values=values)

    def update_history(self, obs, **kwargs):
        self.configs.append(obs.config)
        self.perfs.append(-float(obs.objectives[0]))


class ToyOptimizer(PT4OptLLMTPEOptimizer):
    def update_saver(self, configs, objectives):
        pass


class RetryPolicyTest(unittest.TestCase):
    def optimizer(self, **overrides):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        space = ConfigurationSpace(seed=1)
        space.add_hyperparameters([Constant("algorithm", "toy"),
                                   UniformIntegerHyperparameter("toy:depth", 1, 10000)])
        knowledge = ToyKnowledge(space)
        client = Mock()
        client.complete.return_value = self.response(1)
        params = {"llm_duplicate_policy": "history_tpe5", "llm_client": client,
                  "n_initial_samples": 3, "recent_k": 10, "top_k": 5, "worst_k": 5,
                  "history_window": 20, "response_format": "json_only", "hybrid_selector": "pucb",
                  "pucb_q": "arm_best", "pucb_p1": 0.9, "pucb_p_end": 0.5,
                  "tpe_duplicate_retries": 5, "dataset_profile": {"metric": "validation score"}}
        params.update(overrides)
        with patch("mindware.components.optimizers.pt4opt_llm_optimizer.KnowledgeDatabase.instance",
                   return_value=knowledge):
            opt = ToyOptimizer(evaluator=SimpleNamespace(fixed_config=None), config_space=space,
                               name="hpo", eval_type="holdout", task_id="retry-test", seed=1,
                               timestamp=time.time(), output_dir=folder.name, evaluation_limit=100,
                               optimizer_params=params)
        opt.objective_function = Mock(side_effect=lambda cfg: Observation(
            config=cfg, objectives=[-float(cfg["toy:depth"])], trial_state=SUCCESS, elapsed_time=0.01))
        return opt

    @staticmethod
    def response(depth):
        return json.dumps({"config": {"toy:depth": depth}})

    def add_history(self, opt, depths, scores=None):
        for depth, score in zip(depths, scores if scores is not None else depths):
            cfg = opt.KD._dict_to_cfg({"algorithm": "toy", "toy:depth": depth})
            if math.isfinite(score):
                opt.configs.append(cfg)
                opt.perfs.append(float(score))
            opt.traj_rows.append({"iter_idx": len(opt.traj_rows), "perf": float(score),
                                  "config_json": json.dumps(cfg.get_dictionary()), "stage": "llm"})
        opt.trial_cnt = len(opt.traj_rows)

    def test_five_calls_extend_history_without_dropping_best_worst_and_reset(self):
        opt = self.optimizer()
        self.add_history(opt, list(range(1, 31)))
        # Trial 15 is absent from best five, worst five and recent ten.
        opt.llm_client.complete.side_effect = [self.response(15)] * 4 + [self.response(101), self.response(102)]
        result = opt._suggest_config(30)
        self.assertEqual(result[0]["toy:depth"], 101)
        attempts = result[-1]["attempts"]
        self.assertEqual([a["recent_context_limit"] for a in attempts], [10, 11, 12, 13, 14])
        self.assertEqual([len(a["retry_history"]) for a in attempts], [0, 1, 2, 3, 4])
        self.assertEqual(opt.history_window, 24)
        self.assertNotIn(15, {r["trial"] for r in result[6]["selected_trials"]})
        self.assertNotIn("trial=15,", attempts[0]["messages"][-1]["content"])
        self.assertEqual(len(result[6]["selected_best_trials"]), 5)
        self.assertEqual(len(result[6]["selected_worst_trials"]), 5)
        for attempt in attempts:
            prompt = attempt["messages"][-1]["content"]
            self.assertIn("Strictly do not return a configuration you have previously recommended", prompt)
            self.assertNotIn("Do not return a configuration that duplicates any Selected historical trial", prompt)
        reminder = attempts[1]["retry_history"][0]["previous_evaluations"][0]
        self.assertEqual((reminder["trial"], reminder["score"], reminder["objective"]), (15, 15, -15))
        self.assertIn("trial=15, rel_score=0.4828", attempts[1]["messages"][-1]["content"])
        self.assertEqual(len(opt.traj_rows), 30)
        self.assertEqual(len(opt.study.trials), 0)
        next_result = opt._suggest_config(31)
        self.assertEqual(next_result[6]["retry_history"], [])
        self.assertEqual(next_result[-1]["attempts"][0]["recent_context_limit"], 10)
        self.assertNotIn("temporary recent reminders", next_result[-1]["user_prompt"])
        saved = json.loads((Path(opt.conversation_log_dir) / "trial_0030_proposals.json").read_text())
        self.assertEqual(len(saved["llm_attempts"]), 5)

    def test_all_measurements_of_a_duplicate_warm_start_are_preserved(self):
        opt = self.optimizer()
        self.add_history(opt, [1, 1, 1], [0.2, 0.3, 0.4])
        result = opt._suggest_config(3)
        self.assertTrue(result[6]["llm_retry_exhausted"])
        observations = result[-1]["attempts"][1]["retry_history"][0]["previous_evaluations"]
        self.assertEqual([r["trial"] for r in observations], [1, 2, 3])
        self.assertEqual([r["score"] for r in observations], [0.2, 0.3, 0.4])
        self.assertEqual(opt.llm_call_count, 5)

    def test_warm_starts_accept_duplicates_with_one_call_each(self):
        opt = self.optimizer()
        for _ in range(3):
            opt.iterate()
        self.assertEqual(opt.llm_call_count, 3)
        self.assertEqual(len(opt.traj_rows), 3)
        self.assertTrue(all(r["stage"] == "llm_warm_start" for r in opt.traj_rows))
        self.assertEqual(opt.selector.n, {"llm": 0, "tpe": 0})
        self.assertTrue(all("Strictly do not" not in call.args[1] for call in opt.llm_client.complete.call_args_list))

    def test_five_repeats_handoff_to_real_tpe_with_only_tpe_credit(self):
        opt = self.optimizer()
        for _ in range(4):
            opt.iterate()
        row = opt.traj_rows[-1]
        self.assertEqual(opt.objective_function.call_count, 4)
        self.assertEqual(opt.llm_call_count, 8)
        self.assertEqual((row["stage"], row["selected_arm"], row["configuration_source"]), ("tpe", "llm", "tpe"))
        self.assertTrue(row["llm_retry_exhausted"])
        self.assertEqual(row["llm_attempt_count"], 5)
        self.assertEqual(opt.selector.n, {"llm": 0, "tpe": 1})
        self.assertEqual(opt.selector.attempt_n, {"llm": 1, "tpe": 1})
        self.assertEqual(opt.selector.successes, {"llm": 0, "tpe": 1})
        self.assertEqual(len(opt.study.trials), 4)
        self.assertNotEqual(opt.configs[-1]["toy:depth"], 1)
        saved = json.loads((Path(opt.conversation_log_dir) / "trial_0003.json").read_text())
        self.assertEqual(len(saved["llm_attempts"]), 5)
        self.assertEqual(saved["normalized_config"], opt.configs[-1].get_dictionary())
        self.assertEqual(saved["history_meta"]["configuration_source"], "tpe")

    def test_failed_evaluation_is_also_recognized_as_a_previous_configuration(self):
        opt = self.optimizer()
        self.add_history(opt, [1, 2], [0.5, -math.inf])
        opt.llm_client.complete.side_effect = [self.response(2), self.response(3)]
        result = opt._suggest_config(3)
        previous = result[-1]["attempts"][1]["retry_history"][0]["previous_evaluations"][0]
        self.assertEqual(previous["trial"], 2)
        self.assertFalse(previous["successful"])
        self.assertIsNone(previous["score"])

    def test_old_checkpoint_reconstructs_both_attempt_counts_once(self):
        opt = self.optimizer()
        for _ in range(4):
            opt.iterate()
        del opt.selector.attempt_n
        opt.evaluation_num_limit = 4
        for _ in range(2):
            opt.iterate()
        self.assertEqual(opt.selector.attempt_n, {"llm": 1, "tpe": 1})
        self.assertEqual(opt.selector.n, {"llm": 0, "tpe": 1})
        self.assertEqual(opt.objective_function.call_count, 4)

    def test_malformed_output_keeps_existing_error_policy_and_logs_response(self):
        opt = self.optimizer(fallback_on_bad_json=False)
        opt.llm_client.complete.return_value = "not JSON"
        with self.assertRaises(ValueError):
            opt._suggest_config(3)
        self.assertEqual(opt.llm_call_count, 1)
        saved = json.loads((Path(opt.conversation_log_dir) / "trial_0003_proposals.json").read_text())
        self.assertEqual(saved["llm_attempts"][0]["raw_response"], "not JSON")
        opt.objective_function.assert_not_called()

    def test_tpe_retries_six_times_before_random_and_checks_final_random_draw(self):
        opt = self.optimizer()
        self.add_history(opt, [1])
        old, new = opt.configs[0], opt.KD._dict_to_cfg({"algorithm": "toy", "toy:depth": 2})
        for _ in range(6):
            opt.study.enqueue_trial({"toy:depth": 1})
        opt._sample_config = Mock(side_effect=[old] * 50 + [new])
        cfg, trial, _, note = opt._suggest_tpe_config()
        self.assertEqual(cfg, new)
        self.assertIsNone(trial)
        self.assertEqual(len(opt.study.trials), 6)
        self.assertTrue(all(t.value == -1 for t in opt.study.trials))
        self.assertIn("random fallback after 6 duplicate", note)
        self.assertEqual(opt._sample_config.call_count, 51)
        opt.objective_function.assert_not_called()
        opt._sample_config = Mock(return_value=old)
        with self.assertRaisesRegex(RuntimeError, "51 random proposals"):
            opt._sample_nonduplicate_config()
        self.assertEqual(opt._sample_config.call_count, 51)

    def test_interrupted_fallback_resumes_saved_tpe_proposal_without_llm_calls(self):
        opt = self.optimizer()
        for _ in range(3):
            opt.iterate()
        directory = Path(opt.output_dir) / "session"
        with patch.dict(os.environ, {"ML_RUN_ID": "retry5-test"}):
            opt._ml_session = ml_checkpoint.MLSession(opt, directory)
            opt.objective_function.side_effect = RuntimeError("interrupted")
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                opt.iterate()
            pending_cfg = opt._ml_session.pending["cfg"]
            self.assertEqual(opt.selector.n, {"llm": 0, "tpe": 0})
            resumed = self.optimizer()
            resumed._ml_session = ml_checkpoint.MLSession(resumed, directory)
            self.assertEqual(resumed._ml_session.restore(), 3)
            resumed.iterate()
            resumed.llm_client.complete.assert_not_called()
            self.assertEqual(resumed.objective_function.call_args.args[0], pending_cfg)
            self.assertEqual(resumed.selector.n, {"llm": 0, "tpe": 1})
            self.assertEqual(resumed.selector.attempt_n, {"llm": 1, "tpe": 1})
            self.assertEqual(len(resumed.study.trials), 4)
            ml_checkpoint.commit(resumed)
            self.assertIsNone(resumed._ml_session.pending)

    def test_real_runner_commits_each_completed_proposal(self):
        from mindware.modules.base import BaseAutoML

        opt = self.optimizer()
        model = BaseAutoML.__new__(BaseAutoML)
        model.optimizer = opt
        model.inner_iter_num_per_iter = 1
        model.time_limit, model.timestamp = None, opt.timestamp
        model.timeout_flag = False
        with patch.dict(os.environ, {"ML_RUN_ID": "real-runner-retry5"}):
            opt._ml_session = ml_checkpoint.MLSession(opt, Path(opt.output_dir) / "session")
            for count in range(1, 7):
                model.iterate()
                self.assertIsNone(opt._ml_session.pending)
                self.assertEqual(len(opt.traj_rows), count)
            self.assertEqual([r["iter_idx"] for r in opt.traj_rows], list(range(6)))
            self.assertEqual(opt.objective_function.call_count, 6)
            saved = json.loads((Path(opt.output_dir) / "session/progress.json").read_text())
            self.assertEqual(saved["attempts"], 6)
            self.assertFalse(saved["pending"])

    def test_full_budget_counts_evaluations_including_failure_not_proposal_retries(self):
        opt = self.optimizer()
        successful_evaluator = opt.objective_function.side_effect
        count = 0

        def evaluate(cfg):
            nonlocal count
            count += 1
            if count == 6:
                return Observation(config=cfg, objectives=[math.inf], trial_state=FAILED, elapsed_time=0.01)
            return successful_evaluator(cfg)

        opt.objective_function.side_effect = evaluate
        for _ in range(101):
            opt.iterate()
        self.assertEqual(len(opt.traj_rows), 100)
        self.assertEqual(count, 100)
        # TPE may record a cached score for a repeated proposal without evaluating it.
        self.assertGreaterEqual(len(opt.study.trials), 100)
        self.assertTrue(all(t.state.name != "RUNNING" for t in opt.study.trials))
        self.assertEqual(sum(math.isfinite(r["perf"]) for r in opt.traj_rows), 99)
        self.assertEqual(sum(opt.selector.n.values()), 97)
        self.assertEqual(opt.selector.n["llm"], 0)
        llm_selected = sum(r.get("selected_arm") == "llm" for r in opt.traj_rows[3:])
        self.assertEqual(opt.selector.attempt_n, {"llm": llm_selected, "tpe": 97})
        self.assertEqual(opt.llm_call_count, 3 + llm_selected * 5)
        self.assertAlmostEqual(opt.traj_rows[-1]["omega_llm"], 0.9 - 0.4 * 96 / 97)
        self.assertAlmostEqual(opt.traj_rows[-1]["omega_tpe"], 0.1 + 0.4 * 96 / 97)

    def test_variant_selects_exact_settings_and_current_q(self):
        args = SimpleNamespace()
        apply_hybrid_settings(args, "HPUCB_arm_p05_retry5")
        self.assertEqual((args.llm_n_initial_samples, args.llm_recent_k, args.llm_top_k, args.llm_worst_k), (3, 10, 5, 5))
        self.assertEqual(args.llm_history_window, 24)
        self.assertEqual(args.llm_duplicate_policy, "history_tpe5")
        self.assertEqual(args.llm_response_format, "json_only")
        opt = self.optimizer()
        opt.selector.update("llm", 0.8)
        opt.selector.update("llm", 0.7)
        opt.selector.update("tpe", 0.6)
        self.assertEqual(opt.selector.q_values(), {"llm": 0.5, "tpe": 2 / 3})
        for m in range(98):
            self.assertAlmostEqual(opt.selector.omega("llm", m, 97), 0.9 - 0.4 * m / 97)
            self.assertAlmostEqual(opt.selector.omega("tpe", m, 97), 0.1 + 0.4 * m / 97)

    def test_runner_passes_new_policy_and_isolates_its_output_variant(self):
        import run_toy_categorical_optimizers as runner

        with patch.object(sys, "argv", ["run", "--optimizer", "ptllm_tpe_qwen", "--llm-mock",
                                        "--prompt-variant", "P5plusA5T50X50RS5f05v4",
                                        "--hybrid-variant", "HPUCB_arm_p05_retry5"]):
            args = runner.parse_args()
        variant = runner.expand_optimizer_variants(args.optimizer, args)[0]
        self.assertIn("HPUCB_arm_p05_retry5", variant)
        params = runner.build_optimizer_params(args, variant, ["extra_trees"], {"metric": "bal_acc"})
        self.assertEqual(params["llm_duplicate_policy"], "history_tpe5")
        self.assertEqual((params["recent_k"], params["top_k"], params["worst_k"], params["history_window"]), (10, 5, 5, 24))
        self.assertEqual(params["n_initial_samples"], 3)
        self.assertEqual(params["pucb_p_end"], 0.5)
        self.assertEqual(params["pucb_q"], "arm_best")
        self.assertEqual(params["tpe_duplicate_retries"], 5)
        self.assertEqual(params["response_format"], "json_only")
        apply_hybrid_settings(args, "HPUCB_arm")
        self.assertEqual(args.llm_duplicate_policy, "legacy")


if __name__ == "__main__":
    unittest.main()
