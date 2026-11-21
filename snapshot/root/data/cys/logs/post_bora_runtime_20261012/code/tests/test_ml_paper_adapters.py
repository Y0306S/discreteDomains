"""Failure accounting and exact-config prediction checks for the paper adapters."""

import json
import os
import pickle
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openbox.utils.constants import SUCCESS, TIMEOUT

from test_online_llm_baselines import AdapterTests, Extractor, MockRuntime, space
from mindware.components.optimizers.bora_optimizer import BORAOptimizer
from mindware.components.optimizers.semanticopt_optimizer import SemanticOptOptimizer, weighted_ei
from mindware.components.optimizers.tosfit_optimizer import ToSFiTOptimizer


class FailureChecks(AdapterTests):
    def test_tosfit_weights_retire_only_after_final_evaluation_and_ledger_survives(self):
        from test_tosfit_runtime import RuntimeTests
        from mindware.components.optimizers import ml_checkpoint

        runtime = RuntimeTests().make()
        runtime.recipe["n_marginal_likelihood_warmup_steps"] = 1
        with patch.dict(os.environ, {"ML_RUN_ID": "tosfit-completion-test"}):
            opt = self.make(ToSFiTOptimizer, limit=3, tosfit_runtime=runtime,
                            recommendation_default_fallback=True)
            session = ml_checkpoint.MLSession(opt, self.temp.name)
            opt._ml_session = session
            with patch("mindware.components.optimizers.online_llm_optimizer.wrapper_evaluater",
                       side_effect=self.oracle) as evaluate:
                opt.iterate()
                opt.iterate()
                self.assertEqual(runtime.policy_version, 1)
                self.assertEqual(len(list(session.directory.glob("tosfit-policy/weights-*.pt"))), 1)
                self.assertFalse((session.directory / "verified_completion.json").exists())
                opt.iterate()
                self.assertEqual(evaluate.call_count, 3)
                self.assertEqual(runtime.policy_version, 2)
                self.assertEqual(list(session.directory.glob("tosfit-policy/weights-*.pt")), [])
                with session.path.open("rb") as stream:
                    saved = pickle.load(stream)
                self.assertTrue(saved["progress"]["clean_complete"])
                self.assertTrue(saved["extra"]["weights_retired"])
                self.assertIsNone(saved["extra"]["weights"])
                self.assertEqual(len(saved["optimizer"]["traj_rows"]), 3)
                self.assertEqual(saved["extra"]["gp"]["_n_observations"], 3)
                fresh = self.make(ToSFiTOptimizer, limit=3, tosfit_runtime=RuntimeTests().make(),
                                  recommendation_default_fallback=True)
                fresh._ml_session = ml_checkpoint.MLSession(fresh, self.temp.name)
                self.assertEqual(fresh._ml_session.restore(), 3)
                fresh.iterate()
                self.assertEqual(evaluate.call_count, 3)
                self.assertEqual(fresh.traj_rows, opt.traj_rows)

    def test_bora_local_chat_preserves_merge_content_and_repairs_single_point_container(self):
        from mindware.components.optimizers.bora_ml_runtime import MLComment, QwenBridge
        opt = self.make(recommendation_default_fallback=True)
        client = SimpleNamespace(complete_messages=Mock(return_value="overview"))
        opt.llm_client = client
        messages = [{"role": "system", "content": "BORA"},
                    {"role": "user", "content": "task"},
                    {"role": "system", "content": "three original drafts"},
                    {"role": "user", "content": "merge"}]
        QwenBridge(opt).create(messages=messages)
        sent = client.complete_messages.call_args.args[0]
        self.assertEqual(sent[2], {"role": "assistant", "content": "three original drafts"})
        self.assertEqual(messages[2]["role"], "system")
        raw = json.dumps({"comment": "test", "hypotheses": [{"name": "first", "rationale": "test",
            "confidence": "high", "points": dict(self.cs.get_default_configuration())}]})
        comment = MLComment(raw, -1, opt._assistant()._experiment)
        self.assertTrue(comment.is_valid)
        self.assertEqual(comment.hypotheses[0]["points"], [dict(self.cs.get_default_configuration())])

    def test_all_failed_bora_initializations_continue_and_save_timeout_status(self):
        opt = self.make(BORAOptimizer, limit=6, recommendation_default_fallback=True)
        outcomes = iter([TIMEOUT] * 5 + [SUCCESS])

        def oracle(config, evaluator, limit, **kwargs):
            state = next(outcomes)
            return SimpleNamespace(config=config, trial_state=state, elapsed_time=1,
                                   objectives=[-.5 if state == SUCCESS else float("inf")])

        with patch("mindware.components.optimizers.online_llm_optimizer.wrapper_evaluater", side_effect=oracle):
            opt.run()
        self.assertEqual([r["evaluation_status"] for r in opt.traj_rows], ["TIMEOUT"] * 5 + ["SUCCESS"])
        self.assertTrue(opt.initialized)

    def test_incumbent_timeout_preserves_measured_reward_and_average(self):
        runtime = MockRuntime(self.cs)
        opt = self.make(ToSFiTOptimizer, limit=3, tosfit_runtime=runtime)
        results = iter([(-.8, SUCCESS), (-.5, SUCCESS), (float("inf"), TIMEOUT)])

        def oracle(config, evaluator, limit, **kwargs):
            objective, state = next(results)
            return SimpleNamespace(config=config, trial_state=state, elapsed_time=1, objectives=[objective])

        with patch("mindware.components.optimizers.online_llm_optimizer.wrapper_evaluater", side_effect=oracle):
            opt.run()
        self.assertEqual([r["best_so_far"] for r in opt.traj_rows], [.8, .8, .8])
        self.assertEqual(opt.solutions[0]["count"], 1)
        self.assertEqual(opt.solutions[0]["mean"], .8)
        self.assertEqual(len(runtime.observations), 2)


class SemanticChecks(unittest.TestCase):
    def make(self, replies):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        client = SimpleNamespace(complete_messages=Mock(side_effect=[json.dumps(p) for p in replies]))
        with patch("mindware.components.optimizers.online_llm_optimizer.KnowledgeDatabase.instance",
                   return_value=SimpleNamespace()):
            opt = SemanticOptOptimizer(evaluator=SimpleNamespace(), config_space=space(),
                output_dir=directory.name, evaluation_limit=100, task_type="rgs",
                recommendation_default_fallback=True, extractor_client=Extractor(),
                llm_client=client, logging_enabled=False)
        opt.semantic_history = [{"configuration": dict(space().get_default_configuration()),
                                 "status": "success", "objective": 10.0} for _ in range(5)]
        opt.trial_cnt = 5
        return opt, client

    @staticmethod
    def candidate(depth, mean=1.0, std=.1):
        config = dict(space().get_default_configuration())
        config["fixture:depth"] = depth
        return {"next_configuration": config, "mean": mean, "std": std}

    def test_two_model_candidates_one_selected_without_oracle_calls(self):
        opt, client = self.make([self.candidate(3, mean=8), self.candidate(4, mean=2)])
        proposal = opt._suggest()
        self.assertEqual(proposal["config"]["fixture:depth"], 4)
        self.assertEqual(client.complete_messages.call_count, 2)
        self.assertEqual(opt.trial_cnt, 5)
        self.assertEqual(len(opt.configs), 0)

    def test_changed_configuration_receives_fresh_predictions_once(self):
        opt, client = self.make([self.candidate(-1), self.candidate(9, mean=7)])
        proposal = opt._candidate(opt._messages())
        self.assertEqual(proposal["config"]["fixture:depth"], 9)
        self.assertEqual(proposal["prediction"]["mean"], 7)
        self.assertTrue(proposal["prediction_refreshed"])
        self.assertEqual(client.complete_messages.call_count, 2)

    def test_changed_refresh_is_default_with_no_stale_prediction(self):
        opt, client = self.make([self.candidate(-1), self.candidate(3)])
        proposal = opt._candidate(opt._messages())
        self.assertEqual(proposal["config"], dict(space().get_default_configuration()))
        self.assertIsNone(proposal["prediction"])
        self.assertEqual(client.complete_messages.call_count, 2)

    def test_default_competes_with_usable_candidate_using_its_own_predictions(self):
        broken = {"next_configuration": self.candidate(3)["next_configuration"]}
        for default_mean, selected_depth in ((8.0, 4), (1.0, 9)):
            with self.subTest(default_mean=default_mean):
                opt, client = self.make([broken, broken, self.candidate(4, mean=2),
                                         self.candidate(9, mean=default_mean)])
                proposal = opt._suggest()
                self.assertEqual(proposal["config"]["fixture:depth"], selected_depth)
                self.assertEqual(client.complete_messages.call_count, 4)
                request = client.complete_messages.call_args.args[0][-1]["content"]
                self.assertIn(json.dumps(self.candidate(9)["next_configuration"]), request)
                self.assertEqual(opt.trial_cnt, 5)
                self.assertEqual(len(opt.configs), 0)
                selection = opt.decision_records[-1]
                self.assertEqual(selection["stage"], "candidate_selection")
                self.assertEqual(selection["candidates"][0]["prediction"]["mean"], default_mean)
                self.assertTrue(selection["candidates"][0]["default_prediction_refreshed"])

    def test_failed_default_prediction_keeps_the_usable_candidate(self):
        broken = {"next_configuration": self.candidate(3)["next_configuration"]}
        invalid_defaults = [broken, self.candidate(3), self.candidate(9, std=-1),
                            self.candidate(9, mean=None)]
        for default_reply in invalid_defaults:
            with self.subTest(default_reply=default_reply):
                opt, client = self.make([broken, broken, self.candidate(4, mean=2), default_reply])
                proposal = opt._suggest()
                self.assertEqual(proposal["config"]["fixture:depth"], 4)
                self.assertEqual(client.complete_messages.call_count, 4)

    def test_two_broken_candidates_share_one_default_prediction_request(self):
        broken = {"next_configuration": self.candidate(3)["next_configuration"]}
        for default_reply in (self.candidate(9, mean=7), broken):
            with self.subTest(default_reply=default_reply):
                opt, client = self.make([broken, broken, broken, broken, default_reply])
                proposal = opt._suggest()
                self.assertEqual(proposal["config"], dict(space().get_default_configuration()))
                expected = {"mean": 7, "std": .1} if "mean" in default_reply else None
                self.assertEqual(proposal["prediction"], expected)
                self.assertEqual(client.complete_messages.call_count, 5)
                stages = [record.get("stage") for record in opt.decision_records]
                self.assertEqual(stages.count("default_prediction_refresh"), 1)
                self.assertEqual(opt.trial_cnt, 5)

    def test_default_predictions_require_failure_probability_after_observed_failure(self):
        broken = {"next_configuration": self.candidate(3)["next_configuration"]}
        valid = dict(self.candidate(4, mean=2), failure_probability=.2)
        opt, client = self.make([broken, broken, valid, self.candidate(9, mean=1)])
        opt.semantic_history[-1] = {"configuration": valid["next_configuration"], "status": "failure"}
        proposal = opt._suggest()
        self.assertEqual(proposal["config"]["fixture:depth"], 4)
        request = client.complete_messages.call_args.args[0][-1]["content"]
        self.assertIn("failure_probability", request)

    def test_minimization_ei_and_success_weight(self):
        self.assertGreater(weighted_ei(2, 1, 0, 5), weighted_ei(4, 1, 0, 5))
        self.assertAlmostEqual(weighted_ei(2, 1, .9, 5), .1 * weighted_ei(2, 1, 0, 5))
        self.assertEqual(weighted_ei(2, 0, 0, 5), 3)


if __name__ == "__main__":
    unittest.main()
