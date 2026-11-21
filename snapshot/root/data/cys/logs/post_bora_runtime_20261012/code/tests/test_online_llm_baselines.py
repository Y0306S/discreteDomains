"""Offline regressions: synthetic replies/oracles, real ConfigSpaces and numerics."""

import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Constant
from ConfigSpace import CategoricalHyperparameter, UniformIntegerHyperparameter, EqualsCondition
from ConfigSpace import ForbiddenAndConjunction, ForbiddenEqualsClause
from openbox.utils.constants import FAILED, SUCCESS

from mindware.components.optimizers import ml_checkpoint
from mindware.components.optimizers.bora_optimizer import BORAOptimizer
from mindware.components.optimizers.online_llm_optimizer import OnlineLLMOptimizer
from mindware.components.optimizers.strict_recommendation import (
    GPTConfigurationExtractor, RecommendationError, StrictRecommendationResolver,
)
from mindware.components.optimizers.tosfit_optimizer import ToSFiTOptimizer
from mindware.components.optimizers.transfer_hpo.schema import cs_from_spec


ROOT = Path(__file__).resolve().parents[1]


def space():
    cs = ConfigurationSpace(seed=3)
    cs.add_hyperparameters([Constant("algorithm", "fixture"),
                           UniformIntegerHyperparameter("fixture:depth", 3, 15, default_value=9),
                           CategoricalHyperparameter("fixture:criterion", ["gini", "entropy"])])
    return cs


class Extractor:
    def __init__(self, assignments=(), ambiguous=False):
        self.assignments, self.ambiguous, self.calls = assignments, ambiguous, 0

    def extract(self, text, spec, mode="config"):
        self.calls += 1
        return {"assignments": [{"name": name, "value": value, "evidence": evidence}
                                for name, value, evidence in self.assignments],
                "ambiguous": self.ambiguous, "usage": {"total_tokens": 0}}


class ResolverTests(unittest.TestCase):
    def test_all_16_ml_schemas_valid_defaults_bypass_extraction(self):
        coverage = json.loads((ROOT / "benchmark_data/ml_train_80_transfer_v2/coverage.json").read_text())
        self.assertEqual(len(coverage["groups"]), 16)
        for group in coverage["groups"]:
            cs = cs_from_spec(json.loads(Path(group["dump"]).read_text())["configspace"])
            extractor = Extractor()
            resolver = StrictRecommendationResolver(cs, extractor)
            raw = json.dumps(dict(cs.get_default_configuration()))
            self.assertEqual(dict(resolver.resolve(raw)), dict(cs.get_default_configuration()))
            self.assertEqual(extractor.calls, 0)

    def test_partial_and_prose_go_to_extractor_before_defaults(self):
        for text in ('{"depth": 7}', 'I recommend depth 7.'):
            extractor = Extractor([("depth", 7, text)])
            resolver = StrictRecommendationResolver(space(), extractor)
            result = dict(resolver.resolve(text))
            self.assertEqual(extractor.calls, 1)
            self.assertEqual(result["fixture:depth"], 7)
            self.assertEqual(result["fixture:criterion"], "gini")
            self.assertEqual(resolver.last_record["defaulted"], ["algorithm", "fixture:criterion"])

    def test_invalid_values_use_defaults_after_transcription_without_clipping(self):
        cases = (("depth", -1, "out_of_range"), ("depth", 99, "out_of_range"),
                 ("depth", 3.5, "fractional_integer"), ("depth", True, "invalid_type"),
                 ("depth", "unlimited", "invalid_type"), ("depth", None, "invalid_type"),
                 ("criterion", "invalid", "invalid_category"))
        for name, value, reason in cases:
            with self.subTest(name=name, value=value):
                text = json.dumps({name: value})
                extractor = Extractor([(name, value, text)])
                resolver = StrictRecommendationResolver(space(), extractor)
                config = resolver.resolve(text)
                full_name = "fixture:" + name
                default = space()[full_name].default_value
                self.assertEqual(config[full_name], default)
                self.assertEqual(extractor.calls, 1)
                self.assertEqual(resolver.last_record["assignments"][full_name], value)
                self.assertEqual(resolver.last_record["repairs"][full_name],
                                 {"original": value, "replacement": default, "reason": reason})
                self.assertFalse(resolver.last_record["direct_valid"])

    def test_prose_repair_and_original_invalid_values_survive_extractor_omission(self):
        for raw in ('I recommend depth -1.', 'I recommend criterion invalid.',
                    'I recommend criterion null.', 'I recommend criterion true.',
                    'I recommend depth null.', '{"depth": [3, 7]}', '{"depth": {"bad": 7}}'):
            with self.subTest(raw=raw):
                resolver = StrictRecommendationResolver(space(), Extractor())
                config = resolver.resolve(raw)
                self.assertEqual(dict(config), dict(space().get_default_configuration()))
                self.assertEqual(len(resolver.last_record["repairs"]), 1)

    def test_repair_preserves_valid_parameters_and_rejects_unknown_names(self):
        raw = '{"depth": -1, "criterion": "entropy"}'
        resolver = StrictRecommendationResolver(space(), Extractor())
        config = resolver.resolve(raw)
        self.assertEqual(config["fixture:criterion"], "entropy")
        self.assertEqual(set(resolver.last_record["repairs"]), {"fixture:depth"})
        self.assertEqual(resolver.last_record["defaulted"], ["algorithm"])
        with self.assertRaisesRegex(RecommendationError, "Unknown or ambiguous"):
            resolver.resolve('{"unknown_parameter": 7}')

    def test_repair_does_not_resolve_forbidden_combinations(self):
        cs = space()
        cs.add_forbidden_clause(ForbiddenAndConjunction(
            ForbiddenEqualsClause(cs["fixture:depth"], 9),
            ForbiddenEqualsClause(cs["fixture:criterion"], "entropy")))
        resolver = StrictRecommendationResolver(cs, Extractor())
        with self.assertRaises(RecommendationError):
            resolver.resolve('{"depth": -1, "criterion": "entropy"}')
        self.assertFalse(resolver.last_record["accepted"])
        self.assertEqual(resolver.last_record["repairs"]["fixture:depth"]["replacement"], 9)

    def test_extractor_cannot_rewrite_invalid_or_invent_evidence(self):
        text = 'I recommend depth -1.'
        resolver = StrictRecommendationResolver(space(), Extractor([("depth", 3, text)]))
        with self.assertRaises(RecommendationError):
            resolver.resolve(text)
        resolver = StrictRecommendationResolver(space(), Extractor([("depth", 7, "depth 7")]))
        with self.assertRaises(RecommendationError):
            resolver.resolve("No recommendation.")

    def test_default_policy_repairs_json_without_losing_valid_values(self):
        extractor = Extractor()
        resolver = StrictRecommendationResolver(space(), extractor, default_on_error=True)
        result = resolver.resolve('{"depth": -1, "criterion": "entropy"}')
        self.assertEqual(result["fixture:depth"], 9)
        self.assertEqual(result["fixture:criterion"], "entropy")
        self.assertEqual(extractor.calls, 0)
        self.assertEqual(resolver.last_record["route"], "local_default_repair")
        self.assertNotIn("fallback", resolver.last_record)

    def test_whole_default_fallback_records_failed_extraction_and_forbidden_repair(self):
        cs = space()
        cs.add_forbidden_clause(ForbiddenAndConjunction(
            ForbiddenEqualsClause(cs["fixture:depth"], 9),
            ForbiddenEqualsClause(cs["fixture:criterion"], "entropy")))
        for raw, extractor in (
                ("No usable recommendation.", Extractor(ambiguous=True)),
                ('{"depth": -1, "criterion": "entropy"}', Extractor()),
                ("Try depth seven.", SimpleNamespace(extract=Mock(side_effect=TimeoutError("API timeout"))))):
            with self.subTest(raw=raw):
                resolver = StrictRecommendationResolver(cs, extractor, default_on_error=True)
                result = resolver.resolve(raw)
                self.assertEqual(dict(result), dict(cs.get_default_configuration()))
                record = resolver.last_record
                self.assertEqual(record["raw_text"], raw)
                self.assertEqual(record["fallback"], "configspace_default")
                self.assertTrue(record["accepted"])
                self.assertFalse(record["direct_valid"])
                self.assertTrue(record["fallback_reason"])

    def test_empty_extraction_is_explicit_default_fallback_and_selection_stays_strict(self):
        resolver = StrictRecommendationResolver(space(), Extractor(), default_on_error=True)
        result = resolver.resolve("I have no recommendation.")
        self.assertEqual(dict(result), dict(space().get_default_configuration()))
        self.assertEqual(resolver.last_record["fallback"], "configspace_default")
        with self.assertRaises(RecommendationError):
            resolver.select('{"indices": [99]}', space().sample_configuration(5), 1)

    def test_decoded_json_verifies_values_without_paraphrased_evidence(self):
        resolver = StrictRecommendationResolver(space(), Extractor([("depth", 7, "depth = 7")]))
        config = resolver.resolve('{"rationale": "The previous depth was 15.", "config": {"depth": 7}}')
        self.assertEqual(config["fixture:depth"], 7)
        self.assertEqual(resolver.last_record["extraction"]["assignments"][0]["verification"],
                         "decoded_original_config")
        resolver = StrictRecommendationResolver(space(), Extractor([("depth", -1, "depth = -1")]))
        config = resolver.resolve('{"depth": -1}')
        self.assertEqual(config["fixture:depth"], 9)
        self.assertEqual(resolver.last_record["extraction"]["assignments"][0]["value"], -1)

    def test_decoded_json_cannot_be_changed_or_extended_by_extractor(self):
        for raw, assignments in (( '{"depth": -1}', [("depth", 3, "depth = 3")]),
                                  ('{"depth": "bad"}', [("depth", 9, "depth = 9")]),
                                  ('{"depth": true}', [("depth", 1, "depth = 1")]),
                                  ('{"depth": 7}', [("criterion", "entropy", "criterion = entropy")])):
            resolver = StrictRecommendationResolver(space(), Extractor(assignments))
            with self.assertRaisesRegex(RecommendationError, "changed or invented"):
                resolver.resolve(raw)

    def test_categorical_extraction_uses_native_string_value(self):
        resolver = StrictRecommendationResolver(space(), Extractor([
            ("criterion", "entropy", "criterion = entropy")]))
        result = resolver.resolve('{"criterion": "entropy"}')
        self.assertEqual(result["fixture:criterion"], "entropy")
        self.assertEqual(result["fixture:depth"], space()["fixture:depth"].default_value)

    def test_ambiguous_duplicate_and_inactive_assignments_fail(self):
        resolver = StrictRecommendationResolver(space(), Extractor(ambiguous=True))
        with self.assertRaises(RecommendationError):
            resolver.resolve("depth 7 or 8")
        resolver = StrictRecommendationResolver(space(), Extractor())
        for raw in ('{"depth": -1, "depth": 7}', '{"depth": 7, "fixture:depth": 8}',
                    '{"depth": -1, "fixture:depth": 99}',
                    'I recommend criterion null, then criterion entropy.'):
            with self.assertRaises(RecommendationError):
                resolver.resolve(raw)
        cs = space()
        child = UniformIntegerHyperparameter("fixture:child", 1, 10, default_value=3)
        cs.add_hyperparameter(child)
        cs.add_condition(EqualsCondition(child, cs["fixture:criterion"], "entropy"))
        resolver = StrictRecommendationResolver(cs, Extractor())
        self.assertNotIn("fixture:child", dict(resolver.resolve("{}")))
        activated = resolver.resolve('{"criterion": "entropy"}')
        self.assertEqual(activated["fixture:child"], 3)
        with self.assertRaises(RecommendationError):
            resolver.resolve('{"criterion": "gini", "child": 4}')
        with self.assertRaises(RecommendationError):
            resolver.resolve('{"criterion": "invalid", "child": -1}')

    def test_selection_never_generates_or_repairs_candidates(self):
        cs = space()
        candidates = cs.sample_configuration(5)
        extractor = Extractor()
        resolver = StrictRecommendationResolver(cs, extractor)
        selected = resolver.select('{"indices": [3, 1]}', candidates, 2)
        self.assertIs(selected[0], candidates[3])
        self.assertEqual(extractor.calls, 0)
        for raw in ('{"depth": 10}', '{"indices": [-1, 0]}', '{"indices": [0, 0]}'):
            with self.assertRaises(RecommendationError):
                resolver.select(raw, candidates, 2)

    def test_openai_structured_output_routing_and_call_cap(self):
        completion = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
            content='{"assignments": [], "ambiguous": false}', refusal=None))], usage=None, model="gpt-4o-mini")
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=completion))))
        extractor = GPTConfigurationExtractor(max_tokens=128, timeout=10, max_calls=1, client=client)
        extractor.extract("{}", {})
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "gpt-4o-mini")
        self.assertTrue(kwargs["response_format"]["json_schema"]["strict"])
        item = kwargs["response_format"]["json_schema"]["schema"]["properties"]["assignments"]["items"]
        self.assertIn("value", item["required"])
        self.assertEqual({v["type"] for v in item["properties"]["value"]["anyOf"]},
                         {"string", "number", "boolean", "null"})
        instruction = kwargs["messages"][0]["content"]
        self.assertIn("hyperparameter configuration extraction", instruction)
        self.assertNotIn("assignments named 'index'", instruction)
        with self.assertRaises(RecommendationError):
            extractor.extract("{}", {})
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_extraction_call_budget_survives_crash_before_optimizer_checkpoint(self):
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=Mock(side_effect=RuntimeError("transport interrupted")))))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "usage.json"
            extractor = GPTConfigurationExtractor(max_tokens=128, timeout=10, max_calls=1,
                                                  client=client, budget_path=path)
            with self.assertRaises(RuntimeError):
                extractor.extract("{}", {})
            restarted = GPTConfigurationExtractor(max_tokens=128, timeout=10, max_calls=1,
                                                  client=client, budget_path=path)
            with self.assertRaises(RecommendationError):
                restarted.extract("{}", {})
            self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_extractor_uses_named_repository_route_lazily(self):
        route = {"api_key": "test-extractor-key", "base_url": "https://extractor.example/v1"}
        with patch("mindware.modules.configs.AVAILABLE_LLMs", {"gpt-4o-mini": route}), \
                patch("openai.OpenAI") as factory:
            extractor = GPTConfigurationExtractor(max_tokens=128, timeout=10, max_calls=1)
            factory.assert_not_called()
            factory.return_value.chat.completions.create.side_effect = RuntimeError("transport interrupted")
            with self.assertRaisesRegex(RuntimeError, "transport interrupted"):
                extractor.extract("{}", {})
            factory.assert_called_once_with(api_key=route["api_key"], base_url=route["base_url"],
                                            timeout=10, max_retries=0)


class MockRuntime:
    def __init__(self, cs):
        self.cs, self.proposals, self.observations = cs, 0, []
        self.policy_version, self.saved_version, self.saved_weights = 0, 0, None

    def propose(self, prompt, step):
        self.proposals += 1
        config = dict(self.cs.get_default_configuration())
        config["fixture:depth"] = 3 + step % 13
        return {"raw_text": json.dumps(config), "tokens": [100 + step],
                "features": [float(step), 1.0], "policy_version": step,
                "gradient_step": step >= 16}

    def observe(self, features, reward):
        self.observations.append((features, reward))

    def snapshot(self, directory):
        return {"proposals": self.proposals, "observations": copy.deepcopy(self.observations)}

    def restore(self, state):
        if state:
            self.proposals, self.observations = state["proposals"], state["observations"]


class MemoryHistory:
    def __init__(self):
        self.observations = []

    def update_observation(self, observation):
        self.observations.append(observation)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cs = space()
        self.kd = SimpleNamespace(configs=[], perfs=[], update_history=Mock(),
                                  bo_history_mab={"fixture": MemoryHistory()},
                                  fetch_best_conf_knowledge=Mock(side_effect=AssertionError("cross-task retrieval")))
        self.kd_patch = patch("mindware.components.optimizers.online_llm_optimizer.KnowledgeDatabase.instance",
                              return_value=self.kd)
        self.kd_patch.start()
        self.addCleanup(self.kd_patch.stop)
        self.saver_patch = patch.object(OnlineLLMOptimizer, "update_saver")
        self.saver_patch.start()
        self.addCleanup(self.saver_patch.stop)

    def make(self, cls=BORAOptimizer, limit=2, **params):
        return cls(evaluator=SimpleNamespace(fixed_config=None), config_space=self.cs,
                   name="hpo", output_dir=self.temp.name, evaluation_limit=limit,
                   extractor_client=Extractor(), logging_enabled=False,
                   llm_client=SimpleNamespace(complete=Mock(return_value=json.dumps(dict(self.cs.get_default_configuration())))),
                   n_pending=50, **params)

    @staticmethod
    def oracle(config, evaluator, limit, **kwargs):
        assert kwargs["force_isolation"] is True
        return SimpleNamespace(config=config, objectives=[-0.5], trial_state=SUCCESS, elapsed_time=0.01)

    def test_bora_constructor_fixed_algorithm_target_only_and_quota(self):
        opt = self.make()
        with patch("mindware.components.optimizers.online_llm_optimizer.wrapper_evaluater", side_effect=self.oracle) as evaluate:
            for _ in range(4):
                result = opt.iterate()
                self.assertEqual(len(result), 4)
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(len(opt.traj_rows), 2)
        self.kd.fetch_best_conf_knowledge.assert_not_called()
        self.assertIn("Qwen3.5", opt.chat_engine)
        self.assertEqual(len(self.make().configs), 0)

    def test_target_restarts_qwen_sampling_rng_from_its_seed(self):
        import torch
        self.make()
        first = torch.rand(4)
        self.make()
        self.assertTrue(torch.equal(first, torch.rand(4)))

    def test_bora_real_gp_and_forced_bo_after_assistant(self):
        opt = self.make(limit=8)
        with patch("mindware.components.optimizers.online_llm_optimizer.wrapper_evaluater", side_effect=self.oracle):
            for _ in range(5):
                opt.iterate()
            self.assertTrue(opt.initialized)
            self.assertTrue(np.isfinite(opt._fit_and_uncertainty()))
            opt.policy._failures = opt.policy.plateau_duration
            opt.policy._uncertainty = opt.policy._upper_unc_threshold * 2
            fresh = opt._pool(1)[0]
            with patch.object(opt, "_assistant_suggestions", return_value=[fresh]):
                opt.iterate()
            self.assertEqual(opt.traj_rows[-1]["note"], "ASSISTANT_SUGGESTS")
            opt.iterate()
            self.assertEqual(opt.traj_rows[-1]["note"], "BO")

    def test_failed_evaluations_consume_budget(self):
        opt = self.make()
        def failed(config, evaluator, limit, **kwargs):
            return SimpleNamespace(config=config, objectives=[float("inf")], trial_state=FAILED, elapsed_time=0.01)
        with patch("mindware.components.optimizers.online_llm_optimizer.wrapper_evaluater", side_effect=failed) as evaluate:
            opt.run()
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(opt.trial_cnt, 2)
        self.assertTrue(all(not np.isfinite(p) for p in opt.perfs))

    def test_bora_100_evaluations_despite_rejected_proposals_and_failed_fits(self):
        self.cs = ConfigurationSpace(seed=3)
        self.cs.add_hyperparameters([Constant("algorithm", "fixture"),
            UniformIntegerHyperparameter("fixture:depth", 3, 1000000, default_value=9)])
        opt = self.make(limit=100)
        opt.llm_client.complete.return_value = '{"unknown_parameter": 7}'
        calls = []

        def oracle(config, evaluator, limit, **kwargs):
            calls.append(dict(config))
            result = self.oracle(config, evaluator, limit, **kwargs)
            if len(calls) % 17 == 0:
                result.objectives, result.trial_state = [float("inf")], FAILED
            return result

        with patch("mindware.components.optimizers.online_llm_optimizer.wrapper_evaluater", side_effect=oracle), \
                patch.object(opt, "_fit_and_uncertainty", return_value=0.5), \
                patch.object(opt, "_bo_candidates", side_effect=lambda count: opt._pool(count)):
            opt.run()
            opt.iterate()
        self.assertEqual(len(calls), 100)
        self.assertEqual(opt.trial_cnt, 100)
        self.assertTrue(all(row["oracle_called"] for row in opt.traj_rows))
        self.assertTrue(opt.decision_records)
        self.assertTrue(any(record.get("stage") == "initial_comment_failed" for record in opt.decision_records))
        summary = ml_checkpoint.progress(opt.traj_rows, 100)
        self.assertEqual(summary["successful"], 95)
        self.assertEqual(summary["status"], "complete_with_failures")

    def test_bora_selection_does_not_exceed_last_evaluation_slot(self):
        opt = self.make(limit=7)
        with patch("mindware.components.optimizers.online_llm_optimizer.wrapper_evaluater", side_effect=self.oracle) as evaluate:
            for _ in range(6):
                opt.iterate()
            opt.policy._failures = opt.policy.plateau_duration
            opt.policy._uncertainty = (opt.policy._lower_unc_threshold + opt.policy._upper_unc_threshold) / 2
            with patch.object(opt, "_selection", side_effect=lambda candidates: candidates[:2]):
                opt.run()
        self.assertEqual(evaluate.call_count, 7)
        self.assertEqual(opt.traj_rows[-1]["note"], "ASSISTANT_SELECTS")
        self.assertEqual(len(opt.queue), 1)
        self.assertTrue(ml_checkpoint.progress(opt.traj_rows, 7)["clean_complete"])

    def test_bora_pending_resume_preserves_proposal_and_initial_batch(self):
        with patch.dict(os.environ, {"ML_RUN_ID": "bora-pending-test"}):
            first = self.make(limit=3)
            first._ml_session = ml_checkpoint.MLSession(first, self.temp.name)
            proposal = first._suggest()
            proposal["suggest_time"] = 0.0
            ml_checkpoint.prepare(first, proposal)
            second = self.make(limit=3)
            second._ml_session = ml_checkpoint.MLSession(second, self.temp.name)
            self.assertEqual(second._ml_session.restore(), 0)
            self.assertEqual(ml_checkpoint.pending(second), proposal)
            with patch("mindware.components.optimizers.online_llm_optimizer.wrapper_evaluater", side_effect=self.oracle) as evaluate:
                second.run()
            self.assertEqual(evaluate.call_count, 3)
            self.assertEqual(dict(evaluate.call_args_list[0].args[0]), proposal["config"])
            second.llm_client.complete.assert_not_called()
            self.assertIsNone(ml_checkpoint.pending(second))
            self.assertTrue(ml_checkpoint.progress(second.traj_rows, 3)["clean_complete"])

    def test_tosfit_cached_incumbent_does_not_consume_evaluations(self):
        runtime = MockRuntime(self.cs)
        opt = self.make(ToSFiTOptimizer, limit=5, tosfit_runtime=runtime)
        rewards = iter([-2.0, -5.0, -4.0, -6.0, -4.0])
        def oracle(config, evaluator, limit, **kwargs):
            return SimpleNamespace(config=config, objectives=[-next(rewards)], trial_state=SUCCESS, elapsed_time=0.01)
        with patch("mindware.components.optimizers.online_llm_optimizer.wrapper_evaluater", side_effect=oracle) as evaluate:
            opt.run()
        self.assertEqual(evaluate.call_count, 5)
        self.assertEqual(runtime.proposals, 5)
        self.assertEqual(len(runtime.observations), 5)
        self.assertEqual(runtime.observations[0][1], -2.0)
        cached = [r for r in opt.decision_records if r.get("stage") == "incumbent_cached"]
        self.assertEqual(len(cached), 4)
        self.assertTrue(all(r["source_trial"] == 0 and not r["oracle_called"]
                            and not r["evaluation_slot_consumed"] for r in cached))
        self.assertAlmostEqual(opt.solutions[0]["mean"], -2.0)
        self.assertEqual(opt.incumbent_perf, -2.0)

    def test_tosfit_unresolvable_proposal_has_no_ml_oracle_or_fake_reward(self):
        runtime = MockRuntime(self.cs)
        runtime.propose = Mock(return_value={"raw_text": '{"unknown_parameter": 7}', "tokens": [42],
                                            "features": [1.0, 1.0], "policy_version": 0, "gradient_step": False})
        opt = self.make(ToSFiTOptimizer, tosfit_runtime=runtime)
        with patch("mindware.components.optimizers.online_llm_optimizer.wrapper_evaluater") as evaluate:
            opt.run()
        evaluate.assert_not_called()
        self.assertEqual(runtime.observations, [])
        self.assertEqual(opt.trial_cnt, 2)
        self.assertEqual(opt.decision_records[0]["original_tokens"], [42])

    def test_both_adapters_evaluate_repaired_configs_and_keep_original_recommendations(self):
        raw = '{"depth": -1, "criterion": "entropy"}'
        for cls in (BORAOptimizer, ToSFiTOptimizer):
            with self.subTest(method=cls.optimizer_label):
                runtime = MockRuntime(self.cs)
                runtime.propose = Mock(return_value={"raw_text": raw, "tokens": [42],
                    "features": [1.0, 1.0], "policy_version": 0, "gradient_step": False})
                opt = self.make(cls, limit=1, tosfit_runtime=runtime)
                opt.llm_client.complete.return_value = json.dumps({"comment": "test", "hypotheses": [
                    {"name": "test", "rationale": "test", "confidence": "medium",
                     "points": [json.loads(raw)]}]}) if cls is BORAOptimizer else raw
                with patch("mindware.components.optimizers.online_llm_optimizer.wrapper_evaluater",
                           side_effect=self.oracle) as evaluate:
                    opt.run()
                self.assertEqual(evaluate.call_count, 1)
                evaluated = dict(evaluate.call_args.args[0])
                self.assertEqual(evaluated["fixture:depth"], 9)
                self.assertEqual(evaluated["fixture:criterion"], "entropy")
                record = next(r for r in opt.decision_records if "repairs" in r)
                self.assertTrue(record["accepted"])
                self.assertEqual(record["raw_text"], raw)
                self.assertEqual(record["assignments"]["fixture:depth"], -1)
                self.assertEqual(record["repairs"]["fixture:depth"]["replacement"], 9)
                if cls is ToSFiTOptimizer:
                    self.assertEqual(record["original_tokens"], [42])
                    self.assertEqual(runtime.observations, [([1.0, 1.0], 0.5)])

    def test_both_default_fallbacks_are_actual_evaluations_even_when_repeated(self):
        for cls in (BORAOptimizer, ToSFiTOptimizer):
            with self.subTest(method=cls.optimizer_label):
                runtime = MockRuntime(self.cs)
                runtime.propose = Mock(return_value={"raw_text": '{"unknown_parameter": 7}',
                    "tokens": [42], "features": [1.0, 1.0], "policy_version": 0,
                    "gradient_step": False})
                opt = self.make(cls, limit=3, tosfit_runtime=runtime,
                                recommendation_default_fallback=True)
                opt.llm_client.complete.return_value = '{"unknown_parameter": 7}'
                with patch("mindware.components.optimizers.online_llm_optimizer.wrapper_evaluater",
                           side_effect=self.oracle) as evaluate:
                    opt.run()
                self.assertEqual(evaluate.call_count, 3)
                self.assertEqual(opt.trial_cnt, 3)
                self.assertTrue(all(row["oracle_called"] for row in opt.traj_rows))
                self.assertTrue(all(dict(call.args[0]) == dict(self.cs.get_default_configuration())
                                    for call in evaluate.call_args_list))
                self.assertTrue(any(record.get("fallback") == "configspace_default"
                                    for record in opt.decision_records))
                if cls is ToSFiTOptimizer:
                    self.assertEqual(len(runtime.observations), 3)

    def test_default_fallback_policy_rejects_previous_policy_checkpoint(self):
        opt = self.make(recommendation_default_fallback=True)
        opt.asset_identity["recommendation_policy"] = "configspace-default-repair-v2"
        with self.assertRaisesRegex(RuntimeError, "different recommendation policy"):
            opt._checkpoint_restore(None)

    def test_old_rejection_policy_checkpoint_cannot_resume_as_default_repair(self):
        opt = self.make()
        opt.asset_identity = {"protocol": 1}
        with self.assertRaisesRegex(RuntimeError, "different recommendation policy"):
            opt._checkpoint_restore(None)

    def test_tosfit_pending_resume_does_not_repeat_policy_update(self):
        with patch.dict(os.environ, {"ML_RUN_ID": "offline-test"}):
            runtime = MockRuntime(self.cs)
            first = self.make(ToSFiTOptimizer, tosfit_runtime=runtime)
            session = ml_checkpoint.MLSession(first, self.temp.name)
            first._ml_session = session
            proposal = first._suggest()
            proposal["suggest_time"] = 0.0
            session.save(proposal)
            ml_checkpoint.commit(first, completed=False)
            self.assertIsNotNone(session.pending)
            restored_runtime = MockRuntime(self.cs)
            second = self.make(ToSFiTOptimizer, tosfit_runtime=restored_runtime)
            resumed = ml_checkpoint.MLSession(second, self.temp.name)
            second._ml_session = resumed
            self.assertEqual(resumed.restore(), 0)
            with patch("mindware.components.optimizers.online_llm_optimizer.wrapper_evaluater", side_effect=self.oracle):
                second.iterate()
            self.assertEqual(restored_runtime.proposals, 1)
            self.assertEqual(len(restored_runtime.observations), 1)
            self.assertEqual(second.trial_cnt, 1)
            self.assertIsNone(resumed.pending)


if __name__ == "__main__":
    unittest.main()
