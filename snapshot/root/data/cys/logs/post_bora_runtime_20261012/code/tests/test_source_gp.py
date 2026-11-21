"""Correctness checks for source-only retrieval and durable RS evidence."""

import copy
import json
import logging
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from mindware.components.experience.retrieve import ExperienceBank, retrieve_for_run
from mindware.components.experience.source_gp import (
    SourceGPBank, SourceGPError, attach_source_gp, build_artifact, canonical_score,
    configuration, digest, fanova_advice, rs_position, space_digest, successful_observations,
    subtract_good_region,
)
from mindware.components.experience.surrogate_landscape import get_algo_cs


def card(name, task="regression", usable=True, engine="fanova_full_cs"):
    cs = get_algo_cs("adaboost", task)
    cs.seed(7)
    trials = [{"trial": i + 1, "config": c.get_dictionary(), "score": float(i + 1)}
              for i, c in enumerate(cs.sample_configuration(8))]
    imps = {"learning_rate": 0.4, "max_depth": 0.3} if usable else {}
    factors = [{"name": key, "good_region": {"low": 1, "high": 2}}
               for key in imps]
    return {"id": name, "problem": {"dataset_id": name, "algorithm_id": "adaboost",
            "task_type": task, "metric": "mse" if task == "regression" else "bal_acc",
            "higher_is_better": task != "regression"},
            "retrieval": {"trials": trials}, "delivery": {},
            "knowledge": {"fanova": {"engine": engine, "importance_1d": imps,
                                      "importance_total": imps}, "factors_all": factors}}


class FixedModel:
    def __init__(self, values):
        self.values = np.asarray(values)

    def predict(self, X):
        assert len(X) == len(self.values)
        return self.values.reshape(-1, 1), np.ones((len(X), 1))


def ranked_bank(cards, predictions):
    cs = get_algo_cs("adaboost", "regression")
    bank = ExperienceBank(experiences=cards)
    ranker = SourceGPBank.__new__(SourceGPBank)
    ranker.bank = bank
    ranker.manifest_sha256 = "fixture"
    ranker.models = {c["id"]: FixedModel(predictions[c["id"]]) for c in cards}
    ranker.spaces = {("adaboost", "regression"): (cs, space_digest(cs))}
    ranker.records = {c["id"]: {"id": c["id"], "dataset": c["id"], "algorithm": "adaboost",
                     "task_type": c["problem"]["task_type"], "status": "ready",
                     "space_sha256": space_digest(cs), "sha256": c["id"]} for c in cards}
    bank.source_gp = ranker
    return bank


def observations():
    return [{**row, "trial": index, "score": score, "source": source}
            for row, index, score, source in zip(card("target")["retrieval"]["trials"],
                                                (1, 3, 5), (-9.0, -4.0, -1.0),
                                                ("llm_warm_start", "llm_warm_start", "tpe"))]


class RankingTest(unittest.TestCase):
    def test_fanova_one_parameter_suffices_without_old_task_filters(self):
        source = card("single")
        source["delivery"]["usable_for_prompt"] = False
        source["knowledge"]["fanova"]["importance_1d"] = {"learning_rate": 0.06}
        source["knowledge"]["fanova"]["importance_total"] = {"learning_rate": 0.9}
        block, decision = fanova_advice(source)
        self.assertIn("learning_rate", block)
        self.assertEqual(decision["keep_hps"], ["learning_rate"])
        self.assertTrue(decision["usable_for_prompt"])

    def test_fanova_shows_all_important_parameters_and_skips_only_missing_ranges(self):
        source = card("all")
        names = [f"parameter_{i}" for i in range(7)]
        source["knowledge"]["fanova"]["importance_1d"] = {name: 0.05 for name in names}
        source["knowledge"]["fanova"]["importance_1d"].update(weak=0.049, no_range=0.8)
        source["knowledge"]["factors_all"] = [
            {"name": name, "good_region": {"low": 1, "high": 2}} for name in names + ["weak"]]
        block, decision = fanova_advice(source)
        self.assertEqual(len(decision["approved_parameters"]), 7)
        self.assertNotIn("weak:", block)
        self.assertNotIn("no_range:", block)
        self.assertEqual(len(block.splitlines()), 8)

    def test_fanova_regions_are_valid_and_prompt_recommendation_is_unambiguous(self):
        source = card("regions")
        source["knowledge"]["factors_all"] = [
            {"name": "learning_rate", "good_region": {"low": 1, "high": 2},
             "bad_region": {"low": 1, "high": 3}},
            {"name": "max_depth", "good_region": {"low": 9, "high": 2}}]
        block, decision = fanova_advice(source)
        self.assertEqual(decision["keep_hps"], ["learning_rate"])
        self.assertIn("prefer [1, 2]; avoid (2, 3]", block)
        self.assertNotIn("max_depth:", block)
        self.assertIn("bad_region", decision["approved_parameters"][0])

    def test_bad_region_difference_preserves_good_boundaries_and_disconnected_parts(self):
        good = {"low": 2, "high": 4}
        parts = subtract_good_region({"low": 1, "high": 5}, good)
        self.assertEqual(parts, [
            {"low": 1, "high": 2, "low_inclusive": True, "high_inclusive": False},
            {"low": 4, "high": 5, "low_inclusive": False, "high_inclusive": True}])
        self.assertEqual(good, {"low": 2, "high": 4})
        self.assertEqual(subtract_good_region({"low": 2, "high": 4}, good), [])
        self.assertEqual(subtract_good_region({"low": 3, "high": 3}, good), [])
        self.assertEqual(subtract_good_region({"low": 1, "high": 2}, good)[0]["high_inclusive"], False)
        self.assertEqual(subtract_good_region({"low": 0, "high": 1}, good)[0]["high_inclusive"], True)
        self.assertEqual(subtract_good_region(["a", "b", "c"], ["b"]), [["a", "c"]])
        self.assertEqual(subtract_good_region(["b"], ["b"]), [])
        self.assertEqual(subtract_good_region(None, good), [])

    def test_rank_all_then_fill_advice_beyond_old_pool(self):
        cards = [card(f"bad_{i:02}", usable=False) for i in range(13)]
        cards += [card("proxy", engine="proxy_too_few")]
        cards += [card(f"valid_{i}") for i in range(4)]
        bank = ranked_bank(cards, {c["id"]: [1, 3, 2] if c["id"].startswith("valid") else [1, 2, 3]
                                  for c in cards})
        block, hits, audit = bank.source_gp.rank("target", "adaboost", "regression", observations())
        self.assertEqual(len(audit["candidates"]), 18)
        self.assertEqual(len(hits), 3)
        self.assertTrue(all(h.exp_id.startswith("valid_") for h in hits))
        self.assertGreater(hits[0].features["rank"], 12)
        self.assertNotIn("proxy", block)
        self.assertNotIn('"config"', block)
        self.assertIn("learning_rate", block)
        self.assertIn("choose the next configuration from the full search space", block)

    def test_tied_ranking_is_seeded_and_independent_of_bank_order(self):
        cards = [card(f"source_{i}") for i in range(8)]
        predictions = {c["id"]: [1, 2, 3] for c in cards}
        bank, reversed_bank = ranked_bank(cards, predictions), ranked_bank(cards[::-1], predictions)
        selections = []
        for seed in range(8):
            _, hits, audit = bank.source_gp.rank("target", "adaboost", "regression", observations(), seed=seed)
            _, other, _ = reversed_bank.source_gp.rank("target", "adaboost", "regression", observations(), seed=seed)
            selections.append(tuple(h.exp_id for h in hits))
            self.assertEqual(selections[-1], tuple(h.exp_id for h in other))
            self.assertEqual(audit["tie_break_seed"], seed)
        self.assertGreater(len(set(selections)), 1)

    def test_three_successful_results_qualify_even_when_configurations_repeat(self):
        bank = ranked_bank([card("source")], {"source": [1, 2, 1]})
        rows = observations()
        rows[-1]["config"] = dict(rows[0]["config"])
        rows[-1]["score"] = rows[0]["score"]
        block, hits, audit = bank.source_gp.rank("target", "adaboost", "regression", rows)
        self.assertEqual(audit["n_distinct_configurations"], 2)
        self.assertEqual(audit["status"], "selected")
        self.assertEqual(audit["n_observations"], 3)
        self.assertTrue(block)
        self.assertEqual(len(hits), 1)

    def test_window_tpe_slots_insufficient_and_constant_evidence(self):
        bank = ranked_bank([card("source")], {"source": [1, 2, 3]})
        warm_start = [{**row, "trial": i} for i, row in enumerate(observations(), 1)]
        for trial in (1, 3, 4, 8, 9, 100):
            audit = {}
            with patch("mindware.components.experience.retrieve.attach_semantic_vectors",
                       side_effect=AssertionError("semantic retrieval called")):
                block, _ = retrieve_for_run(bank, dataset_id="target", algorithm_id="adaboost",
                    task_type="regression", dataset_profile=None, successes=warm_start,
                    ranker="source_gp", start_trial=4, stop_trial=9, current_trial=trial, audit=audit)
            self.assertEqual(bool(block), 4 <= trial <= 8)
            self.assertEqual(audit["rs_position"], trial - 3 if 4 <= trial <= 8 else None)
        self.assertEqual(rs_position(5, source="tpe")["status"], "tpe_slot_no_prompt")
        self.assertEqual(bank.source_gp.rank("target", "adaboost", "regression", observations()[:2])[2]["status"],
                         "insufficient_successful_observations")
        tied = [{**t, "score": 1.0} for t in observations()]
        self.assertEqual(bank.source_gp.rank("target", "adaboost", "regression", tied)[2]["status"],
                         "constant_target_scores")

    def test_rejects_future_or_duplicated_target_observations(self):
        bank = ranked_bank([card("source")], {"source": [1, 2, 3]})
        for rows in (observations(), [observations()[0]] * 3):
            with self.assertRaisesRegex(SourceGPError, "unique completed evaluations"):
                retrieve_for_run(bank, dataset_id="target", algorithm_id="adaboost",
                    task_type="regression", dataset_profile=None, successes=rows,
                    ranker="source_gp", start_trial=4, stop_trial=9, current_trial=4)

    def test_floor_constant_prediction_and_same_task_type(self):
        cards = [card("negative"), card("constant"), card("boundary"), card("wrong_type", "classification")]
        bank = ranked_bank(cards, {"negative": [3, 2, 1], "constant": [1, 1, 1],
                                   "boundary": [1, 3, 2], "wrong_type": [1, 2, 3]})
        _, hits, audit = bank.source_gp.rank("target", "adaboost", "regression", observations())
        self.assertEqual([h.exp_id for h in hits], ["boundary"])
        self.assertEqual(len(audit["candidates"]), 3)

    def test_global_ids_and_tpe_observations_reach_actual_prompt(self):
        from mindware.components.optimizers.pt4opt_llm_optimizer import build_history_prompt_block
        obs = observations()
        rows = [{"iter_idx": t["trial"] - 1, "perf": t["score"], "config_json": json.dumps(t["config"]),
                 "stage": t["source"]} for t in obs]
        rows.insert(1, {"iter_idx": 1, "perf": float("-inf"), "stage": "llm_warm_start"})
        saved = successful_observations(rows)
        self.assertEqual([t["trial"] for t in saved], [1, 3, 5])
        bank = ranked_bank([card("source")], {"source": [1, 2, 3]})
        prompt, meta = build_history_prompt_block([], [], "adaboost", get_algo_cs("adaboost", "regression"),
            current_trial=6, prompt_blocks={"experience_retrieve": True}, experience_bank=bank,
            retrieval_observations=saved, experience_retrieve_ranker="source_gp",
            experience_retrieve_start_trial=4, experience_retrieve_stop_trial=9,
            dataset_id="target", task_type="regression")
        self.assertIn("Similar task: source", prompt)
        self.assertEqual(meta["experience_retrieval"]["observations"], saved)
        self.assertEqual(meta["experience_retrieval"]["rs_position"], 3)
        self.assertEqual(meta["selected_best_trials"], [5, 3])

    def test_missing_artifact_never_falls_back(self):
        from mindware.components.optimizers.pt4opt_llm_optimizer import PT4OptLLMOptimizer
        opt = SimpleNamespace(_build_prompts=lambda **kwargs: (_ for _ in ()).throw(SourceGPError("missing")),
                              fallback_on_transport_error=True, fallback_on_bad_json=True)
        with self.assertRaises(SourceGPError):
            PT4OptLLMOptimizer._suggest_config(opt, 3)

    def test_adaboost_inner_algorithm_and_score_direction(self):
        cs = get_algo_cs("adaboost", "classification")
        raw = {"adaboost:" + k: v for k, v in cs.get_default_configuration().get_dictionary().items()}
        raw["algorithm"] = "adaboost"
        self.assertEqual(configuration(cs, raw, "adaboost")["algorithm"], raw["adaboost:algorithm"])
        self.assertEqual(canonical_score(9.0, False), -9.0)
        self.assertEqual(canonical_score(0.9, True), 0.9)
        with self.assertRaises(SourceGPError):
            configuration(cs, {}, "adaboost")


class ArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.cards = [card("source_cls", "classification"), card("source_rgs")]
        cls.bank_path = cls.root / "bank.jsonl"
        cls.bank_path.write_text("\n".join(json.dumps(c) for c in cls.cards))
        cls.train = cls.root / "train.json"
        cls.test = cls.root / "test.json"
        cls.train.write_text(json.dumps([{"dataset": c["id"], "algorithm": "adaboost"} for c in cls.cards]))
        cls.test.write_text(json.dumps([{"dataset": "target", "algorithm": "adaboost"}]))
        cls.cache = cls.root / "artifact"
        cls.payload = build_artifact(cls.bank_path, cls.train, cls.test, cls.cache, workers=2)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def load(self):
        bank = ExperienceBank.load(self.bank_path)
        return attach_source_gp(bank, self.bank_path, self.cache,
                                expected_train_sha256=digest(self.train))

    def test_production_path_requires_exact_fixed_training_manifest(self):
        with self.assertRaisesRegex(SourceGPError, "approved fixed training split"):
            attach_source_gp(ExperienceBank.load(self.bank_path), self.bank_path, self.cache)
        output = self.root / "wrong_fixed_split"
        with self.assertRaisesRegex(SourceGPError, "approved fixed training split"):
            build_artifact(self.bank_path, self.train, self.test, output,
                           expected_train_sha256="different_manifest")
        self.assertFalse(output.exists())

    def test_real_gp_build_load_and_regression_orientation(self):
        bank = self.load()
        for task in ("classification", "regression"):
            bank.source_gp.preflight("adaboost", task)
            c = next(c for c in self.cards if c["problem"]["task_type"] == task)
            obs = [{**t, "score": canonical_score(t["score"], c["problem"]["higher_is_better"])}
                   for t in c["retrieval"]["trials"]]
            _, hits, audit = bank.source_gp.rank("target", "adaboost", task, obs)
            self.assertEqual(len(hits), 1)
            self.assertGreater(audit["candidates"][0]["spearman"], 0.99)

    def test_rejects_contamination_before_creating_artifact(self):
        output = self.root / "contaminated"
        with self.assertRaises(SourceGPError):
            build_artifact(self.bank_path, self.train, self.train, output)
        self.assertFalse(output.exists())

    def test_rejects_missing_changed_and_incompatible_artifacts(self):
        for name in (self.payload["sources"][0]["file"], self.payload["train"]["file"], "manifest.json"):
            path = self.cache / name
            original = path.read_bytes()
            try:
                path.unlink()
                with self.assertRaises(SourceGPError):
                    self.load()
                path.write_bytes(b"changed")
                with self.assertRaises(SourceGPError):
                    self.load()
            finally:
                path.write_bytes(original)
        bank = self.load()
        bank.source_gp.records["source_rgs"]["space_sha256"] = "different"
        with self.assertRaises(SourceGPError):
            bank.source_gp.preflight("adaboost", "regression")


class EvidenceTest(unittest.TestCase):
    def test_failed_result_and_tpe_metadata_do_not_reuse_previous_llm_audit(self):
        from mindware.components.optimizers.pt4opt_llm_tpe_optimizer import PT4OptLLMTPEOptimizer
        cs = get_algo_cs("adaboost", "regression")
        with tempfile.TemporaryDirectory() as directory:
            opt = SimpleNamespace(logging_enabled=True, llm_query_cost=[], llm_query_time=[],
                start_wall_time=time.time(), traj_rows=[], perfs=[], output_dir=directory,
                fixed_algorithm="adaboost", llm_prob=0.5, llm_phase_trials=0, config_space=cs,
                traj_csv_path=str(Path(directory) / "trial_traj.csv"), study=None,
                logger=logging.getLogger(__name__), _last_history_meta={"stale": True})
            meta = {"experience_retrieval": rs_position(5, source="tpe")}
            PT4OptLLMTPEOptimizer._record_hybrid_event(opt, "tpe", 4, cs.get_default_configuration(),
                                                      float("-inf"), 0, 0, 0, 3, 3, None, "failed", meta)
            result = json.loads((Path(directory) / "retrieval/trial_0005.json").read_text())
            self.assertFalse(result["successful"])
            self.assertIsNone(result["performance"])
            self.assertEqual(result["status"], "tpe_slot_no_prompt")
            self.assertEqual(result["selected_ids"], [])

    def test_pending_checkpoint_preserves_proposal_and_retrieval_evidence(self):
        from mindware.components.optimizers.ml_checkpoint import MLSession
        from ConfigSpace import ConfigurationSpace
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"ML_RUN_ID": "rs-test"}):
            def optimizer():
                kd = SimpleNamespace(configs=[], perfs=[])
                return SimpleNamespace(config_space=ConfigurationSpace(seed=2), KD=kd,
                    configs=kd.configs, perfs=kd.perfs, evaluator=SimpleNamespace(),
                    evaluation_num_limit=100, traj_rows=[], traj_csv_path=str(Path(directory) / "t.csv"))
            original = optimizer()
            proposal = {"cfg": {"depth": 2}, "meta": {"experience_retrieval": {
                **rs_position(4), "selected_ids": ["source"], "observations": observations()}}}
            MLSession(original, directory).save(copy.deepcopy(proposal))
            resumed = MLSession(optimizer(), directory)
            resumed.restore()
            self.assertEqual(resumed.pending, proposal)


if __name__ == "__main__":
    unittest.main()
