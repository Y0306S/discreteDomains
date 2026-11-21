"""TaskSet LLM-RGPE infrastructure: prompts, parser, rerank and the hybrid loop.

Runs offline: dummy TaskSet oracle, scripted LLM replies, fake RGPE arm and fake
source predictions. No model, GPU or TaskSet training is used.
"""

import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from mindware.components.optimizers.experience_gate import ExplorePolicy
from mindware.domains.dlhpo.taskset_adapter import TaskSetAdapter
from mindware.domains.dlhpo.taskset_llm import experience as E
from mindware.domains.dlhpo.taskset_llm import loop as L
from mindware.domains.dlhpo.taskset_llm import p5_prompt as P
from mindware.domains.dlhpo.taskset_llm import spec as S
from mindware.domains.dlhpo.taskset_llm.llm_client import ScriptedClient
from mindware.domains.dlhpo.taskset_space import space_from_taskset

TASK = "mlp_family_seed0"
SPECS = {TASK: {"task_family": "mlp_family", "spec": {"layer_sizes": [32], "dataset": ["cifar10", {"bs": 64}]}},
         "mlp_family_seed1": {"task_family": "mlp_family", "spec": {"layer_sizes": [64], "dataset": ["cifar10", {"bs": 32}]}},
         "mlp_family_seed2": {"task_family": "mlp_family", "spec": {"layer_sizes": [16, 16], "dataset": ["svhn", {"bs": 64}]}}}


def sample(cs, seed):
    cs.seed(seed)
    return dict(cs.sample_configuration().get_dictionary())


def reply(config):
    return json.dumps({"config": config})


class SpecTest(unittest.TestCase):
    def setUp(self):
        self.cs = space_from_taskset()

    def test_canonical_configs_round_trip(self):
        for seed in range(50):
            config = sample(self.cs, seed)
            values, notes = S.coerce_config(config, self.cs)
            self.assertEqual(values, config)
            self.assertFalse(any(notes.values()))

    def test_lenient_repairs_like_ml(self):
        values, notes, _ = S.parse_reply('```json\n{"config": {"optimizer_family": "adam4p_wide_grid", '
                                         '"learning_rate": 99, "adam4p_wide_grid:epsilon": 1e-3, '
                                         '"nadamw_grid:learning_rate": 0.1, "bogus": 1}}\n```', self.cs)
        self.assertEqual(values["adam4p_wide_grid:learning_rate"], 10.0)
        self.assertIn("adam4p_wide_grid:learning_rate", notes["clipped"])
        self.assertEqual(sorted(notes["defaulted"]), ["adam4p_wide_grid:beta1_inv", "adam4p_wide_grid:beta2_inv"])
        self.assertEqual(notes["other_family"], ["nadamw_grid:learning_rate"])
        self.assertEqual(notes["skipped"], ["bogus"])
        with self.assertRaises(ValueError):
            S.parse_reply('{"optimizer_family": "sgd"}', self.cs)

    def test_prompt_names_no_optimizer_and_matches_nothink_template(self):
        messages = S.warm_start_messages(TASK, SPECS, self.cs)
        task_section = messages[1]["content"].split("OPTIMIZATION")[0]
        self.assertNotIn("adam", task_section)
        rendered = S.render_nothink(messages)
        self.assertTrue(rendered.endswith("<|im_end|>\n<|im_start|>assistant\n"))
        self.assertNotIn("<think>", rendered)


class PromptTest(unittest.TestCase):
    def setUp(self):
        self.cs = space_from_taskset()
        rng = random.Random(0)
        self.rows = [{"trial": i + 1, "config": sample(self.cs, i), "loss": rng.random(), "successful": True}
                     for i in range(60)]

    def test_history_selection_and_family_recent3(self):
        _, user, meta = P.build_p5_prompt(TASK, SPECS, self.cs, self.rows[:40], trial=41, seed=1)
        tags = meta["selected_tags"]
        self.assertEqual(sum("best" in t for t in tags.values()), 5)
        self.assertEqual(sum("worst" in t for t in tags.values()), 5)
        self.assertEqual(sum("recent" in t for t in tags.values()), 15)
        for family, trials in meta["family_recent_trials"].items():
            own = [r["trial"] for r in self.rows[:40] if r["config"]["optimizer_family"] == family]
            self.assertEqual(trials, own[-3:])
        self.assertIsNone(meta["experience_branch"])
        self.assertNotIn("Good/Bad regions", user)
        self.assertNotIn("Exploration counts", user)

    def test_explore_exploit_after_trial_50(self):
        branches = {}
        for trial in range(50, 61):
            _, user, meta = P.build_p5_prompt(TASK, SPECS, self.cs, self.rows[:trial - 1], trial=trial, seed=1)
            branches[trial] = meta["experience_branch"]
            if meta["experience_branch"] == "explore":
                self.assertIn("Exploration counts per optimizer family", user)
                self.assertNotIn("Good/Bad regions", user)
            else:
                self.assertIn("Good/Bad regions", user)
                self.assertNotIn("Exploration counts", user)
        self.assertEqual(set(branches.values()), {"explore", "exploit"})

    def test_coin_is_the_ml_fifty_fifty_policy(self):
        for seed in (1, 2):
            for trial in range(50, 101):
                policy = ExplorePolicy(mode="fifty_fifty", seed=seed, p_default=0.5)
                ml = policy.choose(trial=trial, successes=[], mode_tag="marginal_nonoverlap")
                self.assertEqual(P.explore_branch(seed, trial), ml)


class RerankTest(unittest.TestCase):
    def test_floor_ties_and_window(self):
        cs = space_from_taskset()
        cards = {t: {"problem": {"task_id": t, "task_family": "mlp_family",
                                 "features": E.spec_features(SPECS[t]["spec"])}} for t in SPECS}
        stats = E.profile_stats({t: c["problem"]["features"] for t, c in cards.items()})
        rows = [{"trial": i + 1, "config": sample(cs, i), "loss": float(i), "successful": True} for i in range(4)]
        predictions = {"mlp_family_seed1": [0, 1, 2, 3], "mlp_family_seed2": [3, 2, 1, 0]}
        audit = E.rerank(TASK, rows, cards, stats, SPECS, lambda pool, configs: predictions)
        self.assertEqual([h["task"] for h in audit["hits"]], ["mlp_family_seed1"])
        self.assertNotIn(TASK, [s["task"] for s in audit["scored"]])
        few = E.rerank(TASK, rows[:2], cards, stats, SPECS, lambda pool, configs: predictions)
        self.assertEqual(few["skipped"], "too_few_or_constant_observations")
        self.assertEqual([t for t in range(1, 12) if E.retrieval_window(t)], [4, 5, 6, 7, 8])


class LoopTest(unittest.TestCase):
    def make(self, run_dir, llm, budget=30, bank=None, predict=None):
        adapter = TaskSetAdapter(task={"task_id": TASK}, dummy=True)
        cs = adapter.search_space()
        presets = [sample(cs, 1000 + i) for i in range(3)]
        self.rgpe_calls = []

        def rgpe(observations, trial):
            self.rgpe_calls.append(trial)
            seen = {S.signature(o["config"]) for o in observations}
            for seed in range(10 ** 6):
                config = sample(cs, 7919 * trial + seed)
                if S.signature(config) not in seen:
                    return config, {"phase": "fake_rgpe"}

        return L.TaskSetLLMRGPE(task_id=TASK, cs=cs, evaluate=adapter.evaluate_config, run_dir=run_dir, llm=llm,
                                presets=presets, specs=SPECS, rgpe_suggest=rgpe, bank=bank, source_predict=predict,
                                identity={"test": True}, budget=budget, seed=1), cs

    def test_schedule_retries_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            cs = space_from_taskset()
            counter = iter(range(10 ** 6))
            first = sample(cs, 1000)  # equals preset 1 -> always a duplicate

            def answer(system, user):
                n = next(counter)
                if 1 <= n <= 5:  # trial 5 (calls 1-5): five duplicates -> RGPE fills the trial
                    return reply(first)
                if n == 6:  # trial 6: unparseable -> random credited to the LLM arm
                    return "no json here"
                return reply(sample(cs, 50000 + n))

            llm = ScriptedClient(answer)
            loop, _ = self.make(run_dir, llm)
            for _ in range(12):
                loop.step()
            rows = [o["extra_info"] for o in loop.observations]
            self.assertEqual([r["source"] for r in rows[:3]], ["llm_warm_start"] * 3)
            self.assertEqual([r["selected_arm"] for r in rows[3:8]], ["llm"] * 5)
            self.assertEqual(rows[4]["source"], "rgpe")
            self.assertTrue(rows[4]["meta"]["llm_retry_exhausted"])
            self.assertEqual(rows[5]["meta"]["configuration_source"], "random")
            self.assertEqual(rows[5]["source"], "llm")
            for i in range(9, 12):  # cooldown: an LLM-selected trial is followed by RGPE
                if rows[i - 1]["selected_arm"] == "llm":
                    self.assertEqual(rows[i]["selected_arm"], "rgpe")
            retry_prompts = [c["user"] for c in llm.calls[2:6]]
            self.assertNotIn("Retry reminder", llm.calls[1]["user"])
            self.assertTrue(all("Retry reminder" in p for p in retry_prompts))
            self.assertEqual(loop.selector.attempt_counts()["llm"],
                             sum(r["selected_arm"] == "llm" for r in rows[3:]) )

            # Crash between proposal and evaluation, then resume with a fresh object.
            pending = loop.propose(len(loop.observations) + 1)
            L._atomic_json(loop.pending_path, pending)
            resumed, _ = self.make(run_dir, ScriptedClient(lambda s, u: (_ for _ in ()).throw(AssertionError("no LLM"))))
            event = resumed.step()
            self.assertEqual(resumed.observations[-1]["config"], pending["config"])
            self.assertEqual(event["trial"], 13)
            self.assertEqual(resumed.selector.snapshot().keys(), loop.selector.snapshot().keys())

            with self.assertRaises(ValueError):
                L.TaskSetLLMRGPE(task_id=TASK, cs=cs, evaluate=None, run_dir=run_dir, llm=llm,
                                 presets=[first] * 3, specs=SPECS, rgpe_suggest=None,
                                 identity={"test": False}, budget=30)

    def test_full_budget_unique_configs_and_retrieval_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            cs = space_from_taskset()
            counter = iter(range(10 ** 6))
            cards = {t: {"problem": {"task_id": t, "task_family": "mlp_family",
                                     "features": E.spec_features(SPECS[t]["spec"])},
                         "families": {"nadamw_grid": {"best_loss": 0.1, "n": 5, "median_loss": 0.3}},
                         "family_ranking": ["nadamw_grid"],
                         "result": {"best_loss": 0.1, "best_config": sample(cs, 3)},
                         "best_family": "nadamw_grid", "regions": {"family_level": None, "by_family": {}}} for t in SPECS if t != TASK}
            stats = E.profile_stats({t: c["problem"]["features"] for t, c in cards.items()})
            predict_calls = []

            oracle = TaskSetAdapter(task={"task_id": TASK}, dummy=True)

            def predict(pool, configs):  # perfectly agreeing source surrogates
                predict_calls.append(len(configs))
                return {t: [oracle.evaluate_config(c).objective for c in configs] for t in pool}

            llm = ScriptedClient(lambda s, u: reply(sample(cs, 90000 + next(counter))))
            loop, _ = self.make(Path(tmp), llm, budget=40, bank=(cards, stats), predict=predict)
            loop.run()
            configs = [S.signature(o["config"]) for o in loop.observations]
            self.assertEqual(len(configs), 40)
            self.assertEqual(len(set(configs)), 40)
            self.assertTrue(all(math.isfinite(o["objectives"][0]) for o in loop.observations))
            self.assertEqual(len(predict_calls), sum(1 for o in loop.observations[3:8]))
            prompts_with_cards = [c["user"] for c in llm.calls if "Similar task:" in c["user"]]
            self.assertTrue(prompts_with_cards)
            self.assertTrue(json.loads((Path(tmp) / "progress.json").read_text())["status"] == "complete")



class TrainingDataTest(unittest.TestCase):
    def setUp(self):
        self.cs = space_from_taskset()
        losses = [0.9, 0.8, 0.85, 0.7, 0.75, 0.72, 0.6, 0.95, 0.61, 0.5]
        self.rows = [{"trial": i + 1, "config": sample(self.cs, 300 + i), "loss": y, "successful": True}
                     for i, y in enumerate(losses)]

    def test_improvement_events_start_after_warm_start(self):
        from mindware.domains.dlhpo.taskset_llm import training_data as T
        events = T.improvement_events(self.rows)
        self.assertEqual([e["trial"] for e in events], [4, 7, 10])
        self.assertTrue(all(e["delta_rel_score"] > 0 for e in events))
        messages, meta = T.prefix_prompt(TASK, SPECS, self.cs, self.rows, 7)
        self.assertIn("rank is the 1-based standing among all 6 evaluated trials", messages[1]["content"])
        self.assertEqual(json.loads(T.answer(events[1]["config"]))["config"], events[1]["config"])

    def test_attribution_pair_repairs_only_blamed_same_family(self):
        from mindware.domains.dlhpo.taskset_llm import training_data as T
        good = sample(self.cs, 1)
        family = good["optimizer_family"]
        bad = dict(good)
        names = S.family_params(self.cs, family)
        target = next(n for n in names if not n.endswith(("_is_zero", "_is_one")) and "use_nesterov" not in n)
        hp = self.cs.get_hyperparameter(target)
        bad[target] = float(hp.upper) if good[target] < (hp.lower + hp.upper) / 2 else float(hp.lower)
        rows = [{"trial": 1, "config": bad, "loss": 0.9}, {"trial": 2, "config": good, "loss": 0.2}]
        event = {"config": good, "loss": 0.2}
        pair = T.attribution_pair(self.cs, rows, event)
        if len(names) == 1:  # adam1p: blaming the only parameter exceeds the 50% cap
            self.assertIsNone(pair)
        else:
            self.assertEqual(pair["blamed"], [target])
            self.assertEqual(pair["chosen"], good)
            self.assertEqual(pair["rejected"], bad)

    def test_grpo_train_members_v14(self):
        import importlib.util
        path = Path(__file__).resolve().parents[2] / "examples/benchmark/domains/collect_taskset_grpo_rollouts.py"
        spec = importlib.util.spec_from_file_location("collector", path)
        C = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(C)
        rec = lambda r, imp=False, ex=False: {"reward": {"reward": r, "delta_best": r}, "improver": imp,
                                              "excluded_from_training": ex}
        members, status = C.train_members([rec(0.1), rec(0.2), rec(0.0, ex=True), rec(0.3), rec(0.4), rec(0.9, True)],
                                          random.Random(0))
        self.assertEqual(status, "ok")
        self.assertIn(5, members)
        self.assertEqual(len(members), 4)
        self.assertEqual(C.train_members([rec(0.1)] * 4, random.Random(0)), ([], "no_reward_signal"))
        adapter = TaskSetAdapter(task={"task_id": TASK}, dummy=True)
        prefix = [{"trial": i + 1, "config": sample(self.cs, i), "loss": 5.0} for i in range(3)]
        from rl_reward_utils import RewardWeights
        dup = C.score_sample(adapter, self.cs, prefix, reply(prefix[0]["config"]), RewardWeights())
        self.assertEqual(dup["evaluation_status"], "DUPLICATE")
        bad = C.score_sample(adapter, self.cs, prefix, "nope", RewardWeights())
        self.assertEqual((bad["evaluation_status"], bad["reward"]["reward"]), ("PARSE_FAILURE", -0.5))
        ok = C.score_sample(adapter, self.cs, prefix, reply(sample(self.cs, 99)), RewardWeights())
        self.assertEqual(ok["evaluation_status"], "SUCCESS")


if __name__ == "__main__":
    unittest.main()
