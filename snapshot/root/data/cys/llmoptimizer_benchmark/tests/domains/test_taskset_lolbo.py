"""CASH validity, real LOLBO proposals, exact budgets and deterministic resume."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import optuna
import torch
from ConfigSpace import Configuration

from mindware.components.optimizers.lolbo_sampler import LOLBOTurboSampler
from mindware.domains import get_adapter
from mindware.domains.bo_checkpoint import load_optuna_study
from mindware.domains.dlhpo.taskset_lolbo import TaskSetCodec, TaskSetLOLBOSampler
from mindware.domains.dlhpo.taskset_space import FAMILY_HP, TASKSET_FAMILIES
from mindware.domains.search import _optuna_distributions, run_domain_search


def adapter():
    return get_adapter("taskset", task={"task_id": "mlp_family_seed12"}, dummy=True)


def configs(result):
    return [o["config"] for o in result["observations"]]


class TestTaskSetLOLBO(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def test_fixed_codec_and_projection_cover_every_family(self):
        cs = adapter().search_space()
        codec = TaskSetCodec(cs, _optuna_distributions(cs))
        defaults = {hp.name: hp.default_value for hp in cs.get_hyperparameters()}
        for family in TASKSET_FAMILIES:
            cfg = {k: v for k, v in defaults.items() if k.startswith(family + ":")}
            cfg[FAMILY_HP] = family
            vector = codec.encode(cfg)
            decoded = codec.decode(vector)
            self.assertEqual(set(cfg), set(decoded))
            np.testing.assert_allclose(vector, codec.encode(decoded), atol=1e-6)
            candidates = torch.rand(60, codec.dim)
            parent = codec.slices[FAMILY_HP]
            candidates[:, parent] = 0
            for i in range(60):
                candidates[i, parent.start + i % len(TASKSET_FAMILIES)] = 1
            projected = codec.project_candidates(candidates, torch.tensor(vector))
            families = set()
            for row in projected.numpy():
                values = codec.decode(row)
                cs.check_configuration(Configuration(cs, values=values))
                np.testing.assert_allclose(row, codec.encode(values), atol=1e-6)
                families.add(values[FAMILY_HP])
            self.assertEqual(families, set(TASKSET_FAMILIES))
            changed = dict(cfg)
            lr_name = family + ":learning_rate"
            changed[lr_name] = cs.get_hyperparameter(lr_name).lower
            self.assertFalse(np.array_equal(vector, codec.encode(changed)))

    def test_all_three_methods_use_exactly_100_evaluations(self):
        for method in ("random", "tpe", "lolbo"):
            with self.subTest(method=method):
                task = adapter()
                with patch.object(task, "evaluate", wraps=task.evaluate) as evaluate:
                    result = run_domain_search(
                        task, optimizer=method, seed=7, n_trials=100,
                        n_startup_trials=2, lolbo_batch_size=4, lolbo_n_candidates=64,
                    )
                self.assertEqual(evaluate.call_count, 100)
                self.assertEqual(len(result["observations"]), 100)
                for obs in result["observations"]:
                    self.assertEqual(obs["trial_state"], 0)
                    task.search_space().check_configuration(Configuration(task.search_space(), values=obs["config"]))
                    self.assertEqual(obs["extra_info"]["source"], method)
                if method == "lolbo":
                    info = result["observations"][-1]["extra_info"]["info"]
                    self.assertEqual(info["optimizer_backend"], "lolbo_no_vae_cash")
                    self.assertEqual(info["proposal_phase"], "gp_trust_region")
                    self.assertGreater(info["surrogate_fit_count"], 20)
                    self.assertEqual(info["modeled_parameters"], len(task.search_space().get_hyperparameters()))

    def test_random_seeds_and_resume(self):
        full = run_domain_search(adapter(), optimizer="random", seed=7, n_trials=9)
        different = run_domain_search(adapter(), optimizer="random", seed=8, n_trials=9)
        self.assertNotEqual(configs(full), configs(different))
        part = run_domain_search(adapter(), optimizer="random", seed=7, n_trials=4)
        resumed = run_domain_search(adapter(), optimizer="random", seed=7, n_trials=9,
                                    resume_observations=part["observations"])
        self.assertEqual(configs(full), configs(resumed))

    def test_random_resume_during_pending_evaluation(self):
        full = run_domain_search(adapter(), optimizer="random", seed=7, n_trials=9)
        with tempfile.TemporaryDirectory() as td:
            task = adapter()
            evaluate = task.evaluate
            count = 0

            def interrupt(*args, **kwargs):
                nonlocal count
                count += 1
                if count == 4:
                    raise RuntimeError("simulated interruption")
                return evaluate(*args, **kwargs)

            with patch.object(task, "evaluate", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    run_domain_search(task, optimizer="random", seed=7, n_trials=9, output_dir=Path(td))
            observations = json.loads((Path(td) / "history.json").read_text())["observations"]
            self.assertEqual(len(observations), 3)
            resumed = run_domain_search(adapter(), optimizer="random", seed=7, n_trials=9,
                                        output_dir=Path(td), resume_observations=observations)
            self.assertEqual(configs(full), configs(resumed))

    def test_lolbo_resume_preserves_gp_rng_and_pending_batch(self):
        params = dict(optimizer="lolbo", seed=7, n_startup_trials=2,
                      lolbo_batch_size=4, lolbo_n_candidates=64)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            full = run_domain_search(adapter(), n_trials=13, **params)
            part = run_domain_search(adapter(), n_trials=7, output_dir=out, **params)
            saved = load_optuna_study(out, n_complete=7)
            self.assertIsInstance(saved.sampler, TaskSetLOLBOSampler)
            self.assertIsNotNone(saved.sampler._model)
            self.assertGreater(len(saved.sampler._pending), 0)
            torch.manual_seed(99999)
            observations = json.loads((out / "history.json").read_text())["observations"]
            resumed = run_domain_search(adapter(), n_trials=13, output_dir=out,
                                        resume_observations=observations, **params)
            self.assertEqual(configs(full), configs(resumed))
            self.assertEqual(resumed["resumed_from"], len(part["observations"]))

    def test_lolbo_rejects_checkpoint_from_other_sampler(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            prior = run_domain_search(adapter(), optimizer="tpe", n_trials=3, output_dir=out)
            with self.assertRaisesRegex(ValueError, "matching optimizer checkpoint"):
                run_domain_search(adapter(), optimizer="lolbo", n_trials=5, output_dir=out,
                                  resume_observations=prior["observations"])

    def test_cli_cannot_skip_completed_mislabeled_lolbo_history(self):
        from examples.benchmark.domains.run_taskset_bo import main

        with tempfile.TemporaryDirectory() as td:
            task = adapter()
            out = Path(td) / task.task_id() / "lolbo" / "seed1"
            run_domain_search(task, optimizer="tpe", n_trials=3, output_dir=out)
            with self.assertRaisesRegex(ValueError, "old or unverified LOLBO"):
                main(["--optimizer", "lolbo", "--n-trials", "3", "--task-ids", task.task_id(),
                      "--output-dir", td, "--dummy"])

    def test_flat_ml_sampler_still_uses_gp(self):
        sampler = LOLBOTurboSampler(n_init=2, bsz=2, n_candidates=64, device="cpu")
        study = optuna.create_study(sampler=sampler, direction="minimize")
        for _ in range(7):
            trial = study.ask()
            value = trial.suggest_float("x", 0.01, 1.0, log=True)
            category = trial.suggest_categorical("category", ["a", "b"])
            study.tell(trial, (value - 0.2) ** 2 + (category == "b"))
        self.assertIsNotNone(sampler._model)
        self.assertGreater(sampler._n_seen, 2)


if __name__ == "__main__":
    unittest.main()
