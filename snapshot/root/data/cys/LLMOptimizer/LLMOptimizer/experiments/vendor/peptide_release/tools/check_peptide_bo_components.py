#!/usr/bin/env python3
"""Bounded real-GPU peptide BO check. Never evaluates a production campaign."""

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import peptide_bo_campaign as campaign
from mindware.domains.protein import verified_bo as bo


def assert_weights_equal(model, weights):
    current = model.state_dict()
    assert current.keys() == weights.keys()
    assert all(torch.equal(current[k].detach().cpu(), value) for k, value in weights.items())


def check(args):
    started = time.monotonic()
    production = campaign.verify_recipe(args.production)
    recipe = campaign.verify_recipe(args.campaign)
    if recipe["purpose"] != "component_check" or (recipe["initialization"], recipe["proposals"],
            recipe["batch_size"], recipe["e2e_every"]) != (100, 11, 4, 2):
        raise ValueError("Component recipe must use 100 init, 11 proposals, batch 4, E2E every 2 rounds")
    for key in ("code", "environment", "vae", "pool_manifest", "init_policy", "e2e_policy",
                "center_policy", "gp_initial_epochs", "gp_update_epochs", "learning_rate",
                "latent_dimensions", "similarity", "oracle"):
        if recipe[key] != production[key]:
            raise ValueError("Component and production differ at " + key)
    if (args.campaign / "runs").exists():
        raise ValueError("Use a fresh component campaign for a complete diagnostic")
    result = dict(passed=False, recipe_sha256=bo.digest(args.production / "recipe.json"),
                  component_recipe_sha256=bo.digest(args.campaign / "recipe.json"),
                  checks={}, tasks=[], archives={}, oracle_spot_checks=[])
    bo.atomic_json(args.report, result)
    spec = importlib.util.spec_from_file_location("protocol_tests", ROOT / "tests/domains/test_verified_peptide_bo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    with campaign.resources(args):
        tested = unittest.TextTestRunner(verbosity=1).run(suite)
        if not tested.wasSuccessful():
            raise RuntimeError("Protocol unit tests failed")
        result["checks"]["unit_tests"] = True
        result["unit_test_count"] = tested.testsRun
        result["hardware"] = dict(host=campaign.socket.gethostname(), gpu=args.gpu, cpus=args.cpus,
                                  gpu_name=torch.cuda.get_device_name(0))
        first = "pep_000"
        task_args = SimpleNamespace(campaign=args.campaign, task=first, launch=False,
                                    threads=len(args.cpus.split(",")), stop_after_rounds=2)
        campaign.run_task(task_args)
        directory = args.campaign / "runs" / first
        saved = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=False)
        assert saved["progress"]["rounds"] == 2 and saved["progress"]["proposals"] == 8
        assert saved["state"]["tot_num_e2e_updates"] == 1
        original = torch.load(recipe["vae"], map_location="cpu", weights_only=False)
        changed = sum(not torch.equal(value, original[key]) for key, value in saved["vae"].items())
        assert changed > 0
        result["checks"]["vae_update"] = True
        result["vae_changed_tensors"] = changed
        state = bo.create_state(recipe, first, saved)
        progress, rows = bo.restore(state, saved, saved["identity"])
        assert_weights_equal(state.model, saved["model"])
        assert_weights_equal(state.objective.vae, saved["vae"])
        for model, weights in zip(state.c_models, saved["constraints"]):
            assert_weights_equal(model, weights)
        assert state.tr_state.failure_counter == saved["state"]["tr_state"].failure_counter
        assert state.tr_state.length == saved["state"]["tr_state"].length
        random_values = (random.random(), float(np.random.rand()), float(torch.rand(1)), float(torch.rand(1, device="cuda")))
        bo.restore_rng(saved["rng"])
        assert random_values == (random.random(), float(np.random.rand()), float(torch.rand(1)), float(torch.rand(1, device="cuda")))
        bo.restore_rng(saved["rng"])
        with torch.no_grad():
            prediction = state.model.posterior(state.train_z[:5].cuda()).mean.cpu()
        assert torch.isfinite(prediction).all()
        bo.restore(state, saved, saved["identity"])
        with torch.no_grad():
            restored_prediction = state.model.posterior(state.train_z[:5].cuda()).mean.cpu()
        assert torch.allclose(prediction, restored_prediction, atol=1e-5, rtol=1e-5)
        result["checks"]["resume"] = True
        result["checks"]["encode_only"] = (len(state.train_x) == 108
            and len(rows) == 108 and state.objective.num_calls == progress["fresh_evaluations"])
        assert result["checks"]["encode_only"]
        # Persist a real, scored final partial batch, then simulate restart before commit.
        bo.restore(state, saved, saved["identity"])
        state.update_surrogate_model()
        z, sequences = bo.propose(state, 3)
        pending = dict(identity=saved["identity"], round=3, sequences=sequences, z=bo.cpu_copy(z),
            before_oracle=bo.snapshot(state, saved["identity"], progress, rows),
            oracle_attempts=[dict(started_at=time.time(), diagnostic=True)])
        pending["scored"] = bo.score_batch(state.objective, sequences)
        bo.atomic_torch(directory / "pending.pt", pending)
        del state, original
        gc.collect()
        torch.cuda.empty_cache()
        task_args.stop_after_rounds = None
        # A fresh interpreter must load pickled upstream classes before constructing models.
        resume_code = """
import sys
from unittest.mock import patch
sys.path.insert(0, sys.argv.pop(1))
import peptide_bo_campaign as campaign
assert 'lolbo' not in sys.modules
with patch.object(campaign, 'score_batch', side_effect=AssertionError('Scored pending batch was re-evaluated')):
    campaign.main()
"""
        subprocess.run([sys.executable, "-c", resume_code, str(ROOT / "tools"), "run-task",
            "--campaign", str(args.campaign), "--task", first,
            "--threads", str(task_args.threads)], check=True, cwd=ROOT,
            env=dict(os.environ, PEPTIDE_WORKER_PARENT=str(os.getpid())))
        result["checks"]["fresh_process_resume"] = True
        path = args.campaign / "runs" / (first + ".tar.gz")
        history = campaign.verify_archive(path, saved["identity"])
        assert history["observations"][:108] == rows
        assert history["rounds"] == 3 and len(history["observations"]) == 111
        assert sum(row["round"] == 3 for row in history["observations"]) == 3
        result["checks"]["partial_batch"] = True
        result["checks"]["gp_acquisition"] = True
        result["checks"]["archive"] = True
        result["pending_resume_without_rescoring"] = True
        campaign.run_task(task_args)  # archive-aware completion check, no model construction
        result["tasks"].append({key: history[key] for key in
            ("reference", "rounds", "proposals", "fresh_evaluations", "cache_hits", "e2e_updates", "best_feasible_mic")})
        result["archives"][str(path)] = bo.digest(path)
        second = "pep_050"
        task_args.task = second
        campaign.run_task(task_args)
        path = args.campaign / "runs" / (second + ".tar.gz")
        other = campaign.verify_archive(path)
        assert history["reference"] != other["reference"]
        assert other["e2e_updates"] == 1 and other["rounds"] == 3 and other["proposals"] == 11
        result["checks"]["two_references"] = True
        result["tasks"].append({key: other[key] for key in
            ("reference", "rounds", "proposals", "fresh_evaluations", "cache_hits", "e2e_updates", "best_feasible_mic")})
        result["archives"][str(path)] = bo.digest(path)
        # These six extra oracle predictions are diagnostics, outside the two BO budgets.
        from apex_oracle import apex_wrapper
        for task in (first, second):
            _, sequences, scores = bo.load_pool(recipe, task)
            with torch.no_grad():
                observed = -apex_wrapper(sequences[:3])[:, 0]
            np.testing.assert_allclose(observed, scores[:3], rtol=1e-4, atol=1e-3)
            result["oracle_spot_checks"].append(dict(task=task, count=3,
                max_absolute_error=float(np.max(np.abs(observed - scores[:3])))))
        result["checks"]["oracle_scores"] = True
        result["peak_gpu_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
        result["peak_gpu_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
    result["seconds"] = time.monotonic() - started
    result["passed"] = all(result["checks"].values())
    result["launch_held"] = (args.production / "launch_hold.json").exists()
    bo.atomic_json(args.report, result)
    bo.atomic_json(args.production / "preflight_passed.json", dict(
        recipe_sha256=result["recipe_sha256"], report=str(args.report), report_sha256=bo.digest(args.report)))
    # A passing component check does not remove a separate scientific launch hold.
    if not result["launch_held"]:
        campaign.verify_preflight(args.production)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--cpus", required=True)
    check(parser.parse_args())
