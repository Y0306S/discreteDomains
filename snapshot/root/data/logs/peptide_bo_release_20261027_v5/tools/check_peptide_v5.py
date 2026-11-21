#!/usr/bin/env python3
"""Real inference, joint-update, bounded-recenter and fresh-interpreter recovery checks."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import torch
import peptide_bo_v5 as runner
from mindware.domains.protein import verified_bo as bo
from mindware.domains.protein import peptide_v5 as protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--server", required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    torch.set_num_threads(1)
    production = runner.verify(args.production)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(passed=False, recipe_sha256=bo.digest(args.production / "recipe.json"), checks={}, tasks=[])
    report_path = args.output / "report.json"
    bo.atomic_json(report_path, report)
    tested = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests/domains"),
        "-p", "test_peptide_v5.py", "-v"], env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), check=True)
    report["checks"]["unit_tests"] = tested.returncode == 0
    pools = json.loads(Path(production["prepared_pool"]).read_text())
    for task, pool in pools.items():
        assert len(set(pool["sequences"])) == production["initialization"]
        for x, y in zip(pool["sequences"], pool["scores"]):
            bo.validate_sequence(x)
            assert bo.similarity(x, pool["reference"]) >= 0.75
            assert torch.isfinite(torch.tensor(y))
    report["checks"]["strict_initialization"] = True
    case = args.output / "remote_resume"
    case.mkdir()
    recipe = dict(production, purpose="component_check", proposals=11, fresh_budget=400, batch_size=4, e2e_every=2)
    bo.atomic_json(case / "recipe.json", recipe)
    command = [sys.executable, str(ROOT / "tools/peptide_bo_v5.py"), "run-task", "--campaign", str(case),
        "--task", "pep_000", "--cpu", str(args.cpu), "--server", args.server, "--stop-after-events", "3"]
    subprocess.run(command, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    sys.path.insert(0, recipe["cpu_upstream"])
    saved = torch.load(case / "runs/pep_000/checkpoint.pt", map_location="cpu", weights_only=False)
    assert saved["state"]["tot_num_e2e_updates"] >= 1
    original = torch.load(recipe["vae"], map_location="cpu", weights_only=False)
    changed = sum(not torch.equal(v, original[k]) for k, v in saved["vae"].items())
    assert changed > 0
    report["vae_changed_tensors"] = changed
    report["checks"]["vae_gradient_update"] = True
    report["checks"]["remote_cpu_worker"] = True
    # A cached scored acquisition exercises recovery independently of the active recenter cursor.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    state, backend = protocol.create_state(recipe, "pep_000", saved, args.server)
    progress, rows = bo.restore(state, saved, saved["identity"])
    backend.dirty()
    plan = progress["recenter_plan"]
    assert plan is not None
    end = min(len(plan["inputs"]), plan["cursor"] + plan["batch"])
    z = bo.encode(state.objective, plan["inputs"][plan["cursor"]:end])
    with torch.no_grad():
        sequences = state.objective.vae_decode(z)
    sequences = protocol.bounded_prefix(sequences, state.objective.xs_to_scores_dict, recipe["fresh_budget"] - progress["fresh_evaluations"])
    pending = dict(identity=saved["identity"], event=progress["events"] + 1, phase="recenter",
        sequences=sequences, z=z[:len(sequences)], next_cursor=end,
        before_oracle=bo.snapshot(state, saved["identity"], progress, rows))
    pending["scored"] = bo.score_batch(state.objective, sequences)
    bo.atomic_torch(case / "runs/pep_000/pending.pt", pending)
    backend.close()
    del state, backend
    gc.collect()
    replay = """import sys; from pathlib import Path; from types import SimpleNamespace
sys.path.insert(0, sys.argv[1]); import peptide_bo_v5 as runner
def bad(*a, **k): raise RuntimeError('Scored pending event must not be rescored')
runner.bo.score_batch=bad
runner.run_task(SimpleNamespace(campaign=Path(sys.argv[2]), task='pep_000', server=sys.argv[3], launch=False, stop_after_events=4))
"""
    subprocess.run([sys.executable, "-c", replay, str(ROOT / "tools"), str(case), args.server], check=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), cwd=str(ROOT))
    replayed = torch.load(case / "runs/pep_000/checkpoint.pt", map_location="cpu", weights_only=False)
    assert replayed["progress"]["events"] == 4
    assert replayed["progress"]["fresh_evaluations"] == saved["progress"]["fresh_evaluations"] + pending["scored"]["fresh_evaluations"]
    report["checks"]["fresh_process_resume"] = True
    report["checks"]["pending_without_rescore"] = True
    subprocess.run(command[:-2], check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    identity = dict(format=protocol.FORMAT, recipe_sha256=bo.digest(case / "recipe.json"), task="pep_000")
    history = runner.verify_archive(case / "runs/pep_000.tar.gz", identity)
    assert history["recenter_fresh"] > 0 and history["fresh_evaluations"] <= recipe["fresh_budget"]
    report["checks"]["bounded_recenter"] = True
    report["checks"]["archive"] = True
    report["tasks"].append({k: history[k] for k in ["proposals", "recenter_proposals", "fresh_evaluations", "recenter_fresh", "best_feasible_mic"]})
    # Run through the original first contraction threshold, without changing its tolerance.
    diagnostic = args.output / "contraction"
    diagnostic.mkdir()
    recipe = dict(production, purpose="component_check", proposals=1800, fresh_budget=5000, batch_size=50, e2e_every=10)
    bo.atomic_json(diagnostic / "recipe.json", recipe)
    subprocess.run([sys.executable, str(ROOT / "tools/peptide_bo_v5.py"), "run-task", "--campaign", str(diagnostic),
        "--task", "pep_003", "--cpu", str(args.cpu), "--gpu", str(args.gpu)], check=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu)))
    identity = dict(format=protocol.FORMAT, recipe_sha256=bo.digest(diagnostic / "recipe.json"), task="pep_003")
    history = runner.verify_archive(diagnostic / "runs/pep_003.tar.gz", identity)
    assert history["rounds"] >= 32
    if history["current_tr_length"] >= 0.8:
        raise RuntimeError("Diagnostic did not verify an actual trust-region contraction")
    report["checks"]["trust_region_contraction"] = True
    report["checks"]["real_acquisition"] = True
    report["diagnostic"] = {k: history[k] for k in ["rounds", "proposals", "fresh_evaluations", "recenter_fresh", "current_tr_length", "best_feasible_mic"]}
    report["diagnostic"]["feasible_new"] = sum(row["feasible"] for row in history["observations"] if row["phase"] != "initialization")
    report["passed"] = all(report["checks"].values())
    bo.atomic_json(report_path, report)
    bo.atomic_json(args.production / "preflight_passed.json", dict(recipe_sha256=report["recipe_sha256"],
        report=str(report_path), report_sha256=bo.digest(report_path)))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
