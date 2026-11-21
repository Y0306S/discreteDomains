#!/usr/bin/env python3
"""Bounded, real TaskSet pipeline check for random, TPE and no-VAE LOLBO.

Each method evaluates six configurations on one task, stopping after three and
resuming in a fresh process. Each evaluation uses the BO corpus protocol:
1000 steps, evaluation every 1000 steps, one replica.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ConfigSpace import Configuration

from mindware.domains.dlhpo.taskset_oracle import DEFAULT_OBJECTIVE_METRIC, bo_protocol
from mindware.domains.dlhpo.taskset_space import space_from_taskset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-id", default="mlp_family_seed12")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists() or (output / args.task_id).exists():
        parser.error("Use a fresh output directory for a new component check")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    from mindware.domains.dlhpo.taskset_oracle import _load_wrapper

    source_root = _load_wrapper()["mod"].TASK_SET_ROOT
    runtime_root = output / "taskset_runtime"
    shutil.copytree(source_root / "task_set", runtime_root / "task_set",
                    ignore=shutil.ignore_patterns("configs", "__pycache__"))
    (runtime_root / "task_set/tasks/configs").symlink_to(source_root / "task_set/tasks/configs")
    (runtime_root / "task_set/optimizers/configs").mkdir()
    env["TASKSET_RUNTIME_ROOT"] = str(runtime_root)
    cs = space_from_taskset()
    protocol = bo_protocol()
    report = {
        "task_id": args.task_id, "diagnostic_only": True,
        "protocol": protocol, "objective_metric": DEFAULT_OBJECTIVE_METRIC,
        "evaluations_per_method": 6, "n_startup_trials": 2,
        "lolbo_batch_size": 2, "lolbo_n_candidates": 256,
        "python": sys.executable, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "gpu_enabled": False, "methods": {}, "passed": False,
        "taskset_source_root": str(source_root), "taskset_runtime_root": str(runtime_root),
    }
    started = time.monotonic()
    for method in ("random", "tpe", "lolbo"):
        begin = time.monotonic()
        row = {"passed": False}
        report["methods"][method] = row
        prefix = None
        run_dir = output / args.task_id / method / "seed7"
        try:
            for trials in (3, 6):
                command = [
                    sys.executable, str(Path(__file__).with_name("run_taskset_bo.py")),
                    "--optimizer", method, "--task-ids", args.task_id,
                    "--seed", "7", "--n-trials", str(trials), "--no-dummy",
                    "--training-steps", str(protocol["training_steps"]),
                    "--eval-every-n", str(protocol["eval_every_n"]),
                    "--num-replicas", str(protocol["num_replicas"]),
                    "--objective-metric", DEFAULT_OBJECTIVE_METRIC, "--n-startup-trials", "2",
                    "--lolbo-batch-size", "2", "--lolbo-n-candidates", "256",
                    "--warm-timeout", "180", "--replica-timeout", "180",
                    "--output-dir", str(output),
                ]
                log = output / f"{method}_{trials}.log"
                print(json.dumps({"method": method, "target_evaluations": trials, "log": str(log)}), flush=True)
                with log.open("w") as handle:
                    subprocess.run(command, env=env, cwd=ROOT, stdout=handle,
                                   stderr=subprocess.STDOUT, check=True)
                observations = json.loads((run_dir / "history.json").read_text())["observations"]
                if len(observations) != trials:
                    raise AssertionError(f"Expected {trials} evaluations, found {len(observations)}")
                if prefix is None:
                    prefix = observations
                elif observations[:3] != prefix:
                    raise AssertionError("Resume changed completed observations")

            for obs in observations:
                cs.check_configuration(Configuration(cs, values=obs["config"]))
                info = obs["extra_info"]["info"]
                if (info.get("oracle") != "train_inner" or info.get("protocol") != protocol
                        or info.get("objective_metric") != DEFAULT_OBJECTIVE_METRIC):
                    raise AssertionError(f"Wrong oracle/protocol: {info}")
            successes = sum(obs["trial_state"] == 0 and math.isfinite(obs["objectives"][0])
                            for obs in observations)
            row.update({
                "evaluations": len(observations), "successful_evaluations": successes,
                "all_configs_valid": True, "resume_preserved_history": True,
                "families": sorted({obs["config"]["optimizer_family"] for obs in observations}),
                "history": str(run_dir / "history.json"),
                "best_objective": min(obs["objectives"][0] for obs in observations),
            })
            if successes != 6:
                row["failed_evaluations"] = [obs for obs in observations if obs["trial_state"] != 0]
                raise AssertionError(f"Only {successes}/6 real evaluations succeeded")
            if method == "lolbo":
                from mindware.domains.bo_checkpoint import load_optuna_study
                from mindware.domains.dlhpo.taskset_lolbo import TaskSetLOLBOSampler

                study = load_optuna_study(run_dir, n_complete=6)
                if study is None or not isinstance(study.sampler, TaskSetLOLBOSampler):
                    raise AssertionError("LOLBO checkpoint has the wrong sampler")
                if study.sampler.n_model_updates < 2 or study.sampler._model is None:
                    raise AssertionError("LOLBO did not fit and update its GP")
                row.update(study.sampler.diagnostics())
                row["gp_proposals"] = sum(
                    obs["extra_info"]["info"]["proposal_phase"] == "gp_trust_region"
                    for obs in observations
                )
            row["passed"] = True
        except Exception as exc:
            row["error"] = str(exc)
        row["elapsed_seconds"] = round(time.monotonic() - begin, 2)
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"method": method, **row}), flush=True)
    report["passed"] = all(row["passed"] for row in report["methods"].values())
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
