#!/usr/bin/env python3
"""Exercise CPU LOLBO past surrogate initialization with real ML evaluations."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

import ml_fixed_campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = ml_fixed_campaign.load(args.campaign / "tasks_132.json")
    task = next(t for t in tasks if t["dataset"] == "car" and t["algorithm"] == "extra_trees")
    argv, env, _ = ml_fixed_campaign.command(args.campaign, "lolbo", task, [40, 41, 42, 43], None)
    os.environ.update(env)
    import torch
    import run_toy_categorical_optimizers as runner
    from mindware.modules.base import BaseAutoML
    if torch.cuda.is_available():
        raise RuntimeError("CPU validation unexpectedly has CUDA enabled")
    argv = argv[4:]
    for flag, path in (("--output-dir", args.output / "outputs"),
                       ("--history-dir", args.output / "history"),
                       ("--session-dir", args.output / "session")):
        argv[argv.index(flag) + 1] = str(path)
    argv.remove("--resume")
    models = []
    original_run = BaseAutoML.run

    def short_run(model, *positional, **kwargs):
        if model.optimizer.evaluation_num_limit != 100:
            raise RuntimeError("Validation must preserve the production attempt budget")
        models.append(model)
        model.amount_of_resource = 12
        return original_run(model, *positional, **kwargs)

    with patch.object(sys, "argv", argv), patch.object(BaseAutoML, "run", short_run):
        try:
            runner.main()
        except RuntimeError as exc:
            if not str(exc).endswith("stopped before reaching 100 trials"):
                raise
    progress = ml_fixed_campaign.load(args.output / "session/progress.json")
    if (progress["attempts"], progress["successful"], progress["failed"]) != (12, 12, 0):
        raise RuntimeError(f"CPU evaluations failed: {progress}")
    sampler = models[0].optimizer.study.sampler
    if str(sampler._device) != "cpu" or sampler._model is None or not sampler._initial_fit_done:
        raise RuntimeError("CPU LOLBO surrogate did not train")
    with Path(progress["trajectory"]).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 12 or any(not json.loads(r["config_json"]) or not math.isfinite(float(r["perf"])) for r in rows):
        raise RuntimeError("Missing saved configuration or performance")
    report = {"passed": True, "purpose": "CPU correctness validation; excluded from production results",
              "device": "cpu", "surrogate_trained": True, "actual_evaluations": 12,
              "successful": 12, "failed": 0, "trajectory": progress["trajectory"]}
    ml_fixed_campaign.save(args.output / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
