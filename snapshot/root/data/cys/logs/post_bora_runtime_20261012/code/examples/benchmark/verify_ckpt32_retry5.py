"""Live nine-evaluation check of the new ckpt32 policy with a 100-evaluation budget."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

import ml_fixed_campaign as campaign


def preflight(root, output):
    from mindware.components.experience.retrieve import ExperienceBank
    from mindware.components.experience.source_gp import attach_source_gp, configuration
    from mindware.components.experience.surrogate_landscape import get_algo_cs

    bank_path = root / "experience_bank.jsonl"
    bank = attach_source_gp(ExperienceBank.load(bank_path), bank_path, root / "source_gp_train80_v1")
    presets = campaign.load(root / "warm_start_presets.json")
    verified = []
    for task in campaign.load(root / "tasks_132.json"):
        kind = "classification" if task["task_type"] == "cls" else "regression"
        bank.source_gp.preflight(task["algorithm"], kind)
        cs = get_algo_cs(task["algorithm"], kind)
        key = f"{task['dataset']}:{task['algorithm']}"
        configs = [configuration(cs, raw, task["algorithm"]) for raw in presets[key]]
        if len(configs) != 3:
            raise RuntimeError(f"Invalid warm starts for {key}")
        observations = [{"trial": i + 1, "score": float(i), "source": "synthetic_preflight",
                         "config": c.get_dictionary()} for i, c in enumerate(configs)]
        _, _, audit = bank.source_gp.rank(task["dataset"], task["algorithm"], kind, observations)
        verified.append({"task": key, "candidates": len(audit["candidates"]), "status": audit["status"]})
    campaign.save(output / "all_task_preflight.json", verified)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    root, output = args.campaign, args.output
    output.mkdir(parents=True, exist_ok=False)
    if args.preflight:
        preflight(root, output)
    import torch
    import run_toy_categorical_optimizers as runner
    from mindware.modules.base import BaseAutoML
    from peft import get_model_status

    task = next(t for t in campaign.load(root / "tasks_132.json") if t["dataset"] == "car" and t["algorithm"] == "extra_trees")
    argv, _, _ = campaign.command(root, "ckpt32", task, [0, 1, 2, 3], 2)
    argv = argv[4:]
    for flag, path in (("--output-dir", output / "outputs"), ("--history-dir", output / "history"),
                       ("--session-dir", output / "session")):
        argv[argv.index(flag) + 1] = str(path)
    argv.remove("--resume")
    original = BaseAutoML.run
    model_holder = []

    def short_run(model, *positional, **kwargs):
        if model.optimizer.evaluation_num_limit != 100:
            raise RuntimeError("Production selector budget changed")
        model.amount_of_resource = 9
        model_holder.append(model.optimizer)
        return original(model, *positional, **kwargs)

    with patch.object(sys, "argv", argv), patch.object(BaseAutoML, "run", short_run):
        try:
            runner.main()
        except RuntimeError as exc:
            progress = campaign.load(output / "session/progress.json", {})
            if not str(exc).endswith("stopped before reaching 100 trials") or progress.get("successful") != 9:
                raise
    progress = campaign.load(output / "session/progress.json")
    if progress["attempts"] != 9 or progress["successful"] != 9 or progress["failed"]:
        raise RuntimeError(f"Dry-run evaluation failures: {progress}")
    csv.field_size_limit(16 * 1024 * 1024)
    trajectory = Path(progress["trajectory"])
    with trajectory.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    opt = model_holder[0]
    params = campaign.load(trajectory.parent / "config.json")["optimizer_params"]
    expected = {"llm_duplicate_policy": "history_tpe5", "history_window": 24, "recent_k": 10,
                "top_k": 5, "worst_k": 5, "pucb_p1": 0.9, "pucb_p_end": 0.5,
                "pucb_q": "arm_best", "response_format": "json_only"}
    for key, value in expected.items():
        if params.get(key) != value:
            raise RuntimeError(f"Wrong setting {key}: {params.get(key)}")
    protocol = campaign.load(root / "protocol.json")
    adapter = params["adapter_verification"]
    if adapter["sha256"] != protocol["adapter_sha256"] or adapter["verified_tensors"] != 496:
        raise RuntimeError("Wrong ckpt32 adapter")
    if params["source_gp_manifest_sha256"] != protocol["source_gp_manifest_sha256"]:
        raise RuntimeError("Wrong GP bank")
    if params["preset_warm_start_sha256"] != protocol["warm_start_sha256"]:
        raise RuntimeError("Wrong warm starts")
    state = get_model_status(opt.llm_client._model)
    if state.enabled is not True or not state.active_adapters:
        raise RuntimeError("Adapter is disabled")
    devices = sorted({str(p.device) for p in opt.llm_client._model.parameters()})
    if devices != ["cuda:0"]:
        raise RuntimeError(f"Model was offloaded or spread across GPUs: {devices}")
    calls, rs_delivered = 0, []
    for index, row in enumerate(rows):
        if int(row["iter_idx"]) != index or not math.isfinite(float(row["perf"])) or not json.loads(row["config_json"]):
            raise RuntimeError("Missing or invalid evaluation record")
        if index < 3 and row["stage"] != "llm_warm_start":
            raise RuntimeError("Warm-start policy changed")
        if index >= 3:
            m = index - 3
            if not math.isclose(float(row["omega_llm"]), 0.9 - 0.4*m/97):
                raise RuntimeError("Wrong LLM prior schedule")
            if not math.isclose(float(row["omega_tpe"]), 0.1 + 0.4*m/97):
                raise RuntimeError("Wrong TPE prior schedule")
        if row["selected_arm"] == "llm":
            conversation = campaign.load(trajectory.parent / "llm_conversations" / f"trial_{index:04d}.json")
            attempts = conversation["llm_attempts"]
            if not 1 <= len(attempts) <= 5:
                raise RuntimeError("Wrong LLM call budget")
            calls += len(attempts)
            for j, attempt in enumerate(attempts):
                json.loads(attempt["raw_response"])
                prompt = attempt["messages"][-1]["content"]
                if "Strictly do not return a configuration you have previously recommended" not in prompt:
                    raise RuntimeError("Wrong no-repeat prompt")
                if attempt["recent_context_limit"] != 10 + j or len(attempt["retry_history"]) != j:
                    raise RuntimeError("Wrong retry history growth")
                if j and not attempts[j-1]["duplicate"]:
                    raise RuntimeError("Unexpected retry of a unique proposal")
            if row["stage"] == "tpe" and not (len(attempts) == 5 and all(a["duplicate"] for a in attempts)):
                raise RuntimeError("Unexpected TPE handoff")
            if "Source-task advice:" in attempts[0]["messages"][-1]["content"]:
                rs_delivered.append(index + 1)
        audit = json.loads(row["experience_retrieval"])
        if any(o["trial"] >= index + 1 for o in audit.get("observations", [])):
            raise RuntimeError("Future observation in GP inputs")
    if not calls or not rs_delivered:
        raise RuntimeError("Live LLM or RS was not exercised")
    report = {"passed": True, "production_results": False, "actual_evaluations": 9,
              "successful": 9, "llm_calls": calls, "rs_delivered_trials": rs_delivered,
              "adapter_verification": adapter, "active_adapters": state.active_adapters,
              "source_gp_manifest_sha256": params["source_gp_manifest_sha256"],
              "settings": expected, "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
              "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
              "trajectory": str(trajectory)}
    campaign.save(output / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
