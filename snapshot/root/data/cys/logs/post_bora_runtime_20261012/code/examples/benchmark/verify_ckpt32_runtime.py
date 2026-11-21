#!/usr/bin/env python3
"""Run actual ckpt32 evaluations with a short outer loop and the production selector budget."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mindware.components.experience.retrieve import ExperienceBank
from mindware.components.experience.source_gp import attach_source_gp, configuration, digest, json_digest
from mindware.components.experience.surrogate_landscape import get_algo_cs
from mindware.components.optimizers.ml_checkpoint import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="car")
    parser.add_argument("--algorithm", default="extra_trees")
    parser.add_argument("--verify-existing", action="store_true",
                        help="Verify saved nine-evaluation results and reload the adapter without rerunning ML evaluations.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.verify_existing)
    campaign = args.campaign
    tasks = json.loads((campaign / "tasks_132.json").read_text())
    task = next(t for t in tasks if t["dataset"] == args.dataset and t["algorithm"] == args.algorithm)
    bank_path = campaign / "experience_bank.jsonl"
    bank = attach_source_gp(ExperienceBank.load(bank_path), bank_path, campaign / "source_gp_train80_v1")
    presets = json.loads((campaign / "warm_start_presets.json").read_text())
    verified = []
    for target in tasks:
        kind = "classification" if target["task_type"] == "cls" else "regression"
        bank.source_gp.preflight(target["algorithm"], kind)
        cs = get_algo_cs(target["algorithm"], kind)
        key = f"{target['dataset']}:{target['algorithm']}"
        configs = [configuration(cs, raw, target["algorithm"]) for raw in presets[key]]
        if len(configs) != 3:
            raise RuntimeError(f"Expected three valid warm-start configurations: {key}")
        # Synthetic utilities exercise prediction compatibility, never optimization performance.
        observations = [{"trial": i + 1, "score": float(i), "source": "preflight_synthetic",
                         "config": c.get_dictionary()} for i, c in enumerate(configs)]
        _, _, audit = bank.source_gp.rank(target["dataset"], target["algorithm"], kind, observations)
        verified.append({"task": key, "candidate_count": len(audit["candidates"]),
                         "synthetic_prediction_status": audit["status"]})
    atomic_json(args.output / "all_task_preflight.json", verified)

    import ml_fixed_campaign
    import run_toy_categorical_optimizers as runner
    from mindware.modules.base import BaseAutoML
    argv, _, _ = ml_fixed_campaign.command(campaign, "ckpt32", task, [36], 0)
    argv = argv[4:]
    for flag, path in (("--output-dir", args.output / "outputs"),
                       ("--history-dir", args.output / "history"),
                       ("--session-dir", args.output / "session")):
        argv[argv.index(flag) + 1] = str(path)
    argv.remove("--resume")
    original_run = BaseAutoML.run

    def short_run(model, *positional, **kwargs):
        if model.optimizer.evaluation_num_limit != 100:
            raise RuntimeError("Dry run must preserve the production 100-attempt selector budget")
        model.amount_of_resource = 9
        return original_run(model, *positional, **kwargs)

    if not args.verify_existing:
        with patch.object(sys, "argv", argv), patch.object(BaseAutoML, "run", short_run):
            try:
                runner.main()
            except RuntimeError as exc:
                saved = json.loads((args.output / "session/progress.json").read_text())
                if (not str(exc).endswith("stopped before reaching 100 trials") or
                        saved.get("attempts") != 9 or saved.get("successful") != 9 or saved.get("failed")):
                    raise
    progress = json.loads((args.output / "session/progress.json").read_text())
    if progress["attempts"] != 9 or progress["successful"] != 9 or progress["failed"]:
        raise RuntimeError(f"Dry-run evaluations did not all succeed: {progress}")
    trajectory = Path(progress["trajectory"])
    with trajectory.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [int(r["iter_idx"]) for r in rows] != list(range(9)):
        raise RuntimeError("Dry-run trajectory has incorrect evaluation indices")
    if [r["stage"] for r in rows[:8]] != ["llm_warm_start"] * 3 + ["llm", "tpe", "llm", "llm", "llm"]:
        raise RuntimeError("Unexpected early proposer schedule")
    run_dir = trajectory.parent
    config = json.loads((run_dir / "config.json").read_text())
    if (config["optimizer_params"].get("response_format") != "json_only" or
            config["optimizer_params"].get("accept_duplicate_configs", False)):
        raise RuntimeError("Dry run did not use JSON-only output with the existing duplicate retry policy")
    verification = config["optimizer_params"]["adapter_verification"]
    expected = "8affcfe5fe458e64ff99ada1091b1de21651e0a597de57e2fda5f539354a7e8b"
    if verification["sha256"] != expected or verification["verified_tensors"] != 496:
        raise RuntimeError("The dry run did not load the audited ckpt32 adapter")
    if (config["optimizer_params"]["preset_warm_start_sha256"] != digest(campaign / "warm_start_presets.json") or
            config["optimizer_params"]["source_gp_manifest_sha256"] != bank.source_gp.manifest_sha256):
        raise RuntimeError("Saved dry-run inputs differ from the release inputs")
    from hf_qwen_client import get_hf_qwen_client
    from peft import get_model_status
    protocol = json.loads((campaign / "protocol.json").read_text())
    client = get_hf_qwen_client(model_path=protocol["base_model"], adapter_path=protocol["adapter"],
                               temperature=0.2, max_new_tokens=768)
    client._ensure_loaded()
    if client.adapter_verification["sha256"] != verification["sha256"]:
        raise RuntimeError("Reloaded adapter differs from the one used for the dry run")
    model_status = get_model_status(client._model)
    if model_status.enabled is not True or not model_status.active_adapters:
        raise RuntimeError("ckpt32 adapter is not enabled")
    retrieval = []
    delivered_rs, model_proposals, duplicate_retries, random_replacements = [], [], [], []
    recorded_model_calls = 0
    for number, row in enumerate(rows, 1):
        saved = json.loads((run_dir / "retrieval" / f"trial_{number:04d}.json").read_text())
        if not math.isfinite(float(row["perf"])) or not json.loads(row["config_json"]):
            raise RuntimeError("Missing saved configuration or performance")
        if any(o["trial"] >= number for o in saved.get("observations", [])):
            raise RuntimeError("Retrieval audit includes a future observation")
        if number not in (4, 6, 7, 8) and saved["selected_ids"]:
            raise RuntimeError("Advice appears outside eligible LLM turns")
        if row["stage"] == "llm":
            conversation = json.loads((run_dir / "llm_conversations" / f"trial_{number - 1:04d}.json").read_text())
            attempts = conversation.get("llm_attempts", [])
            if len(attempts) not in (1, 2):
                raise RuntimeError("An LLM evaluation slot exceeded the existing one-retry policy")
            if len(attempts) == 2:
                if not attempts[0]["duplicate"] or "reprompt_user_prompt" not in conversation["llm_input"]:
                    raise RuntimeError("An extra model response was not a duplicate retry")
                duplicate_retries.append(number)
            recorded_model_calls += len(attempts)
            for attempt in attempts:
                json.loads(attempt["raw_response"])
            if conversation["llm_input"].get("duplicate_accepted"):
                raise RuntimeError("Adaptive duplicate acceptance was enabled")
            if not conversation.get("note", "").startswith("llm "):
                random_replacements.append(number)
            prompt = conversation["llm_input"]["user_prompt"]
            if "No rationale" not in prompt:
                raise RuntimeError("JSON-only instruction was absent from the prompt")
            if saved["selected_ids"]:
                if "Source-task advice:" not in prompt or "full search space" not in prompt:
                    raise RuntimeError("Selected RS advice did not reach the LLM prompt")
                delivered_rs.append(number)
            elif number >= 9 and "Source-task advice:" in prompt:
                raise RuntimeError("Source advice remained in a prompt after the RS window")
            if conversation.get("raw_response") and conversation.get("note", "").startswith("llm "):
                model_proposals.append(number)
        retrieval.append({"evaluation": number, "status": saved["status"], "selected_ids": saved["selected_ids"]})
    if not delivered_rs or not model_proposals:
        raise RuntimeError("Dry run did not exercise both RS prompt delivery and valid model proposals")
    report = {"passed": True, "purpose": "correctness dry run; excluded from production results",
              "production_selector_attempts": 100, "actual_evaluations": 9, "successful": 9,
              "adapter_verification": verification, "active_adapters": model_status.active_adapters,
              "all_target_preflights": len(verified), "source_gp_manifest_sha256": bank.source_gp.manifest_sha256,
              "warm_start_sha256": digest(campaign / "warm_start_presets.json"), "retrieval": retrieval,
              "rs_prompt_evaluations": delivered_rs, "valid_model_proposal_evaluations": model_proposals,
              "response_format": "json_only", "duplicate_policy": "one reprompt then random replacement",
              "recorded_model_calls": recorded_model_calls, "duplicate_retry_evaluations": duplicate_retries,
              "random_replacement_evaluations": random_replacements,
              "trajectory": str(trajectory)}
    atomic_json(args.output / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
