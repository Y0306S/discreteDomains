"""Replay saved Qwen replies through both adapters and real ML fits, without LLM calls."""

import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mindware import CASH
from mindware.components.optimizers.llm_optimizer_utils import wrapper_evaluater
from mindware.components.optimizers.ml_checkpoint import atomic_json
from mindware.components.optimizers.strict_recommendation import (
    GPTConfigurationExtractor, RECOMMENDATION_POLICY, config_payload, decode_object,
)
from mindware.components.optimizers.transfer_hpo.protocol import allowed_task_ids
from mindware.components.optimizers.transfer_hpo.schema import build_hpo_cs, schema_fingerprint
from mindware.components.utils.constants import CLASSIFICATION
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase
from mindware.modules.configs import CONFIGS
import run_toy_categorical_optimizers as runner
from run_online_llm_live import benchmark_args


class ReplayRuntime:
    def __init__(self, record):
        self.record, self.observations = record, []

    def propose(self, prompt, step):
        return {"raw_text": self.record["raw_text"], "tokens": self.record["original_tokens"],
                "features": [1.0], "policy_version": 0, "gradient_step": False}

    def observe(self, features, reward):
        self.observations.append(reward)


def transcribe_fixture(**request):
    text = json.loads(request["messages"][1]["content"])["recommendation"]
    values = config_payload(decode_object(text))
    body = {"ambiguous": False, "assignments": [
        {"name": name, "value": value, "evidence": json.dumps(name) + ": " + json.dumps(value)}
        for name, value in values.items()]}
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
        content=json.dumps(body), refusal=None))], usage=None, model="offline-transcriber-fixture")


def recorded_invalid_reply(root, method, cs):
    source = sorted((root / method / "outputs").glob("*/*.decisions.json"))[-1]
    for record in json.loads(source.read_text()):
        try:
            values = config_payload(decode_object(record["raw_text"]))
        except (ValueError, KeyError):
            continue
        value = values.get("lightgbm:num_leaves")
        if value is not None and value < cs["lightgbm:num_leaves"].lower:
            assert record["schema_fingerprint"] == schema_fingerprint(cs)
            return source, record, values
    raise AssertionError("No recorded out-of-range Qwen configuration found for " + method)


def run_method(method, directory, replay_directory):
    cs = build_hpo_cs("cls", "lightgbm")
    source, original, values = recorded_invalid_reply(replay_directory, method, cs)
    args = benchmark_args(method, directory)
    _, train, _, dataset_path, profile = runner.load_real_data(args)
    create = Mock(side_effect=transcribe_fixture)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    extractor = GPTConfigurationExtractor(max_tokens=512, timeout=60, max_calls=1, client=client)
    runtime = ReplayRuntime(original) if method == "tosfit" else None
    params = {"llm_client": SimpleNamespace(complete=Mock(return_value=original["raw_text"])),
              "tosfit_runtime": runtime, "extractor_client": extractor,
              "dataset_profile": profile, "n_pending": 50}
    with patch.object(KnowledgeDatabase, "_instance", None), \
            patch.object(KnowledgeDatabase, "fetch_best_conf_knowledge",
                         side_effect=AssertionError("Dry run must not retrieve cross-task history")), \
            patch.object(CONFIGS, "ORIGINAL_DATA_DIR", str(directory / "history")), \
            patch.object(CONFIGS, "HISTORY_RUN_SCOPE", str(directory / "history" / "scope"), create=True), \
            patch("mindware.components.optimizers.online_llm_optimizer.wrapper_evaluater",
                  wraps=wrapper_evaluater) as evaluate:
        cash = CASH(include_algorithms=["lightgbm"], task_type=CLASSIFICATION, metric="bal_acc",
                    data_node=train, evaluation="holdout",
                    resampling_params={"test_size": args.valid_size, "shuffle": False},
                    optimizer=method, time_limit=None, amount_of_resource=1,
                    per_run_time_limit=120, output_dir=str(directory / "outputs"),
                    seed=args.seed, n_jobs=1, task_id="car_lightgbm_" + method + "_repair_dry",
                    topk=1, optimizer_params=params)
        best = cash.optimizer.run()
    opt = cash.optimizer
    record = opt.decision_records[0]
    name = "lightgbm:num_leaves"
    default = cs[name].default_value
    assert math.isfinite(best)
    assert evaluate.call_count == 1
    assert len(opt.traj_rows) == 1 and opt.traj_rows[0]["oracle_called"]
    assert record["accepted"] and record["route"] == "gpt_extraction"
    assert record["assignments"][name] == values[name]
    assert record["repairs"][name] == {
        "original": values[name], "replacement": default, "reason": "out_of_range"}
    assert record["config"][name] == default
    assert json.loads(opt.traj_rows[0]["config_json"]) == record["config"]
    assert create.call_count == 1
    transcribed = {item["name"]: item["value"] for item in record["extraction"]["assignments"]}
    assert transcribed == values
    for key, value in values.items():
        if key != name:
            assert record["config"][key] == value
    if runtime:
        assert record["original_tokens"] == original["original_tokens"]
        assert len(runtime.observations) == 1 and math.isfinite(runtime.observations[0])
    return {"method": method, "passed": True, "task": "cls/car/lightgbm",
            "dataset_path": str(dataset_path), "recorded_response_source": str(source),
            "ml_evaluator_calls": evaluate.call_count, "incumbent_refit_calls": 0,
            "score": best, "transcriber_fixture_calls": create.call_count,
            "qwen_inference_calls": 0, "api_calls": 0, "qwen_gradient_updates": 0,
            "original_tokens_preserved": True if runtime else None,
            "decision": record}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path,
                        default=Path("/root/reports/bora_tosfit_live_20260928"))
    args = parser.parse_args()
    if "cls/car/lightgbm" not in allowed_task_ids():
        raise RuntimeError("Dry-run task is outside train80")
    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError("Use a new output directory for the dry run")
    for key in ("ML_SESSION_DIR", "ML_RUN_ID", "ML_EVALUATION_ID"):
        os.environ.pop(key, None)
    report = {"recommendation_policy": RECOMMENDATION_POLICY,
              "mode": "offline Qwen response replay, fixture GPT transcription, real ML evaluations",
              "methods": []}
    for method in ("bora", "tosfit"):
        result = run_method(method, args.out / method, args.replay_dir)
        report["methods"].append(result)
        atomic_json(args.out / "report.json", report)
        print(json.dumps({key: result[key] for key in (
            "method", "passed", "ml_evaluator_calls", "score", "api_calls")}), flush=True)
    report["passed"] = all(result["passed"] for result in report["methods"])
    atomic_json(args.out / "report.json", report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
