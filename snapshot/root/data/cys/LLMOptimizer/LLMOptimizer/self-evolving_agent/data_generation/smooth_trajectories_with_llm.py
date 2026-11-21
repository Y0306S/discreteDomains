from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (PACKAGE_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bo.spaces import ConfigSpaceAdapter, HyperparameterSpec
from llm.client import LLMClient, LLMClientConfig
from llm.prompts import trajectory_smoothing_prompt


def _load_trace(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _adapter_from_trace(trace: dict[str, Any]) -> ConfigSpaceAdapter:
    specs = [
        HyperparameterSpec(
            name=item["name"],
            kind=item["kind"],
            lower=item["lower"],
            upper=item["upper"],
            default=item["default"],
            log=bool(item.get("log", False)),
        )
        for item in trace.get("config_space", [])
    ]
    return ConfigSpaceAdapter(specs)


def _validate_smoothed(trace: dict[str, Any], smoothed: dict[str, Any]) -> dict[str, Any]:
    valid_steps = {trial["step"]: trial for trial in trace.get("trials", [])}
    cleaned = []
    for item in smoothed.get("trajectory", []):
        step = item.get("step")
        if step not in valid_steps:
            continue
        original = valid_steps[step]
        if item.get("config") != original.get("config"):
            continue
        try:
            same_loss = abs(float(item.get("loss")) - float(original.get("loss"))) <= 1e-12
        except Exception:
            same_loss = False
        if not same_loss:
            continue
        cleaned.append(
            {
                "step": step,
                "config": original["config"],
                "loss": original["loss"],
                "rationale": item.get("rationale", ""),
                "change_rationales": item.get("change_rationales", []),
            }
        )
    return {
        "task_id": trace["task_id"],
        "benchmark": trace["benchmark"],
        "metric": trace.get("metric", "loss"),
        "trajectory": cleaned,
        "round_summaries": trace.get("round_summaries", []),
    }


def smooth_trace(
    trace: dict[str, Any],
    model: str,
    base_url: str,
    api_key_env: str,
    offline: bool = False,
) -> dict[str, Any]:
    adapter = _adapter_from_trace(trace)
    prefix = [
        {
            "step": trial["step"],
            "config": trial["config"],
            "loss": trial["loss"],
            "cost": trial.get("cost"),
            "source": trial.get("source", "bo"),
        }
        for trial in trace.get("trials", [])
        if trial.get("loss") is not None
    ]

    if offline:
        ordered = sorted(prefix, key=lambda x: x["loss"], reverse=True)
        trajectory = []
        previous = None
        for item in ordered:
            change_rationales = []
            if previous is not None:
                changed = {
                    k: f"{previous['config'][k]} -> {item['config'][k]}"
                    for k in item["config"].keys()
                    if previous["config"].get(k) != item["config"].get(k)
                }
                change_rationales.append(
                    {
                        "from_step": previous["step"],
                        "to_step": item["step"],
                        "changed": changed,
                        "reason": "Offline smoothing kept the real BO transition and annotated the parameter movement toward lower loss.",
                    }
                )
            trajectory.append(
                {
                    "step": item["step"],
                    "config": item["config"],
                    "loss": item["loss"],
                    "rationale": "Offline smoothing retained the evaluated point and narrated how it relates to the optimization path.",
                    "change_rationales": change_rationales,
                }
            )
            previous = item
        return {
            "task_id": trace["task_id"],
            "benchmark": trace["benchmark"],
            "metric": trace.get("metric", "loss"),
            "trajectory": trajectory,
            "round_summaries": trace.get("round_summaries", []),
        }

    client = LLMClient(LLMClientConfig(model=model, base_url=base_url, api_key=os.getenv(api_key_env)))
    messages = trajectory_smoothing_prompt(
        task_description=f"Optimize {trace['benchmark']} on task {trace['task_id']} to minimize loss.",
        config_space_text=adapter.prompt_space_lines(),
        round_summaries=trace.get("round_summaries", []),
        trajectory_prefix=prefix,
    )
    response = client.chat_json(messages, temperature=0.4)
    raw = response.choices[0].message.content or "{}"
    try:
        payload = json.loads(raw)
    except Exception:
        payload = {"trajectory": []}
    return _validate_smoothed(trace, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smooth BO trajectories with an LLM.")
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--base-url", default="https://api.openai-hk.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_dir = Path(args.trace_dir)
    output_dir = Path(args.output_dir)
    for path in sorted(trace_dir.rglob("bo_trace_*.json")):
        trace = _load_trace(path)
        smoothed = smooth_trace(
            trace=trace,
            model=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            offline=args.offline,
        )
        out_path = output_dir / trace["benchmark"].lower() / str(trace["task_id"]) / "smoothed_trajectory.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(smoothed, f, indent=2, ensure_ascii=False)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
