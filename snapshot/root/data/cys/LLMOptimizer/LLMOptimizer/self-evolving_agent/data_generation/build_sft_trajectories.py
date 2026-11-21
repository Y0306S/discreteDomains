from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (PACKAGE_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bo.spaces import ConfigSpaceAdapter, HyperparameterSpec
from bo.trajectory import build_trajectory_prefix


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


def build_trajectory_sft(trace_dir: Path, smoothed_dir: Path | None, output_path: Path) -> int:
    rows = []
    for path in sorted(trace_dir.rglob("bo_trace_*.json")):
        trace = _load_trace(path)
        adapter = _adapter_from_trace(trace)
        task_desc = f"Optimize {trace['benchmark']} on task {trace['task_id']} to minimize loss."

        smoothed_path = None
        if smoothed_dir:
            candidate = smoothed_dir / trace["benchmark"].lower() / str(trace["task_id"]) / "smoothed_trajectory.json"
            if candidate.exists():
                smoothed_path = candidate

        smoothed = None
        if smoothed_path:
            with open(smoothed_path, "r", encoding="utf-8") as f:
                smoothed = json.load(f)

        trials = trace.get("trials", [])
        for idx in range(1, len(trials)):
            if trials[idx].get("loss") is None or trials[idx - 1].get("loss") is None:
                continue
            prefix = build_trajectory_prefix(trials, idx - 1)
            next_trial = trials[idx]
            assistant_payload = {
                "next_config": next_trial["config"],
                "rationale": "",
                "change_rationales": [],
            }
            if smoothed and smoothed.get("trajectory"):
                match = next((item for item in smoothed["trajectory"] if item.get("step") == next_trial["step"]), None)
                if match:
                    assistant_payload["rationale"] = match.get("rationale", "")
                    assistant_payload["change_rationales"] = match.get("change_rationales", [])
            rows.append(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You propose the next HPO configuration from an optimization trajectory.",
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "task_description": task_desc,
                                    "config_space": adapter.prompt_space_lines(),
                                    "round_summaries": trace.get("round_summaries", []),
                                    "trajectory_prefix": prefix,
                                },
                                ensure_ascii=False,
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": json.dumps(assistant_payload, ensure_ascii=False),
                        },
                    ]
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build trajectory SFT data.")
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--smoothed-dir", default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = build_trajectory_sft(
        trace_dir=Path(args.trace_dir),
        smoothed_dir=Path(args.smoothed_dir) if args.smoothed_dir else None,
        output_path=Path(args.output),
    )
    print(f"Wrote {count} trajectory examples to {args.output}")


if __name__ == "__main__":
    main()
