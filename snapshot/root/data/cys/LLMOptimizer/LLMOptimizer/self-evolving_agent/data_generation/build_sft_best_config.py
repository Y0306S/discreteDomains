import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (PACKAGE_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bo.spaces import ConfigSpaceAdapter, HyperparameterSpec


def _load_trace(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _task_prompt(trace: dict) -> str:
    lines = [
        f"Benchmark: {trace['benchmark']}",
        f"Task id: {trace['task_id']}",
        f"Metric: {trace.get('metric', 'loss')}",
        "You must recommend exactly one best configuration.",
    ]
    if trace.get("round_summaries"):
        lines.append("Observed rounds and summaries are available in the trace artifact.")
    lines.append("Return only the best configuration, not an explanation.")
    return "\n".join(lines)


def build_best_config_sft(trace_dir: Path, output_path: Path) -> int:
    rows = []
    for path in sorted(trace_dir.rglob("bo_trace_*.json")):
        trace = _load_trace(path)
        space = trace.get("config_space", [])
        specs = []
        for item in space:
            specs.append(
                HyperparameterSpec(
                    name=item["name"],
                    kind=item["kind"],
                    lower=item["lower"],
                    upper=item["upper"],
                    default=item["default"],
                    log=bool(item.get("log", False)),
                )
            )
        adapter = ConfigSpaceAdapter(specs)
        prompt = _task_prompt(trace)
        task_desc = f"Optimize {trace['benchmark']} on task {trace['task_id']} to minimize loss."
        rows.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert AutoML HPO assistant.",
                    },
                    {
                        "role": "user",
                        "content": f"{task_desc}\n\n{adapter.prompt_space_lines()}\n\n{prompt}",
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps({"best_config": trace["best_config"]}, ensure_ascii=False),
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
    parser = argparse.ArgumentParser(description="Build BOLT-style best-config SFT data.")
    parser.add_argument("--trace-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = build_best_config_sft(Path(args.trace_dir), Path(args.output))
    print(f"Wrote {count} best-config examples to {args.output}")


if __name__ == "__main__":
    main()
