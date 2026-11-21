from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (PACKAGE_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_trace(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _best_trial(trials: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [trial for trial in trials if trial.get("loss") is not None]
    if not valid:
        return None
    return min(valid, key=lambda item: item["loss"])


def _playbook_text(trace: dict[str, Any]) -> str:
    trials = trace.get("trials", [])
    best = _best_trial(trials)
    valid = [trial for trial in trials if trial.get("loss") is not None]
    if not best:
        return "No valid trials available."

    losses = [trial["loss"] for trial in valid]
    median_loss = statistics.median(losses) if losses else None
    top = sorted(valid, key=lambda t: t["loss"])[:5]
    bottom = sorted(valid, key=lambda t: t["loss"], reverse=True)[:5]
    round_summaries = trace.get("round_summaries", [])

    lines = [
        f"Benchmark: {trace['benchmark']}",
        f"Task id: {trace['task_id']}",
        f"Best loss: {best['loss']}",
        f"Best config: {json.dumps(best['config'], ensure_ascii=False)}",
        "",
        "What worked:",
    ]
    for item in top[:3]:
        lines.append(f"- Step {item['step']}: loss={item['loss']} config={json.dumps(item['config'], ensure_ascii=False)}")
    lines.extend(
        [
            "",
            "What failed or underperformed:",
        ]
    )
    for item in bottom[:3]:
        lines.append(f"- Step {item['step']}: loss={item['loss']} config={json.dumps(item['config'], ensure_ascii=False)}")

    lines.extend(
        [
            "",
            "Round guidance:",
        ]
    )
    for rs in round_summaries:
        lines.append(
            f"- Round {rs['round']}: recommend {rs['exploration_recommendation']} because {rs['recommendation_rationale']}"
        )
    lines.extend(
        [
            "",
            "Generalized lessons:",
        ]
    )
    for note in trace.get("round_summaries", [])[:3]:
        for s in note.get("hyperparameter_sensitivity_notes", [])[:3]:
            lines.append(f"- {s}")
    if median_loss is not None:
        lines.append(f"- Median loss across valid trials: {median_loss}")
    return "\n".join(lines)


def build_playbooks(trace_dir: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(trace_dir.rglob("bo_trace_*.json")):
        trace = _load_trace(path)
        task_dir = output_dir / trace["benchmark"].lower() / str(trace["task_id"])
        task_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "benchmark": trace["benchmark"],
            "task_id": trace["task_id"],
            "best_config": trace.get("best_config"),
            "best_loss": trace.get("best_loss"),
            "round_summaries": trace.get("round_summaries", []),
            "playbook": _playbook_text(trace),
        }
        with open(task_dir / "playbook.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        with open(task_dir / "playbook.md", "w", encoding="utf-8") as f:
            f.write(payload["playbook"])
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cross-task playbooks.")
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = build_playbooks(Path(args.trace_dir), Path(args.output_dir))
    print(f"Wrote {count} playbooks to {args.output_dir}")


if __name__ == "__main__":
    main()
