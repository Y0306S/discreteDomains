#!/usr/bin/env python3
"""Housekeep legacy shared history JSON files into per-variant directories.

Before the benchmark runner change, all optimizers on a dataset wrote to:
  history_data/{cls|rgs}/{metric}/{dataset}/random_forest.json

This script moves each flat file to:
  history_data/{cls|rgs}/{metric}/{dataset}/{variant}/random_forest.json

using the OpenBox History ``task_id`` field (e.g. yeast_ptllm__P2 -> ptllm__P2).

If the target variant path already exists, the flat file is archived under:
  .../{dataset}/_legacy_collapsed/{task_id}_random_forest.json

Optionally rewrites history_paths in benchmark summary.json files.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_ROOT = REPO / "history_data"
BENCHMARK_ROOTS = [
    REPO / "benchmark_data" / "real_hpo_v2",
    REPO / "benchmark_data" / "real_hpo_cls",
]


def variant_from_task_id(task_id: str, dataset: str) -> str | None:
    prefix = f"{dataset}_"
    if task_id.startswith(prefix):
        return task_id[len(prefix) :]
    return None


def is_flat_history_path(path: Path, history_root: Path) -> bool:
    """True if path is .../{task_type}/{metric}/{dataset}/random_forest.json."""
    try:
        rel = path.relative_to(history_root)
    except ValueError:
        return False
    parts = rel.parts
    if len(parts) != 4 or parts[-1] != "random_forest.json":
        return False
    task_type, metric, dataset, _ = parts
    if task_type not in ("cls", "rgs"):
        return False
    if dataset.startswith("_"):
        return False
    return True


def load_task_id(history_path: Path) -> str | None:
    try:
        with history_path.open(encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("task_id")
    except (OSError, json.JSONDecodeError):
        return None


def migrate_flat_file(history_path: Path, history_root: Path, dry_run: bool) -> dict:
    rel = history_path.relative_to(history_root)
    task_type, metric, dataset, _ = rel.parts
    task_id = load_task_id(history_path)
    if not task_id:
        return {"path": str(history_path), "action": "skip", "reason": "no task_id"}

    variant = variant_from_task_id(task_id, dataset)
    if not variant:
        return {"path": str(history_path), "action": "skip", "reason": f"unparsed task_id={task_id!r}"}

    target_dir = history_root / task_type / metric / dataset / variant
    target_path = target_dir / "random_forest.json"
    archive_dir = history_root / task_type / metric / dataset / "_legacy_collapsed"
    archive_path = archive_dir / f"{task_id}_random_forest.json"

    if target_path.resolve() == history_path.resolve():
        return {"path": str(history_path), "action": "skip", "reason": "already in variant dir"}

    action = "move"
    dest = target_path
    if target_path.exists():
        action = "archive"
        dest = archive_path

    record = {
        "path": str(history_path),
        "action": action,
        "task_id": task_id,
        "variant": variant,
        "dest": str(dest),
    }

    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(history_path), str(dest))
    return record


def new_history_path(old_path: str, history_root: Path) -> str | None:
    p = Path(old_path)
    if not p.is_absolute():
        p = (REPO / p).resolve()
    if not is_flat_history_path(p, history_root):
        return None
    task_id = load_task_id(p)
    if not task_id:
        return None
    rel = p.relative_to(history_root)
    task_type, metric, dataset, _ = rel.parts
    variant = variant_from_task_id(task_id, dataset)
    if not variant:
        return None
    return str(history_root / task_type / metric / dataset / variant / "random_forest.json")


def rewrite_json_paths(obj, history_root: Path, dry_run: bool) -> int:
    updated = 0
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in ("history_paths", "histories") and isinstance(value, dict):
                for algo, path in list(value.items()):
                    if not isinstance(path, str):
                        continue
                    new_path = new_history_path(path, history_root)
                    if new_path and new_path != path:
                        if not dry_run:
                            value[algo] = new_path
                        updated += 1
            else:
                updated += rewrite_json_paths(value, history_root, dry_run)
    elif isinstance(obj, list):
        for item in obj:
            updated += rewrite_json_paths(item, history_root, dry_run)
    return updated


def patch_summary_from_results(summary_path: Path, history_root: Path, dry_run: bool) -> int:
    """Rewrite flat history_paths using optimizer variant + dataset from summary."""
    with summary_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    dataset = payload.get("dataset")
    task_type = payload.get("task_type", "cls")
    metric = payload.get("metric", "bal_acc" if task_type == "cls" else "mse")
    if not dataset:
        return 0

    updated = 0
    for result in payload.get("results", []):
        optimizer = result.get("optimizer")
        if not optimizer:
            continue
        new_path = str(history_root / task_type / metric / dataset / optimizer / "random_forest.json")
        for key in ("histories",):
            bucket = result.get(key)
            if not isinstance(bucket, dict):
                continue
            for algo, old in list(bucket.items()):
                if isinstance(old, str) and old.endswith(f"/{dataset}/random_forest.json") and old != new_path:
                    if not dry_run:
                        bucket[algo] = new_path
                    updated += 1
        model_info = result.get("model_info")
        if isinstance(model_info, dict):
            hp = model_info.get("history_paths")
            if isinstance(hp, dict):
                for algo, old in list(hp.items()):
                    if isinstance(old, str) and old.endswith(f"/{dataset}/random_forest.json") and old != new_path:
                        if not dry_run:
                            hp[algo] = new_path
                        updated += 1

    if updated and not dry_run:
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return updated


def patch_best_model_info(info_path: Path, history_root: Path, dry_run: bool) -> int:
    parts = info_path.parts
    try:
        idx = parts.index("benchmark_data")
        # .../benchmark_data/real_hpo_v2/cls/yeast/ptllm__P0/CASH-.../best_model_info.json
        task_type = parts[idx + 2]
        dataset = parts[idx + 3]
        optimizer = parts[idx + 4]
    except (ValueError, IndexError):
        return 0
    metric = "bal_acc" if task_type == "cls" else "mse"
    new_path = str(history_root / task_type / metric / dataset / optimizer / "random_forest.json")

    with info_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    updated = 0
    hp = payload.get("history_paths")
    if isinstance(hp, dict):
        for algo, old in list(hp.items()):
            if isinstance(old, str) and old.endswith(f"/{dataset}/random_forest.json") and old != new_path:
                if not dry_run:
                    hp[algo] = new_path
                updated += 1
    if updated and not dry_run:
        with info_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return updated


def patch_benchmark_summaries(history_root: Path, dry_run: bool) -> int:
    total = 0
    for root in BENCHMARK_ROOTS:
        if not root.is_dir():
            continue
        for summary_path in root.glob("**/summary.json"):
            total += patch_summary_from_results(summary_path, history_root, dry_run)
        for info_path in root.glob("**/best_model_info.json"):
            total += patch_best_model_info(info_path, history_root, dry_run)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-summaries", action="store_true")
    args = parser.parse_args()

    history_root = args.history_root.resolve()
    flat_files = sorted(
        p
        for p in history_root.glob("**/random_forest.json")
        if is_flat_history_path(p, history_root)
    )

    print(f"History root: {history_root}")
    print(f"Flat files found: {len(flat_files)}")
    moves = archive = skip = 0
    for path in flat_files:
        rec = migrate_flat_file(path, history_root, args.dry_run)
        print(f"  [{rec['action']}] {rec['path']}")
        if rec["action"] == "move":
            moves += 1
            print(f"         -> {rec['dest']}")
        elif rec["action"] == "archive":
            archive += 1
            print(f"         -> {rec['dest']} (variant file existed)")
        else:
            skip += 1
            print(f"         reason: {rec.get('reason')}")

    print(f"\nSummary: move={moves} archive={archive} skip={skip}")

    if not args.skip_summaries:
        n = patch_benchmark_summaries(history_root, args.dry_run)
        mode = "would update" if args.dry_run else "updated"
        print(f"Benchmark metadata: {mode} {n} history path reference(s)")


if __name__ == "__main__":
    main()
