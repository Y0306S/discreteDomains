#!/usr/bin/env python3
"""Audit TFDS dataset readiness for dl_test_15_modified.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "scripts" / "eval"))

from precache_datasets_v2 import (  # noqa: E402
    DATA_DIR,
    DATASET_ALIASES,
    DEFAULT_TEST_DATA,
    NO_TFDS_FAMILIES,
    extract_dataset_parts,
    is_build_complete,
    load_jsonl,
    normalize_dataset_name,
)

TOTAL_TASKS = 143
SYNTHETIC_TASKS = 44


def discover_dataset_tasks(test_path: Path) -> tuple[Counter[str], Counter[str]]:
    """Return (dataset -> task count, task_family -> task count)."""
    ds_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    for record in load_jsonl(test_path):
        user = json.loads(record["messages"][1]["content"])
        family = user.get("task_family", "unknown")
        family_counts[family] += 1
        if family in NO_TFDS_FAMILIES:
            continue
        ds = user.get("task", {}).get("dataset", [None])[0]
        if ds:
            ds_counts[normalize_dataset_name(ds)] += 1
    return ds_counts, family_counts


def has_incomplete_dir(builder_name: str, config_name: str | None) -> bool:
    builder_dir = DATA_DIR / builder_name
    if not builder_dir.exists():
        return False
    search_root = builder_dir / config_name if config_name else builder_dir
    if not search_root.exists():
        return False
    return any(".incomplete" in entry.name for entry in search_root.iterdir() if entry.is_dir())


def audit_status(dataset_name: str) -> tuple[str, Path | None]:
    builder_name, config_name = extract_dataset_parts(dataset_name)
    if is_build_complete(builder_name, config_name):
        search_root = DATA_DIR / builder_name
        if config_name:
            search_root = search_root / config_name
        for entry in search_root.iterdir():
            if entry.is_dir() and ".incomplete" not in entry.name:
                if (entry / "dataset_info.json").exists():
                    return "READY", entry
        return "READY", search_root
    if has_incomplete_dir(builder_name, config_name):
        return "INCOMPLETE", DATA_DIR / builder_name
    return "MISSING", None


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dl_test_15 TFDS dataset readiness")
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA)
    args = parser.parse_args()

    if not args.test_data.exists():
        print(f"ERROR: {args.test_data} not found")
        sys.exit(1)

    ds_counts, family_counts = discover_dataset_tasks(args.test_data)
    datasets = sorted(ds_counts.keys())

    print("=" * 70)
    print(f"Dataset audit: {args.test_data.name}")
    print(f"Cache dir: {DATA_DIR}")
    print("=" * 70)

    ready = 0
    missing = 0
    incomplete = 0
    tfds_tasks_ready = 0
    tfds_tasks_blocked = 0

    for ds in datasets:
        status, path = audit_status(ds)
        tasks = ds_counts[ds]
        if status == "READY":
            ready += 1
            tfds_tasks_ready += tasks
            loc = path if path else "?"
            print(f"  [READY     ] {ds:50} {tasks:3} tasks  -> {loc}")
        elif status == "INCOMPLETE":
            incomplete += 1
            tfds_tasks_blocked += tasks
            print(f"  [INCOMPLETE] {ds:50} {tasks:3} tasks")
        else:
            missing += 1
            tfds_tasks_blocked += tasks
            print(f"  [MISSING   ] {ds:50} {tasks:3} tasks")

    synthetic = sum(
        c for fam, c in family_counts.items() if fam in NO_TFDS_FAMILIES
    )
    evaluable = synthetic + tfds_tasks_ready

    print("=" * 70)
    print(f"Datasets: {ready}/{len(datasets)} READY, {missing} MISSING, {incomplete} INCOMPLETE")
    print(f"TFDS tasks: {tfds_tasks_ready}/{sum(ds_counts.values())} ready")
    print(f"Total evaluable tasks: {evaluable}/{TOTAL_TASKS} ({synthetic} synthetic + {tfds_tasks_ready} TFDS)")
    print("=" * 70)

    if missing or incomplete:
        sys.exit(1)


if __name__ == "__main__":
    main()
