#!/usr/bin/env python3
"""Filter the existing TaskSet split to families at least as large as MLP-VAE."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path(
    "/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data"
)
TASKSET_ROOT = Path("/root/data/cys/taskset_runs")
TASK_CONFIGS = Path("/root/data/cys/google-research/task_set/tasks/configs")
HISTORY_ROOT = Path("/root/data/logs/taskset_corpus_train_1000s_100t")
DEFAULT_OUTPUT = DATA_DIR / "taskset_major_families_split_80_20_v1.json"
CUTOFF_FAMILY = "mlp_vae_family"


def load_json(path: Path):
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def build_split(source: Path):
    # Use the task registry and existing fixed-task parser for family identity.
    sys.path.insert(0, str(TASKSET_ROOT / "scripts"))
    from build_sft_dataset import load_task_config_index, normalize_best_rows
    from task_parsers import resolve_task

    prior, source_sha = load_json(source)
    universe, universe_sha = load_json(TASKSET_ROOT / "best_optimizer_per_task.json")
    tasks = normalize_best_rows(universe)
    index = load_task_config_index(TASK_CONFIGS)
    families = {task: resolve_task(task, index)[0] for task in tasks}
    counts = Counter(families.values())
    if "unknown" in counts:
        raise ValueError("Task family resolution failed for one or more tasks")

    sides = {side: set(prior[side]) for side in ("train", "test")}
    for side, ids in sides.items():
        if len(ids) != len(prior[side]) or len(ids) != prior[f"n_{side}"]:
            raise ValueError(f"Duplicate task IDs or inconsistent {side} count")
    if sides["train"] & sides["test"]:
        raise ValueError("Source train and test overlap")
    if sides["train"] | sides["test"] != set(tasks):
        raise ValueError("Source split does not match the scoreable task universe")

    cutoff = counts[CUTOFF_FAMILY]
    eligible = {family for family, count in counts.items() if count >= cutoff}
    selected = {
        side: sorted(task for task in ids if families[task] in eligible)
        for side, ids in sides.items()
    }
    actual = (cutoff, len(eligible), len(selected["train"]), len(selected["test"]))
    if actual != (55, 12, 838, 210):
        raise ValueError(f"Inputs changed: expected (55, 12, 838, 210), got {actual}")

    excluded = sorted(task for task in tasks if families[task] not in eligible)
    inventory = []
    for family in sorted(counts, key=lambda name: (-counts[name], name)):
        inventory.append({
            "family": family,
            "n_tasks": counts[family],
            "included": family in eligible,
            "source_train": sum(families[task] == family for task in sides["train"]),
            "source_test": sum(families[task] == family for task in sides["test"]),
        })

    manifest = {
        "schema_version": 1,
        "split_id": "taskset_major_families_80_20_v1",
        "source": str(source.resolve()),
        "source_sha256": source_sha,
        "task_universe": str(TASKSET_ROOT / "best_optimizer_per_task.json"),
        "task_universe_sha256": universe_sha,
        "assignment": "preserve_existing_train_test_membership",
        "selection": {
            "reference_family": CUTOFF_FAMILY,
            "minimum_family_size": cutoff,
            "count_scope": "all_scoreable_tasks_before_train_test_filtering",
        },
        "seed": prior["seed"],
        "test_frac": prior["test_frac"],
        "n_train": len(selected["train"]),
        "n_test": len(selected["test"]),
        "n_excluded": len(excluded),
        "included_families": sorted(eligible),
        "excluded_families": sorted(set(counts) - eligible),
        "family_inventory": inventory,
        "train": selected["train"],
        "test": selected["test"],
        "excluded": excluded,
    }
    return manifest


def audit_histories(manifest: dict, history_root: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from mindware.domains.history import history_exists, load_openbox_history
    from mindware.domains.taskset_archive import archive_path, read_archive_member

    available, missing = [], []
    for task in manifest["train"]:
        path = history_root / task / "tpe" / "seed1" / "history.json"
        if not history_exists(path):
            missing.append(task)
            continue
        history = load_openbox_history(path)
        packed = archive_path(path)
        data = read_archive_member(packed) if packed.exists() else path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        meta = history.get("meta_info", {})
        if history.get("task_id") != task or meta.get("domain") != "taskset":
            raise ValueError(f"History task/domain mismatch: {path}")
        if meta.get("optimizer") != "tpe" or meta.get("seed") != 1:
            raise ValueError(f"History optimizer/seed mismatch: {path}")
        observations = history.get("observations")
        if not isinstance(observations, list):
            raise ValueError(f"Missing observation list: {path}")
        available.append({
            "task_id": task,
            "path": str(path.resolve()),
            "sha256": digest,
            "archive": str(packed) if packed.exists() else None,
            "n_observations": len(observations),
            "complete_100_trials": len(observations) == 100,
        })

    n_complete = sum(row["complete_100_trials"] for row in available)
    return {
        "split_id": manifest["split_id"],
        "split_integrity_passed": True,
        "checks": {
            "source_ids_unique": True,
            "source_matches_scoreable_universe": True,
            "train_test_disjoint": True,
            "existing_membership_preserved": True,
            "family_cutoff_applied": True,
            "expected_train_test_counts": True,
        },
        "history_root": str(history_root.resolve()),
        "history_audit_scope": "task identity, optimizer, seed, observation count, file digest",
        "n_train": manifest["n_train"],
        "n_test": manifest["n_test"],
        "n_available_train_histories": len(available),
        "n_complete_train_histories": n_complete,
        "n_missing_train_histories": len(missing),
        "n_train_observations": sum(row["n_observations"] for row in available),
        "training_history_coverage_complete": n_complete == manifest["n_train"],
        "missing_train_task_ids": missing,
        "incomplete_train_task_ids": [
            row["task_id"] for row in available if not row["complete_100_trials"]
        ],
        "train_histories": available,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DATA_DIR / "taskset_full_split_80_20.json")
    parser.add_argument("--history-root", type=Path, default=HISTORY_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    manifest = build_split(args.source)
    audit = audit_histories(manifest, args.history_root)
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    audit["split_path"] = str(args.output.resolve())
    audit["split_sha256"] = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    audit_path = args.output.with_suffix(".audit.json")
    outputs = {
        args.output: manifest_text,
        audit_path: json.dumps(audit, indent=2) + "\n",
    }
    if not args.dry_run:
        for path, content in outputs.items():
            if path.exists() and path.read_text(encoding="utf-8") != content:
                raise ValueError(f"Refusing to replace a different existing artifact: {path}")
        for path, content in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                with path.open("x", encoding="utf-8") as stream:
                    stream.write(content)

    print(json.dumps({
        "dry_run": args.dry_run,
        "split": str(args.output),
        "audit": str(audit_path),
        "n_train": manifest["n_train"],
        "n_test": manifest["n_test"],
        "n_excluded": manifest["n_excluded"],
        "n_complete_train_histories": audit["n_complete_train_histories"],
        "missing_train_task_ids": audit["missing_train_task_ids"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
