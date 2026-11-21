"""Portable provenance checks shared by isolated transfer workers."""

import hashlib
import json
import math
import os
from pathlib import Path

TRAIN_MANIFEST = Path(
    os.environ.get("ML_TRANSFER_TRAIN_MANIFEST",
                   "/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/ml_train_80_fixed.jsonl")
)
HISTORY_ROOT = Path("/root/data/cys/blockdata_archive/llmoptimizer_benchmark/history_data/history_data")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path):
    path = Path(path)
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError("Empty or missing artifact directory: %s" % path)
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(path)).encode())
        digest.update(sha256_file(file).encode())
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".%d.tmp" % os.getpid())
    try:
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def acquisition_diagnostics(name, scores):
    values = [float(value) for value in scores]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Acquisition scores must be finite and nonempty")
    return {"name": name, "values": values, "candidate_count": len(values),
            "min": min(values), "max": max(values), "zero_count": values.count(0.0),
            "distinct_count": len(set(values)), "all_zero": all(value == 0 for value in values)}


def allowed_task_ids(manifest=TRAIN_MANIFEST):
    tasks = set()
    with open(manifest) as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                row = row.get("metadata") or row
                tasks.add("%s/%s/%s" % (row["task_type"], row["dataset"], row["algorithm"]))
    return tasks


def checked_source_path(path, root=HISTORY_ROOT):
    path, root = Path(path), Path(root)
    mirror = os.environ.get("ML_TRANSFER_HISTORY_MIRROR")
    if mirror:
        # Preserve the original provenance paths while validating a portable copy.
        try:
            relative = path.relative_to(root)
        except ValueError:
            raise ValueError("Source path is outside the allowed history root: %s" % path)
        root = Path(mirror)
        path = root / relative
    path, root = path.resolve(), root.resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError("Source path is outside the allowed history root: %s" % path)
    if not path.is_file():
        raise ValueError("Missing source history: %s" % path)
    return path


def validate_dump(dump, allowed=None, history_root=HISTORY_ROOT):
    allowed = allowed_task_ids() if allowed is None else set(allowed)
    if dump.get("format_version") != 2 or not dump.get("schema_fingerprint"):
        raise ValueError("Re-export transfer sources with format_version=2")
    tasks = dump.get("tasks") or {}
    if not tasks:
        raise ValueError("No eligible source tasks")
    for task_id, record in tasks.items():
        expected = "%s/%s/%s" % (dump["task_type"], record["dataset"], dump["algorithm"])
        if task_id != expected or task_id not in allowed:
            raise ValueError("Source task is not an allowed train80 task: %s" % task_id)
        path = checked_source_path(record["source_path"], history_root)
        if sha256_file(path) != record.get("source_sha256"):
            raise ValueError("Source history changed: %s" % path)
        n = len(record["configs"])
        if n == 0 or any(len(record[key]) != n for key in ("X", "y_minimize", "y_maximize")):
            raise ValueError("Source configurations and labels do not align: %s" % task_id)
        if any(not math.isfinite(float(y)) for y in record["y_minimize"]):
            raise ValueError("Non-finite source objective: %s" % task_id)
        if any(float(a) != -float(b) for a, b in zip(record["y_minimize"], record["y_maximize"])):
            raise ValueError("Inconsistent objective direction: %s" % task_id)
    return sorted(tasks)


def partition_tasks(task_ids, validation_file):
    """Partition by explicit IDs; never select validation tasks from their names."""
    requested = json.loads(Path(validation_file).read_text())
    if not isinstance(requested, list) or not all(isinstance(t, str) for t in requested):
        raise ValueError("Validation file must be a JSON list of train80 task IDs")
    if len(requested) != len(set(requested)):
        raise ValueError("Duplicate validation task IDs")
    if not set(requested) <= allowed_task_ids():
        raise ValueError("Validation tasks must be inside train80")
    validation = sorted(set(task_ids) & set(requested))
    train = sorted(set(task_ids) - set(requested))
    if not train or not validation:
        raise ValueError("Both training and validation need eligible tasks in this group")
    return train, validation
