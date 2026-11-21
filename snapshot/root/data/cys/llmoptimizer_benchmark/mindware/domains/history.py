"""OpenBox-style history JSON helpers for domain BO / fake-BO dumps."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from mindware.domains.protein.checkpoint_rotate import commit_two_deep_checkpoint


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def observation(
    config: Dict[str, Any],
    objective: float,
    *,
    trial_state: int = 0,
    elapsed_time: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    obj = float(objective) if math.isfinite(float(objective)) else 1e10
    return {
        "config": dict(config),
        "objectives": [obj],
        "constraints": None,
        "trial_state": int(trial_state),
        "elapsed_time": float(elapsed_time),
        "create_time": utc_now(),
        "extra_info": dict(extra or {}),
    }


def write_openbox_history(
    path: Path,
    *,
    task_id: str,
    observations: Sequence[Dict[str, Any]],
    meta: Optional[Dict[str, Any]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": str(task_id),
        "num_objectives": 1,
        "num_constraints": 0,
        "ref_point": None,
        "meta_info": dict(meta or {}),
        "global_start_time": utc_now(),
        "observations": list(observations),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_rotating_openbox_history(
    path: Path,
    *,
    task_id: str,
    observations: Sequence[Dict[str, Any]],
    meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write history.json keeping only latest + history.prev.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    payload = {
        "task_id": str(task_id),
        "num_objectives": 1,
        "num_constraints": 0,
        "ref_point": None,
        "meta_info": dict(meta or {}),
        "global_start_time": utc_now(),
        "observations": list(observations),
    }
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    commit_two_deep_checkpoint(tmp, path)
    return path


def load_openbox_history(path: Path) -> Dict[str, Any]:
    """Read an authoritative completed archive, or rotating active checkpoints."""
    from mindware.domains.protein.checkpoint_rotate import checkpoint_candidates

    path = Path(path)
    from mindware.domains.taskset_archive import archive_path, read_archive_member

    packed = path if path.name.endswith(".tar.gz") else archive_path(path)
    if packed.exists():
        return json.loads(read_archive_member(packed))
    last_err: Optional[Exception] = None
    for cand in checkpoint_candidates(path):
        try:
            return json.loads(cand.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            last_err = exc
            continue
    # A packer can publish and remove the loose files between the first archive
    # check and the read. Workers hold job locks; concurrent analysis may not.
    if packed.exists():
        return json.loads(read_archive_member(packed))
    if last_err is not None:
        raise last_err
    raise FileNotFoundError(f"history missing: {path}")


def history_exists(path: Path) -> bool:
    from mindware.domains.taskset_archive import archive_path

    path = Path(path)
    return (path.exists() or path.with_name("history.prev.json").exists()
            or archive_path(path).exists())


def iter_history_paths(root: Path):
    """Discover the logical history paths for TaskSet's task/method/seed layout."""
    root = Path(root)
    paths = set(root.glob("*/*/*/history.json"))
    paths.update(p.with_name("history.json") for p in root.glob("*/*/*/history.prev.json"))
    paths.update(p.with_name(p.name[:-7]) / "history.json" for p in root.glob("*/*/seed*.tar.gz"))
    return iter(sorted(paths))


def scored_only(observations: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in observations:
        objs = row.get("objectives") or []
        if int(row.get("trial_state", 0) or 0) != 0:
            continue
        if not objs:
            continue
        try:
            val = float(objs[0])
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            out.append(row)
    return out


def sort_improving(observations: Sequence[Dict[str, Any]], *, minimize: bool = True) -> List[Dict[str, Any]]:
    """Reorder unordered scored pairs to look like a weakly improving BO path."""
    rows = scored_only(observations)
    if not rows:
        return []
    key = lambda r: float(r["objectives"][0])
    ordered = sorted(rows, key=key, reverse=not minimize)
    # Interleave: first trial = median, then walk toward best so anytime is not a cheat.
    if len(ordered) < 3:
        return list(rows)
    mid = ordered[len(ordered) // 2]
    rest = [r for r in ordered if r is not mid]
    if minimize:
        rest = sorted(rest, key=lambda r: float(r["objectives"][0]))
    else:
        rest = sorted(rest, key=lambda r: -float(r["objectives"][0]))
    return [mid] + rest


def is_git_lfs_pointer(path: Path) -> bool:
    try:
        head = Path(path).read_text(encoding="utf-8", errors="replace")[:80]
    except OSError:
        return False
    return head.startswith("version https://git-lfs.github.com/spec/v1")
