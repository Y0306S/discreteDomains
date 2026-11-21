"""Keep exactly two on-disk checkpoints: latest + one previous."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List


def prev_checkpoint_path(path: Path) -> Path:
    path = Path(path)
    return path.with_name(f"{path.stem}.prev{path.suffix}")


def commit_two_deep_checkpoint(tmp: Path, latest: Path) -> Path:
    """Atomically promote tmp -> latest, rotating the old latest to .prev (drop older)."""
    latest = Path(latest)
    tmp = Path(tmp)
    prev = prev_checkpoint_path(latest)
    if prev.is_file():
        prev.unlink(missing_ok=True)
    if latest.is_file():
        latest.replace(prev)
    os.replace(tmp, latest)
    for stray_name in (f"{latest.name}.bak", f"{latest.name}.old", f"{latest.stem}.prev.prev{latest.suffix}"):
        stray = latest.with_name(stray_name)
        if stray.is_file():
            stray.unlink(missing_ok=True)
    return latest


def checkpoint_candidates(latest: Path) -> List[Path]:
    latest = Path(latest)
    out: List[Path] = []
    for cand in (latest, prev_checkpoint_path(latest)):
        if cand.is_file():
            out.append(cand)
    return out


def copy_two_deep(src_latest: Path, dst_latest: Path) -> None:
    """Copy latest + prev sibling if present."""
    src_latest = Path(src_latest)
    dst_latest = Path(dst_latest)
    dst_latest.parent.mkdir(parents=True, exist_ok=True)
    for src in checkpoint_candidates(src_latest):
        dst = dst_latest if src == src_latest else prev_checkpoint_path(dst_latest)
        if src.resolve() != dst.resolve():
            dst.write_bytes(src.read_bytes())
