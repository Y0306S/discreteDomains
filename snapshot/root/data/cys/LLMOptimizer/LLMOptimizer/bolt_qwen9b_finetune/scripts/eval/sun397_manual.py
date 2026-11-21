#!/usr/bin/env python3
"""Resolve SUN397 manual files for TFDS sun397/tfds@4.0.0 precache."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

DATA_DIR = Path("/root/data/cys/tensorflow_datasets")
DEFAULT_MANUAL_DIR = DATA_DIR / "downloads" / "manual" / "sun397"

REQUIRED_FILES = ("SUN397.tar.gz", "Partitions.zip")
MIN_BYTES = {
    "SUN397.tar.gz": 30_000_000_000,
    "Partitions.zip": 1_000_000,
}


def _file_ready(path: Path) -> bool:
    if not path.is_file():
        return False
    return path.stat().st_size >= MIN_BYTES.get(path.name, 1_000_000)


def _find_file(name: str, roots: list[Path]) -> Path | None:
    for root in roots:
        if not root.exists():
            continue
        direct = root / name
        if direct.is_file():
            return direct
        for hit in root.rglob(name):
            if hit.is_file():
                return hit
    return None


def _install(src: Path, dest: Path) -> None:
    if src.resolve() == dest.resolve():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if dest.stat().st_size >= src.stat().st_size:
            return
        dest.unlink()
    try:
        os.link(src, dest)
        print(f"  linked {dest.name} from {src.parent}")
    except OSError:
        shutil.copy2(src, dest)
        print(f"  copied {dest.name} from {src} ({dest.stat().st_size / 1e9:.2f} GB)")


def promote_staging_files(manual_dir: Path | None = None) -> None:
    """Move/link complete downloads from _staging into manual_dir."""
    manual_dir = manual_dir or DEFAULT_MANUAL_DIR
    staging_roots = [
        manual_dir / "_staging" / "hfd",
        manual_dir / "_staging" / "hf_xet",
        manual_dir / "_staging" / "hf_mirror",
        manual_dir / "_staging" / "hub_mirror",
        manual_dir / "_staging",
    ]
    for name in REQUIRED_FILES:
        dest = manual_dir / name
        if _file_ready(dest):
            continue
        src = _find_file(name, staging_roots)
        if src and _file_ready(src):
            _install(src, dest)


def sun397_manual_ready(manual_dir: Path | None = None) -> bool:
    manual_dir = manual_dir or DEFAULT_MANUAL_DIR
    return all(_file_ready(manual_dir / name) for name in REQUIRED_FILES)


def ensure_sun397_manual_files(
    manual_dir: Path | None = None,
    *,
    wait: bool = False,
    poll_sec: int = 120,
    max_wait_sec: int = 6 * 3600,
) -> Path:
    """Promote staging files and optionally wait until SUN397.tar.gz is complete."""
    manual_dir = manual_dir or DEFAULT_MANUAL_DIR
    manual_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + max_wait_sec if wait else time.time()

    while True:
        promote_staging_files(manual_dir)
        tar = manual_dir / "SUN397.tar.gz"
        parts = manual_dir / "Partitions.zip"
        if _file_ready(tar) and _file_ready(parts):
            print(f"  SUN397 manual ready under {manual_dir}")
            return manual_dir
        if not wait or time.time() >= deadline:
            tar_gb = tar.stat().st_size / 1e9 if tar.exists() else 0.0
            raise FileNotFoundError(
                f"SUN397 manual files incomplete under {manual_dir} "
                f"(tar {tar_gb:.1f} GB / ~39 GB). "
                "Let download_sun397_manual.py finish or re-run it."
            )
        tar_gb = tar.stat().st_size / 1e9 if tar.exists() else 0.0
        print(f"  Waiting for SUN397.tar.gz ({tar_gb:.1f} GB / ~39 GB) ...")
        time.sleep(poll_sec)
