#!/usr/bin/env python3
"""Prepare DeepWeeds manual files for TFDS deep_weeds@2.0.0 precache.

Source zip: shansoround/sun397 on HF (deepweeds_images.zip) or any local copy.
TFDS expects a flat image zip + labels.csv at file:// URLs (official nextcloud URL is dead).
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

DATA_DIR = Path("/root/data/cys/tensorflow_datasets")
DEFAULT_MANUAL_DIR = DATA_DIR / "downloads" / "manual" / "deep_weeds"
DEFAULT_STAGING = DEFAULT_MANUAL_DIR / "_staging" / "shansoround"

IMAGE_ZIP_NAME = "deep_weeds_images.zip"
LABELS_NAME = "labels.csv"
MIN_IMAGE_ZIP_BYTES = 400_000_000
MIN_LABELS_BYTES = 500_000


def _find_source_zip(roots: list[Path]) -> Path | None:
    for root in roots:
        if not root.exists():
            continue
        direct = root / "deepweeds_images.zip"
        if direct.is_file() and direct.stat().st_size >= MIN_IMAGE_ZIP_BYTES:
            return direct
        for hit in root.rglob("deepweeds_images.zip"):
            if hit.is_file() and hit.stat().st_size >= MIN_IMAGE_ZIP_BYTES:
                return hit
    return None


def _extract_labels_csv(src_zip: Path, dest: Path) -> None:
    with zipfile.ZipFile(src_zip) as zf:
        for name in zf.namelist():
            norm = name.replace("\\", "/")
            if norm.endswith("labels/labels.csv") or norm.endswith("labels.csv"):
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, dest.open("wb") as out:
                    shutil.copyfileobj(src, out)
                return
    raise FileNotFoundError(f"labels.csv not found inside {src_zip}")


def _repack_flat_image_zip(src_zip: Path, dest_zip: Path) -> None:
    """Repack Kaggle layout (images/*.jpg) into a flat zip TFDS can listdir."""
    if dest_zip.exists() and dest_zip.stat().st_size >= MIN_IMAGE_ZIP_BYTES:
        return
    tmp = dest_zip.with_suffix(".zip.part")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(src_zip) as src, zipfile.ZipFile(
        tmp, "w", compression=zipfile.ZIP_DEFLATED
    ) as out:
        for info in src.infolist():
            name = info.filename.replace("\\", "/")
            if not name.lower().endswith(".jpg"):
                continue
            arcname = Path(name).name
            with src.open(info) as f:
                out.writestr(arcname, f.read())
    tmp.replace(dest_zip)


def ensure_deep_weeds_files(
    manual_dir: Path | None = None,
    *,
    staging_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Return (image_zip, labels_csv) paths under manual_dir, building if needed."""
    manual_dir = manual_dir or DEFAULT_MANUAL_DIR
    staging_dir = staging_dir or DEFAULT_STAGING
    manual_dir.mkdir(parents=True, exist_ok=True)

    image_zip = manual_dir / IMAGE_ZIP_NAME
    labels_csv = manual_dir / LABELS_NAME

    if (
        image_zip.exists()
        and image_zip.stat().st_size >= MIN_IMAGE_ZIP_BYTES
        and labels_csv.exists()
        and labels_csv.stat().st_size >= MIN_LABELS_BYTES
    ):
        return image_zip, labels_csv

    src_zip = _find_source_zip([staging_dir, manual_dir / "_staging", manual_dir])
    if not src_zip:
        raise FileNotFoundError(
            f"No deepweeds_images.zip under {staging_dir}. "
            "Download shansoround/sun397 (dataset) via hfd.sh first."
        )

    print(f"  Preparing DeepWeeds manual files from {src_zip}")
    _repack_flat_image_zip(src_zip, image_zip)
    if not labels_csv.exists() or labels_csv.stat().st_size < MIN_LABELS_BYTES:
        _extract_labels_csv(src_zip, labels_csv)

    if image_zip.stat().st_size < MIN_IMAGE_ZIP_BYTES:
        raise RuntimeError(f"Incomplete {image_zip}")
    if labels_csv.stat().st_size < MIN_LABELS_BYTES:
        raise RuntimeError(f"Incomplete {labels_csv}")

    print(f"  DeepWeeds manual ready: {image_zip.name}, {labels_csv.name}")
    return image_zip, labels_csv
