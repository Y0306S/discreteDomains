#!/usr/bin/env python3
"""Download TFDS Wikipedia parents from GCS with correct version scoping.

TFDS gcs_utils flattens all config versions by basename, so 0.0.3 shards can
overwrite 0.0.4. We download only datasets/wikipedia/<config>/<version>/ and
skip diffs/. Supports fetching individual missing shards when a partial cache exists.
"""

from __future__ import annotations

import concurrent.futures
import os
import posixpath
import re
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import requests
import tensorflow as tf

GCS_BUCKET = "http://storage.googleapis.com/tfds-data"
GCS_DATASETS_DIR = "datasets"
DEFAULT_SHARDS = 126


def _shard_count_from_names(names: list[str]) -> int | None:
    for name in names:
        match = re.search(r"-of-(\d+)", posixpath.basename(name))
        if match:
            return int(match.group(1))
    return None


def expected_shard_count(parent_dir: str, gcs_paths: list[str] | None = None) -> int:
    """Infer shard count from local files or GCS listing (en=126, ru=29, etc.)."""
    if tf.io.gfile.exists(parent_dir):
        local = tf.io.gfile.glob(posixpath.join(parent_dir, "wikipedia-train.tfrecord-*"))
        count = _shard_count_from_names(local)
        if count:
            return count
    if gcs_paths:
        count = _shard_count_from_names(gcs_paths)
        if count:
            return count
    return DEFAULT_SHARDS


def _gcs_list(prefix: str) -> list[str]:
    url = f"{GCS_BUCKET}/?prefix={prefix}"
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    root = ElementTree.fromstring(resp.content)
    return [el[0].text for el in root if el.tag.endswith("Contents")]


def gcs_paths_for_version(dataset_name: str, version: str) -> list[str]:
    """GCS object keys for one Wikipedia parent version (no diffs/)."""
    prefix = posixpath.join(GCS_DATASETS_DIR, dataset_name, version, "")
    paths = _gcs_list(prefix)
    return [p for p in paths if "/diffs/" not in p]


def parent_ready(parent_dir, *, expected_shards: int = DEFAULT_SHARDS) -> bool:
    parent_dir = str(parent_dir)
    info = posixpath.join(parent_dir, "dataset_info.json")
    if not tf.io.gfile.exists(info):
        return False
    text = tf.io.gfile.GFile(info).read().lstrip()
    if not text.startswith("{"):
        return False
    for i in range(expected_shards):
        shard = posixpath.join(
            parent_dir, f"wikipedia-train.tfrecord-{i:05d}-of-{expected_shards:05d}"
        )
        if not tf.io.gfile.exists(shard):
            return False
        if tf.io.gfile.stat(shard).length == 0:
            return False
    return True


def _is_shard_object(gcs_path: str) -> bool:
    return "tfrecord" in posixpath.basename(gcs_path)


def _download_gcs_object(gcs_path: str, local_path: str) -> None:
    """Download one GCS object via HTTP; use stdlib I/O (tf GFile races on metadata)."""
    url = f"{GCS_BUCKET}/{gcs_path}"
    local = Path(local_path)
    local.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()
    tmp_path = local.with_name(local.name + ".partial")
    try:
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
        os.replace(tmp_path, local)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def ensure_parent_from_gcs(
    dataset_name: str,
    version: str,
    parent_dir,
    *,
    max_workers: int = 16,
) -> None:
    """Download missing GCS files for one Wikipedia parent version."""
    parent_dir = str(parent_dir)
    tf.io.gfile.makedirs(parent_dir)
    gcs_paths = gcs_paths_for_version(dataset_name, version)
    if not gcs_paths:
        raise RuntimeError(f"No GCS files for {dataset_name}/{version}")
    shard_count = expected_shard_count(parent_dir, gcs_paths)

    todo_meta: list[tuple[str, str]] = []
    todo_shards: list[tuple[str, str]] = []
    for gcs_path in gcs_paths:
        local_path = posixpath.join(parent_dir, posixpath.basename(gcs_path))
        if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
            continue
        item = (gcs_path, local_path)
        if _is_shard_object(gcs_path):
            todo_shards.append(item)
        else:
            todo_meta.append(item)

    todo = todo_meta + todo_shards
    if not todo:
        print(f"  Wikipedia GCS: all {len(gcs_paths)} files already present")
        return

    print(
        f"  Wikipedia GCS: downloading {len(todo)}/{len(gcs_paths)} files "
        f"({len(todo_meta)} metadata, {len(todo_shards)} shards) "
        f"for {dataset_name}/{version} ..."
    )

    for gcs_path, local_path in todo_meta:
        _download_gcs_object(gcs_path, local_path)

    if todo_shards:
        def _copy(item: tuple[str, str]) -> None:
            gcs_path, local_path = item
            _download_gcs_object(gcs_path, local_path)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_copy, item) for item in todo_shards]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    if not parent_ready(parent_dir, expected_shards=shard_count):
        missing = []
        for i in range(shard_count):
            shard = posixpath.join(
                parent_dir,
                f"wikipedia-train.tfrecord-{i:05d}-of-{shard_count:05d}",
            )
            if not tf.io.gfile.exists(shard):
                missing.append(posixpath.basename(shard))
        raise RuntimeError(
            f"Wikipedia parent still incomplete after GCS download; "
            f"missing {len(missing)} shards (e.g. {missing[:3]})"
        )
