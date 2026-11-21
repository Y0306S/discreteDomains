#!/usr/bin/env python3
"""Pre-download and pre-build TFDS datasets needed for DL taskset evaluation.

- Discovers datasets from dl_test JSONL user messages (not only char_rnn configs).
- Relocates stray flat TFDS builds into the correct builder/version layout.
- Builds task_set custom datasets (resized images, tokenized text) with data_dir set.
- Runs sequentially to avoid TFDS lock contention.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
EVAL_SCRIPTS_DIR = Path(__file__).resolve().parent
GOOGLE_RESEARCH_ROOT = Path("/root/data/cys/blockdata_archive/LLMOptimizer/google-research")

sys.path.insert(0, str(GOOGLE_RESEARCH_ROOT))
sys.path.insert(0, str(EVAL_SCRIPTS_DIR))

import amazon_reviews_hf
import deep_weeds_hf
import parallelism
import sun397_manual

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import tensorflow_datasets as tfds
from task_set import datasets as task_set_datasets

DATA_DIR = Path("/root/data/cys/tensorflow_datasets")
DEFAULT_TEST_DATA = BASE_DIR / "data/dl_test_15_modified.jsonl"

NO_TFDS_FAMILIES = {"losg_tasks_family", "quadratic_family"}

# Fix typo in dl_test_15_modified.jsonl
DATASET_ALIASES = {
    "imdb_reviews/subwords8kimdb_reviews/bytes": "imdb_reviews/subwords8k",
}

RESIZED_BUILDERS = {
    "food101_32x32": task_set_datasets.Food101_32x32,
    "coil100_32x32": task_set_datasets.Coil100_32x32,
    "deep_weeds_32x32": task_set_datasets.DeepWeeds_32x32,
    "sun397_32x32": task_set_datasets.Sun397_32x32,
    "colorectal_histology_32x32": task_set_datasets.ColorectalHistology_32x32,
}

# Parent TFDS datasets required before task_set resized builds.
RESIZED_PARENTS = {
    "food101_32x32": ("food101", "1.0.0"),
    "coil100_32x32": ("coil100", "2.0.0"),
    "deep_weeds_32x32": ("deep_weeds", "2.0.0"),
    "sun397_32x32": ("sun397/tfds", "4.0.0"),
    "colorectal_histology_32x32": ("colorectal_histology", "2.0.0"),
}

# Parent TFDS datasets required before task_set tokenized builds.
TOKENIZED_PARENTS = {
    "amazon_us_reviews/Books_v1_02": "0.1.0",
    "amazon_us_reviews/Camera_v1_00": "0.1.0",
    "amazon_us_reviews/Home_v1_00": "0.1.0",
    "amazon_us_reviews/Video_v1_00": "0.1.0",
    "wikipedia/20190301.en": "0.0.4",
    "wikipedia/20190301.hsb": "0.0.4",
    "wikipedia/20190301.ru": "0.0.4",
    "wikipedia/20190301.zh": "0.0.4",
    "wikipedia/20190301.ja": "0.0.4",
}

# Flat dirs at DATA_DIR root: (src_name, builder, version)
STRAY_RELOCATIONS = [
    ("1.0.0", "mnist", "1.0.0"),
]

# Mixed / ambiguous flat builds — quarantine instead of relocating.
STRAY_QUARANTINE = ["0.0.1", "0.1.0"]

# Prefer small/fast builds first; large parents last.
BUILD_PRIORITY = [
    "mnist",
    "coil100_32x32",
    "imdb_reviews/bytes",
    "imdb_reviews/subwords8k",
    "food101_32x32",
    "deep_weeds_32x32",
]

SUN397_LAST = "sun397_32x32"


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_dataset_name(name: str) -> str:
    return DATASET_ALIASES.get(name, name)


def discover_datasets(test_paths: list[Path]) -> set[str]:
    needed: set[str] = set()
    for test_path in test_paths:
        if not test_path.exists():
            print(f"[WARN] {test_path} not found, skipping")
            continue
        for record in load_jsonl(test_path):
            user = json.loads(record["messages"][1]["content"])
            if user.get("task_family") in NO_TFDS_FAMILIES:
                continue
            ds = user.get("task", {}).get("dataset", [None])[0]
            if ds:
                needed.add(normalize_dataset_name(ds))
    return needed


def extract_dataset_parts(dataset_name: str) -> tuple[str, str | None]:
    if "/" not in dataset_name:
        return dataset_name, None
    builder_name, config_name = dataset_name.split("/", 1)
    return builder_name, config_name


def incomplete_build_dirs(builder_name: str, config_name: str | None) -> list[Path]:
    builder_dir = DATA_DIR / builder_name
    if not builder_dir.exists():
        return []
    config_dir = builder_dir / config_name if config_name else builder_dir
    if not config_dir.exists():
        return []
    return sorted(
        entry for entry in config_dir.iterdir()
        if entry.is_dir() and ".incomplete" in entry.name
    )


def clean_incomplete_builds(builder_name: str, config_name: str | None) -> None:
    for entry in incomplete_build_dirs(builder_name, config_name):
        print(f"    Removing stale incomplete dir: {entry}")
        shutil.rmtree(entry, ignore_errors=True)


def is_build_complete(builder_name: str, config_name: str | None) -> bool:
    builder_dir = DATA_DIR / builder_name
    if not builder_dir.exists():
        return False
    search_root = builder_dir / config_name if config_name else builder_dir
    if not search_root.exists():
        return False
    for entry in search_root.iterdir():
        if not entry.is_dir():
            continue
        if ".incomplete" in entry.name:
            continue
        info = entry / "dataset_info.json"
        if not info.exists():
            continue
        try:
            text = info.read_text(encoding="utf-8", errors="replace").lstrip()
        except OSError:
            continue
        if text.startswith("{"):
            return True
    return False


def relocate_stray_builds(dry_run: bool = False) -> None:
    print("=" * 60)
    print("Relocating / quarantining stray TFDS directories")
    print("=" * 60)

    for src_name, builder, version in STRAY_RELOCATIONS:
        src = DATA_DIR / src_name
        dst = DATA_DIR / builder / version
        if not src.exists():
            print(f"  skip {src_name}: not present")
            continue
        if (dst / "dataset_info.json").exists():
            print(f"  skip {src_name}: destination already exists at {dst}")
            continue
        print(f"  relocate {src} -> {dst}")
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

    backup = DATA_DIR / "_stray_backup"
    for src_name in STRAY_QUARANTINE:
        src = DATA_DIR / src_name
        if not src.exists():
            continue
        dst = backup / src_name
        print(f"  quarantine {src} -> {dst}")
        if not dry_run:
            backup.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))


def _builder_is_prepared(builder) -> bool:
    data_dir = builder.data_dir
    if not data_dir:
        return False
    if builder.name == "wikipedia":
        import wikipedia_gcs

        return wikipedia_gcs.parent_ready(data_dir)
    return tf.io.gfile.exists(os.path.join(data_dir, "dataset_info.json"))


def _manual_file_download_config() -> tfds.download.DownloadConfig:
    """Skip checksum validation for local file:// manual artifacts (HF mirrors)."""
    return tfds.download.DownloadConfig(register_checksums=True)


def clean_empty_wikipedia_parents() -> None:
    wiki_root = DATA_DIR / "wikipedia"
    if not wiki_root.exists():
        return
    for config_dir in wiki_root.iterdir():
        if not config_dir.is_dir():
            continue
        version_dirs = [v for v in config_dir.iterdir() if v.is_dir()]
        for version_dir in list(version_dirs):
            info_path = version_dir / "dataset_info.json"
            if ".incomplete" in version_dir.name:
                print(f"  Removing incomplete wikipedia build dir: {version_dir}")
                shutil.rmtree(version_dir, ignore_errors=True)
                continue
            if info_path.exists() and not _dataset_info_json_valid(info_path):
                print(f"  Removing corrupt wikipedia build dir: {version_dir}")
                shutil.rmtree(version_dir, ignore_errors=True)
                continue
            if not any(version_dir.iterdir()):
                print(f"  Removing empty wikipedia build dir: {version_dir}")
                shutil.rmtree(version_dir, ignore_errors=True)
        if not any(config_dir.iterdir()):
            print(f"  Removing empty wikipedia parent dir: {config_dir}")
            shutil.rmtree(config_dir, ignore_errors=True)


def _dataset_info_json_valid(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").lstrip()
        return text.startswith("{")
    except OSError:
        return False


def ensure_resized_parent(dataset_name: str) -> None:
    parent_name, parent_version = RESIZED_PARENTS[dataset_name]
    parent = tfds.builder(parent_name, version=parent_version, data_dir=str(DATA_DIR))
    if _builder_is_prepared(parent):
        print(f"  Parent {parent_name} {parent_version} already prepared")
        return
    print(f"  Downloading parent {parent_name} {parent_version} ...")
    if dataset_name == "deep_weeds_32x32":
        image_zip, labels_csv = deep_weeds_hf.ensure_deep_weeds_files()
        import tensorflow_datasets.image.deep_weeds as deep_weeds_mod

        orig_url = deep_weeds_mod._URL
        orig_labels = deep_weeds_mod._URL_LABELS
        deep_weeds_mod._URL = f"file://{image_zip}"
        deep_weeds_mod._URL_LABELS = f"file://{labels_csv}"
        try:
            parent.download_and_prepare(download_config=_manual_file_download_config())
        finally:
            deep_weeds_mod._URL = orig_url
            deep_weeds_mod._URL_LABELS = orig_labels
        return
    if dataset_name == "sun397_32x32":
        with parallelism.tf_parallel_context():
            manual_dir = sun397_manual.ensure_sun397_manual_files(wait=True)
            image_tar = manual_dir / "SUN397.tar.gz"
            partitions_zip = manual_dir / "Partitions.zip"
            import tensorflow_datasets.image.sun as sun_mod

            orig_images = sun_mod._SUN397_IMAGES_URL
            orig_partitions = sun_mod._SUN397_PARTITIONS_URL
            sun_mod._SUN397_IMAGES_URL = f"file://{image_tar}"
            sun_mod._SUN397_PARTITIONS_URL = f"file://{partitions_zip}"
            try:
                parent.download_and_prepare(download_config=_manual_file_download_config())
            finally:
                sun_mod._SUN397_IMAGES_URL = orig_images
                sun_mod._SUN397_PARTITIONS_URL = orig_partitions
        return
    if dataset_name == "colorectal_histology_32x32":
        tiles_zip = Path(
            "/root/data/cys/taskset_runs/meta/manual_dl/colorectal/"
            "Kather_texture_2016_image_tiles_5000.zip"
        )
        if not tiles_zip.is_file():
            raise FileNotFoundError(
                f"Missing colorectal tiles zip: {tiles_zip}. "
                "Download from Zenodo record 53169 first."
            )
        import tensorflow_datasets.image.colorectal_histology as colo_mod

        orig_url = colo_mod._TILES_DL_URL
        colo_mod._TILES_DL_URL = f"file://{tiles_zip}"
        try:
            parent.download_and_prepare(download_config=_manual_file_download_config())
        finally:
            colo_mod._TILES_DL_URL = orig_url
        return
    parent.download_and_prepare()


def ensure_tokenized_parent(parent_name: str, parent_version: str) -> None:
    parent = tfds.builder(parent_name, version=parent_version, data_dir=str(DATA_DIR))
    if _builder_is_prepared(parent):
        print(f"  Parent {parent_name} {parent_version} already prepared")
        return
    print(f"  Downloading parent {parent_name} {parent_version} ...")
    if parent_name.startswith("amazon_us_reviews/"):
        config = parent_name.split("/", 1)[1]
        local_tsv = amazon_reviews_hf.ensure_amazon_tsv(config)
        import tensorflow_datasets.structured.amazon_us_reviews as amazon_mod

        if config not in amazon_mod._DL_URLS:
            raise ValueError(f"Unknown amazon_us_reviews config: {config}")
        orig_url = amazon_mod._DL_URLS[config]
        amazon_mod._DL_URLS[config] = f"file://{local_tsv}"
        try:
            parent.download_and_prepare(download_config=_manual_file_download_config())
        finally:
            amazon_mod._DL_URLS[config] = orig_url
    elif parent_name.startswith("wikipedia/"):
        import wikipedia_gcs

        parent_dir = parent.data_dir
        if wikipedia_gcs.parent_ready(parent_dir):
            print(f"  Parent {parent_name} {parent_version} already prepared")
            return
        wikipedia_gcs.ensure_parent_from_gcs(parent_name, parent_version, parent_dir)
    else:
        parent.download_and_prepare(
            download_config=parallelism.wikipedia_download_config(tfds)
        )


def ensure_tokenized_parents_for_config(builder_name: str, config_name: str) -> None:
    if builder_name == "tokenized_amazon_reviews":
        base = "_".join(config_name.split("_")[0:-1])
        parent_name = f"amazon_us_reviews/{base}"
        parent_version = TOKENIZED_PARENTS[parent_name]
        ensure_tokenized_parent(parent_name, parent_version)
    elif builder_name == "tokenized_wikipedia":
        base, _ = config_name.split("_")
        parent_name = f"wikipedia/{base}"
        parent_version = TOKENIZED_PARENTS[parent_name]
        ensure_tokenized_parent(parent_name, parent_version)


def build_resized_dataset(dataset_name: str) -> Path:
    ensure_resized_parent(dataset_name)
    # task_set ResizedDataset creates parent builders without data_dir; patch tfds.builder.
    orig_builder = tfds.builder

    def patched_builder(name, version=None, *, data_dir=None, **kwargs):
        if data_dir is None:
            data_dir = str(DATA_DIR)
        return orig_builder(name, version=version, data_dir=data_dir, **kwargs)

    tfds.builder = patched_builder
    try:
        builder_cls = RESIZED_BUILDERS[dataset_name]
        builder = builder_cls(data_dir=str(DATA_DIR))
        if dataset_name == SUN397_LAST:
            with parallelism.tf_parallel_context():
                builder.download_and_prepare()
        else:
            builder.download_and_prepare()
        return Path(builder.data_dir)
    finally:
        tfds.builder = orig_builder


def build_custom_tokenized(builder_name: str, config_name: str) -> Path:
    ensure_tokenized_parents_for_config(builder_name, config_name)
    if builder_name == "tokenized_wikipedia":
        builder_cls = task_set_datasets.TokenizedWikipedia
    elif builder_name == "tokenized_amazon_reviews":
        builder_cls = task_set_datasets.TokenizedAmazonReviews
    else:
        raise ValueError(f"Unknown custom builder: {builder_name}")
    builder = builder_cls(config=config_name, data_dir=str(DATA_DIR))
    builder.download_and_prepare()
    return Path(builder.data_dir)


def build_standard_dataset(builder_name: str, config_name: str | None) -> Path:
    name = f"{builder_name}/{config_name}" if config_name else builder_name
    builder = tfds.builder(name, data_dir=str(DATA_DIR))
    builder.download_and_prepare()
    return Path(builder.data_dir)


def precache_dataset(dataset_name: str, *, resume: bool = False) -> None:
    print(f"\n[{dataset_name}]")
    builder_name, config_name = extract_dataset_parts(dataset_name)

    if is_build_complete(builder_name, config_name):
        print(f"  Already built under {DATA_DIR / builder_name}")
        return

    incomplete = incomplete_build_dirs(builder_name, config_name)
    if resume and incomplete:
        print(f"  Resume: keeping {len(incomplete)} incomplete dir(s), e.g. {incomplete[0].name}")
    elif not resume:
        clean_incomplete_builds(builder_name, config_name)

    print(f"  Building {builder_name}" + (f" / {config_name}" if config_name else "") + " ...")
    if dataset_name in RESIZED_BUILDERS:
        data_dir = build_resized_dataset(dataset_name)
    elif builder_name in ("tokenized_wikipedia", "tokenized_amazon_reviews"):
        data_dir = build_custom_tokenized(builder_name, config_name)
    else:
        data_dir = build_standard_dataset(builder_name, config_name)

    if is_build_complete(builder_name, config_name):
        print(f"  Built at: {data_dir}")
    else:
        raise RuntimeError(f"Build finished but dataset still incomplete: {dataset_name}")


def sort_datasets(datasets: set[str]) -> list[str]:
    priority = {name: i for i, name in enumerate(BUILD_PRIORITY)}

    def sort_key(name: str) -> tuple[int, str]:
        if name == SUN397_LAST:
            return (10_000, name)
        return (priority.get(name, 100), name)

    return sorted(datasets, key=sort_key)


def print_dataset_sources() -> None:
    print(
        """
Dataset download sources (TFDS auto-downloads unless noted):
  Standard image:
    mnist          https://www.tensorflow.org/datasets/catalog/mnist
    cifar10/100    https://www.tensorflow.org/datasets/catalog/cifar10
    fashion_mnist  https://www.tensorflow.org/datasets/catalog/fashion_mnist
    food101        https://www.tensorflow.org/datasets/catalog/food101  (parent for food101_32x32)
    coil100        https://www.tensorflow.org/datasets/catalog/coil100
    deep_weeds     https://www.tensorflow.org/datasets/catalog/deep_weeds
    sun397/tfds    https://www.tensorflow.org/datasets/catalog/sun397
  Text:
    imdb_reviews   https://www.tensorflow.org/datasets/catalog/imdb_reviews
    wikipedia/*    https://www.tensorflow.org/datasets/catalog/wikipedia
    amazon_us_reviews/*  https://www.tensorflow.org/datasets/catalog/amazon_us_reviews
                       HF mirror: huggingface.co/datasets/shansoround/amazon_reviews
                       (S3 bucket returns 403; precache uses HF TSV -> TFDS)
  Custom task_set builds (not in TFDS catalog — built locally via this script):
    food101_32x32, coil100_32x32, deep_weeds_32x32, sun397_32x32
    tokenized_wikipedia/*, tokenized_amazon_reviews/*
  Cache directory: """
        + str(DATA_DIR)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", type=Path, action="append", default=None)
    parser.add_argument(
        "--dataset",
        type=str,
        action="append",
        default=None,
        help="Build a specific dataset (e.g. mnist, food101_32x32).",
    )
    parser.add_argument("--relocate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-sources", action="store_true")
    parser.add_argument("--skip-relocate", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip deleting .incomplete* build dirs; skip datasets already complete.",
    )
    parser.add_argument(
        "--beam-workers",
        type=int,
        default=0,
        help="Beam DirectRunner workers for Wikipedia (0=auto, ~cpus/4).",
    )
    parser.add_argument(
        "--tf-threads",
        type=int,
        default=0,
        help="TF/OpenCV threads for SUN397 builds (0=auto, ~cpus/8).",
    )
    args = parser.parse_args()

    if args.list_sources:
        print_dataset_sources()
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TFDS_DATA_DIR"] = str(DATA_DIR)

    beam_workers, tf_threads = parallelism.configure_parallelism(
        beam_workers=args.beam_workers,
        tf_threads=args.tf_threads,
    )
    print(
        f"Parallelism: beam_workers={beam_workers}, tf_threads={tf_threads} "
        f"(logical_cpus={parallelism.logical_cpu_count()})"
    )

    if not args.skip_relocate:
        relocate_stray_builds(dry_run=args.dry_run)

    clean_empty_wikipedia_parents()

    if args.relocate_only:
        return

    if args.dataset:
        needed_datasets = {normalize_dataset_name(d) for d in args.dataset}
    else:
        test_paths = args.test_data or [DEFAULT_TEST_DATA]
        needed_datasets = discover_datasets(test_paths)

    ordered = sort_datasets(needed_datasets)
    print("=" * 60)
    print(f"Datasets to pre-cache: {len(ordered)}")
    for ds in ordered:
        status = "ready" if is_build_complete(*extract_dataset_parts(ds)) else "pending"
        print(f"  [{status:7}] {ds}")
    print("=" * 60)
    print_dataset_sources()

    failures: list[tuple[str, str]] = []
    for ds in ordered:
        if is_build_complete(*extract_dataset_parts(ds)):
            continue
        try:
            precache_dataset(ds, resume=args.resume)
        except Exception as exc:
            print(f"  FAILED {ds}: {exc}")
            failures.append((ds, str(exc)))

    print("\n" + "=" * 60)
    if failures:
        print(f"Pre-cache finished with {len(failures)} failure(s):")
        for ds, err in failures:
            print(f"  - {ds}: {err}")
        print("=" * 60)
        sys.exit(1)
    print("Pre-cache complete — all datasets ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
