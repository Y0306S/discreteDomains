#!/usr/bin/env python3
"""Download missing Wikipedia TFDS parent caches from GCS (20190301.* @ 0.0.4)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATA_DIR = Path("/root/data/cys/tensorflow_datasets")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import wikipedia_gcs  # noqa: E402

LANGS = ("en", "hsb", "ru", "zh", "ja")
VERSION = "0.0.4"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="*", default=["ru", "zh", "ja"])
    parser.add_argument("--max-workers", type=int, default=16)
    args = parser.parse_args()

    for lang in args.langs:
        config = f"20190301.{lang}"
        dataset_name = f"wikipedia/{config}"
        parent_dir = DATA_DIR / "wikipedia" / config / VERSION
        if wikipedia_gcs.parent_ready(
            str(parent_dir),
            expected_shards=wikipedia_gcs.expected_shard_count(
                str(parent_dir),
                wikipedia_gcs.gcs_paths_for_version(dataset_name, VERSION),
            ),
        ):
            print(f"SKIP {config}: parent already ready at {parent_dir}")
            continue
        print(f"=== Downloading {dataset_name}/{VERSION} -> {parent_dir} ===")
        wikipedia_gcs.ensure_parent_from_gcs(
            dataset_name,
            VERSION,
            parent_dir,
            max_workers=args.max_workers,
        )
        print(f"DONE {config}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Download missing Wikipedia TFDS parent caches from GCS (20190301.* @ 0.0.4)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATA_DIR = Path("/root/data/cys/tensorflow_datasets")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import wikipedia_gcs  # noqa: E402

LANGS = ("en", "hsb", "ru", "zh", "ja")
VERSION = "0.0.4"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="*", default=["ru", "zh", "ja"])
    parser.add_argument("--max-workers", type=int, default=16)
    args = parser.parse_args()

    for lang in args.langs:
        config = f"20190301.{lang}"
        dataset_name = f"wikipedia/{config}"
        parent_dir = DATA_DIR / "wikipedia" / config / VERSION
        if wikipedia_gcs.parent_ready(
            str(parent_dir),
            expected_shards=wikipedia_gcs.expected_shard_count(
                str(parent_dir),
                wikipedia_gcs.gcs_paths_for_version(dataset_name, VERSION),
            ),
        ):
            print(f"SKIP {config}: parent already ready at {parent_dir}")
            continue
        print(f"=== Downloading {dataset_name}/{VERSION} -> {parent_dir} ===")
        wikipedia_gcs.ensure_parent_from_gcs(
            dataset_name,
            VERSION,
            parent_dir,
            max_workers=args.max_workers,
        )
        print(f"DONE {config}")


if __name__ == "__main__":
    main()
