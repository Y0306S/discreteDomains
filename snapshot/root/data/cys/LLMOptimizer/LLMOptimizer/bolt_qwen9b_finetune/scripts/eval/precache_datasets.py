#!/usr/bin/env python3
"""Pre-download and pre-build all Wikipedia datasets needed for DL evaluation.

This script:
1. Parses all dl_test samples to find unique (language, dataset) pairs
2. Pre-downloads raw Wikipedia via tfds
3. Pre-builds tokenized_wikipedia datasets via task_set's custom builder
4. After this, evaluation just trains models (no more I/O waits)
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
GOOGLE_RESEARCH_ROOT = Path("/root/data/cys/blockdata_archive/LLMOptimizer/google-research")

sys.path.insert(0, str(GOOGLE_RESEARCH_ROOT))

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
from task_set import datasets as task_set_datasets

DL_TEST_PATH = BASE_DIR / "data" / "dl_test.jsonl"
TASK_CONFIGS_PATH = GOOGLE_RESEARCH_ROOT / "task_set/tasks/configs/char_rnn_language_model.json"


def load_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def find_task_seed(target_task, task_configs):
    for key, cfg_list in task_configs.items():
        cfg = cfg_list[0]
        if (cfg.get("embed_dim") == target_task.get("embed_dim") and
            cfg.get("vocab_size") == target_task.get("vocab_size") and
            cfg.get("core", [None])[0] == target_task.get("core", [None])[0] and
            cfg.get("core", [None, {}])[1].get("core_dim") == target_task.get("core", [None, {}])[1].get("core_dim") and
            cfg.get("trainable_init") == target_task.get("trainable_init")):
            return key, cfg
    return None, None


def get_dataset_spec(task_cfg):
    """Extract the dataset name and params from task config."""
    dataset = task_cfg.get("dataset", [])
    if not dataset:
        return None
    if isinstance(dataset, list) and len(dataset) >= 1:
        name = dataset[0]
        params = dataset[1] if len(dataset) > 1 else {}
        return name, params
    return None


def extract_wikipedia_version(dataset_name):
    """Extract Wikipedia version from dataset name like 'tokenized_wikipedia/20190301.zh_bytes'."""
    if not dataset_name or "wikipedia" not in dataset_name:
        return None
    parts = dataset_name.split("/")
    if len(parts) >= 2:
        return parts[1]  # e.g., '20190301.zh_bytes'
    return None


def precache_wikipedia(version):
    """Pre-download raw Wikipedia dataset."""
    print(f"  [1/2] Pre-downloading raw Wikipedia: wikipedia/{version} ...")
    builder = tfds.builder(f"wikipedia/{version}")
    builder.download_and_prepare()
    print(f"  [1/2] Done. Cached at: {builder.data_dir}")


def precache_tokenized_wikipedia(version, params):
    """Pre-build tokenized Wikipedia via task_set's custom builder."""
    print(f"  [2/2] Pre-building tokenized Wikipedia: tokenized_wikipedia/{version} ...")
    print(f"        Params: {params}")

    # Create a custom builder instance with the given params
    # The builder class is TokenizedWikipedia, registered as "tokenized_wikipedia/version"
    builder_cls = task_set_datasets.TokenizedWikipedia
    builder = builder_cls()
    builder.download_and_prepare()
    print(f"  [2/2] Done. Cached at: {builder.data_dir}")


def main():
    print("=" * 60)
    print("Pre-caching all Wikipedia datasets for DL evaluation")
    print("=" * 60)
    print()

    # Load test data
    test_records = load_jsonl(DL_TEST_PATH)
    with TASK_CONFIGS_PATH.open() as f:
        task_configs = json.load(f)

    # Find all unique datasets
    unique_datasets = set()
    dataset_details = {}

    for i, record in enumerate(test_records):
        user_msg = json.loads(record["messages"][1]["content"])
        target_task = user_msg["task"]
        task_name, task_cfg = find_task_seed(target_task, task_configs)

        if not task_cfg:
            print(f"  [WARN] Sample {i}: could not find matching task config")
            continue

        spec = get_dataset_spec(task_cfg)
        if not spec:
            print(f"  [WARN] Sample {i}: no dataset spec found")
            continue

        dataset_name, params = spec
        unique_datasets.add(dataset_name)

        if dataset_name not in dataset_details:
            dataset_details[dataset_name] = {
                "params": params,
                "samples": [i],
                "task_name": task_name,
            }
        else:
            dataset_details[dataset_name]["samples"].append(i)

    print(f"Found {len(unique_datasets)} unique dataset(s) across {len(test_records)} test samples:")
    for name in sorted(unique_datasets):
        info = dataset_details[name]
        print(f"  - {name}")
        print(f"    Used by {len(info['samples'])} samples: {info['samples'][:5]}{'...' if len(info['samples']) > 5 else ''}")
    print()

    # Pre-cache each unique dataset
    for dataset_name in sorted(unique_datasets):
        info = dataset_details[dataset_name]
        params = info["params"]
        wiki_version = extract_wikipedia_version(dataset_name)

        print(f"\n[{dataset_name}]")

        if wiki_version:
            # Step 1: Pre-download raw Wikipedia
            try:
                precache_wikipedia(wiki_version)
            except Exception as e:
                print(f"  [ERROR] Failed to download Wikipedia: {e}")
                continue

            # Step 2: Pre-build tokenized version
            try:
                precache_tokenized_wikipedia(wiki_version, params)
            except Exception as e:
                print(f"  [ERROR] Failed to build tokenized dataset: {e}")
                continue
        else:
            print(f"  [SKIP] Not a Wikipedia dataset")

    print()
    print("=" * 60)
    print("Pre-caching complete!")
    print("=" * 60)
    print("All datasets are now cached. Subsequent evaluations will")
    print("skip download/tokenization and go straight to training.")


if __name__ == "__main__":
    main()
