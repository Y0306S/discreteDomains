#!/usr/bin/env python3
"""Build TaskSet warm-start SFT data from the train838 TPE BO corpus.

Same formula as the ML warm-start data (``build_bolt_ml_sft_data_fixed.py``): one
record per BO history, whose target is the single best successful observation
(lowest objective, first occurrence on ties). The prompt follows the ML
warm-start layout with the TaskSet task spec in place of the dataset profile and
the full CASH space in place of a fixed algorithm.

Writes (nothing is trained or launched):
  <out-dir>/taskset_ws_train838.jsonl         sharegpt "messages" records + metadata
  <out-dir>/taskset_ws_test210_prompts.jsonl  same prompts for the 210 test tasks, no target
  <out-dir>/taskset_ws_train838.report.json   provenance and counts
"""

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mindware.domains.dlhpo.taskset_llm import spec as S
from mindware.domains.dlhpo.taskset_space import space_from_taskset
from taskset_transfer_common import METRIC, PROTOCOL, SPLIT_SHA256, digest, load_corpus, read_split

CORPUS = Path("/root/data/logs/taskset_transfer_train838_20261002/corpus")
OUT_DIR = Path("/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data")


def best_observation(record):
    ys = [float(y) for y in record["y_minimize"]]
    index = min(range(len(ys)), key=ys.__getitem__)
    return index, ys[index]


def build(corpus, out_dir, check_sources):
    manifest, tasks = load_corpus(corpus, check_sources=check_sources)
    split = read_split()
    specs = S.load_task_specs()
    cs = space_from_taskset()
    records, families = [], collections.Counter()
    for task_id, record in tasks.items():
        index, objective = best_observation(record)
        config, notes = S.coerce_config(record["configs"][index], cs)
        if config != record["configs"][index] or any(notes.values()):
            raise ValueError(f"Best config of {task_id} is not a canonical CASH configuration")
        families[config[S.FAMILY_HP]] += 1
        records.append({
            "metadata": {
                "task_id": task_id, "task_family": S.task_family_of(task_id),
                "optimizer_family": config[S.FAMILY_HP], "metric": METRIC,
                "objective": objective, "objective_direction": "minimize",
                "source_row": int(record["source_rows"][index]), "n_observations": len(record["y_minimize"]),
                "source_sha256": record["source_sha256"],
            },
            "messages": S.warm_start_messages(task_id, specs, cs)
                        + [{"role": "assistant", "content": S.config_json(config)}],
        })
    test = [{"metadata": {"task_id": t, "task_family": S.task_family_of(t)},
             "messages": S.warm_start_messages(t, specs, cs)} for t in split["test"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"train": out_dir / "taskset_ws_train838.jsonl",
             "test_prompts": out_dir / "taskset_ws_test210_prompts.jsonl"}
    for key, rows in (("train", records), ("test_prompts", test)):
        with paths[key].open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "formula": "top-1 best successful observation per BO history (ML warm-start formula)",
        "corpus": str(corpus), "corpus_manifest_sha256": digest(corpus / "manifest.json"),
        "split_sha256": SPLIT_SHA256, "protocol": PROTOCOL, "metric": METRIC,
        "source_policy": manifest["source_policy"],
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "spec_sha256": hashlib.sha256(Path(S.__file__).read_bytes()).hexdigest(),
        "n_train": len(records), "n_test_prompts": len(test),
        "best_optimizer_family_counts": dict(sorted(families.items())),
        "files": {k: {"path": str(p), "sha256": digest(p)} for k, p in paths.items()},
    }
    (out_dir / "taskset_ws_train838.report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--skip-source-check", action="store_true",
                        help="Do not re-hash the 838 raw TPE histories against the corpus.")
    args = parser.parse_args()
    print(json.dumps(build(args.corpus, args.out_dir, not args.skip_source_check), indent=2))
