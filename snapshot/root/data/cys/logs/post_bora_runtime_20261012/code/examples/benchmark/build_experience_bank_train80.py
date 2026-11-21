#!/usr/bin/env python3
"""Filter experience_bank_v3.jsonl to (dataset, algorithm) pairs in ml_train_80_fixed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_train_pairs(path: Path) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            meta = json.loads(line)["metadata"]
            pairs.add((str(meta["dataset"]), str(meta["algorithm"])))
    return pairs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--bank",
        type=Path,
        default=Path("/root/blockdata/llmoptimizer_benchmark/history_data/experience_bank_v3.jsonl"),
    )
    p.add_argument(
        "--train-jsonl",
        type=Path,
        default=Path("/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/ml_train_80_fixed.jsonl"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("/root/blockdata/llmoptimizer_benchmark/history_data/experience_bank_v3_train80.jsonl"),
    )
    args = p.parse_args()

    pairs = load_train_pairs(args.train_jsonl)
    kept = 0
    skipped = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.bank.open() as fin, args.out.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            prob = rec.get("problem") or {}
            key = (str(prob.get("dataset_id") or ""), str(prob.get("algorithm_id") or ""))
            if key in pairs:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
            else:
                skipped += 1
    print(f"train_pairs={len(pairs)} kept={kept} skipped={skipped} out={args.out}")


if __name__ == "__main__":
    main()
