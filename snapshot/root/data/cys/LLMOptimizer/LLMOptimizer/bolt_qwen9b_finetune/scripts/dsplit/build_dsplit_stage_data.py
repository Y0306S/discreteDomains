#!/usr/bin/env python3
"""Restrict the SFT and ORPO-source data to the dataset-level train split.

Filters by metadata.dataset against the split manifest's train_datasets. Nothing
is regenerated from raw histories and no logic changes -- these are the same
records the original run used, minus those belonging to held-out datasets.

  p5_s1b_sft.jsonl      -> p5_s1b_sft_dsplit.jsonl        (+ _train projection)
  p5_s2_dpo.jsonl       -> p5_s2_dpo_dsplit.jsonl         (ORPO stage-A source)

The ORPO stage-B (T2 attribution) file is NOT produced here -- run
export_p5_attribution_t2_t3.py --src p5_s2_dpo_dsplit.jsonl so the blame /
repair logic stays byte-identical to the original.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

DATA = Path("/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data")


def load_jsonl(p: Path):
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--suffix", default="dsplit")
    args = ap.parse_args()

    d = args.data_dir
    man_path = args.manifest or (d / "ml_ds_split_manifest.json")
    man = json.loads(man_path.read_text())
    train_ds = set(man["train_datasets"])
    test_ds = set(man["test_datasets"])

    report = {"manifest": str(man_path), "split_seed": man["seed"], "stages": {}}

    # ---- SFT ----
    src = d / "p5_s1b_sft.jsonl"
    rows = list(load_jsonl(src))
    kept = [r for r in rows if r["metadata"]["dataset"] in train_ds]
    dropped = [r for r in rows if r["metadata"]["dataset"] in test_ds]
    if len(kept) + len(dropped) != len(rows):
        raise SystemExit("SFT rows reference datasets outside the split")

    full_out = d / f"p5_s1b_sft_{args.suffix}.jsonl"
    train_out = d / f"p5_s1b_sft_{args.suffix}_train.jsonl"
    with full_out.open("w", encoding="utf-8") as f, train_out.open("w", encoding="utf-8") as g:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            g.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
    report["stages"]["sft"] = {
        "src": str(src), "src_n": len(rows), "kept": len(kept), "dropped": len(dropped),
        "kept_pct": round(100.0 * len(kept) / len(rows), 3),
        "datasets": len({r["metadata"]["dataset"] for r in kept}),
        "out": str(full_out), "llamafactory_file": str(train_out),
    }

    # ---- ORPO stage-A source ----
    src = d / "p5_s2_dpo.jsonl"
    rows = list(load_jsonl(src))
    kept = [r for r in rows if r["metadata"]["dataset"] in train_ds]
    dropped = [r for r in rows if r["metadata"]["dataset"] in test_ds]
    if len(kept) + len(dropped) != len(rows):
        raise SystemExit("DPO rows reference datasets outside the split")
    dpo_out = d / f"p5_s2_dpo_{args.suffix}.jsonl"
    with dpo_out.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report["stages"]["orpo_source"] = {
        "src": str(src), "src_n": len(rows), "kept": len(kept), "dropped": len(dropped),
        "kept_pct": round(100.0 * len(kept) / len(rows), 3),
        "datasets": len({r["metadata"]["dataset"] for r in kept}),
        "out": str(dpo_out),
        "next": "export_p5_attribution_t2_t3.py --src " + str(dpo_out),
    }

    # ---- leakage assertion ----
    for name, path in [("sft", full_out), ("orpo_source", dpo_out)]:
        seen = {r["metadata"]["dataset"] for r in load_jsonl(path)}
        leak = sorted(seen & test_ds)
        if leak:
            raise SystemExit(f"{name} still contains test datasets: {leak}")
        report["stages"][name]["test_dataset_leakage"] = "none"

    rep_path = d / f"{args.suffix}_stage_data_report.json"
    rep_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for k, v in report["stages"].items():
        print(f"{k:<12} {v['src_n']:>5} -> {v['kept']:>5} kept ({v['kept_pct']}%)  "
              f"{v['dropped']:>4} dropped  {v['datasets']} datasets  leakage={v['test_dataset_leakage']}")
    print(f"report -> {rep_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
