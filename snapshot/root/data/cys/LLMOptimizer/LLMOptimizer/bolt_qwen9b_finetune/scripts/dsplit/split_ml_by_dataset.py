#!/usr/bin/env python3
"""Dataset-level (task-level) 80/20 split of the ML HPO tasks.

Unlike scripts/split_ml_80_20.py -- which splits (dataset x algorithm) records and
therefore lets 77/95 datasets appear on BOTH sides -- this splits whole datasets.
Every algorithm of a dataset lands entirely in train or entirely in test, so a
model never sees a test dataset under any algorithm during SFT / ORPO / RL.

Two extra constraints on top of plain dataset-level splitting:

  1. Sibling groups are atomic. Three pairs are near-duplicates of each other
     (verified from the dataset profiles in the prompts), so splitting them
     apart would reintroduce leakage one level down:
       topo_2_1 / yprop_4_1              -- identical n_samples (8885), same rows
       QSAR-TID-11 / QSAR-TID-10980      -- both exactly 1025 FCFP features
       wine_quality / wine-quality-white -- 6497 = 4898 white + 1599 red, same 12 feats
     (albert/dilbert/fabert etc. are NOT grouped: shared ChaLearn naming, unrelated data.)

  2. The seed is chosen so the split is simultaneously 80/20 by dataset count,
     by SFT event count, by ORPO pair count and by (dataset x algo) record count.
     Per-dataset SFT event counts vary 20x (2..40), so dataset-count balance does
     NOT imply record-count balance -- the search reconciles them.
     Only COUNTS are balanced, never anything performance-related.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

DATA = Path("/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data")

SIBLING_GROUPS = [
    ("topo_2_1", "yprop_4_1"),
    ("QSAR-TID-11", "QSAR-TID-10980"),
    ("wine_quality", "wine-quality-white"),
]

# Sources whose per-dataset volume we balance. (file, key-in-manifest)
VOLUME_SOURCES = [
    ("p5_s1b_sft.jsonl", "sft_events"),
    ("p5_s2_dpo.jsonl", "orpo_pairs"),
]


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--seeds", type=int, default=4000, help="how many candidate seeds to search")
    ap.add_argument("--seed", type=int, default=None, help="force a specific seed, skip the search")
    ap.add_argument("--out-prefix", default="ml_ds")
    args = ap.parse_args()

    d = args.data_dir

    # ---- inventory: dataset -> task_type, task_id, algorithms ----
    tt: dict[str, str] = {}
    task_id: dict[str, str] = {}
    algos: dict[str, set[str]] = collections.defaultdict(set)
    source_records = []
    for fn in ("ml_train_80_fixed.jsonl", "ml_test_20_fixed.jsonl"):
        for rec in load_jsonl(d / fn):
            m = rec["metadata"]
            ds = m["dataset"]
            tt[ds] = m["task_type"]
            task_id[ds] = str(m.get("task_id", ""))
            algos[ds].add(m["algorithm"])
            source_records.append(rec)

    # ---- per-dataset volumes for the sources we balance ----
    volumes: dict[str, collections.Counter] = {}
    for fn, key in VOLUME_SOURCES:
        p = d / fn
        volumes[key] = collections.Counter(r["metadata"]["dataset"] for r in load_jsonl(p)) if p.is_file() else collections.Counter()
    volumes["ml_records"] = collections.Counter({ds: len(a) for ds, a in algos.items()})

    # ---- build atomic split units ----
    member: dict[str, str] = {}
    for i, grp in enumerate(SIBLING_GROUPS):
        for ds in grp:
            if ds in tt:
                member[ds] = f"pair{i}"
    unit_members: dict[str, list[str]] = collections.defaultdict(list)
    for ds in tt:
        unit_members[member.get(ds, f"solo:{ds}")].append(ds)

    units = []
    for uid, ms in unit_members.items():
        units.append({
            "id": uid,
            "members": sorted(ms),
            "cls": sum(1 for m in ms if tt[m] == "cls"),
            "rgs": sum(1 for m in ms if tt[m] == "rgs"),
            **{k: sum(v[m] for m in ms) for k, v in volumes.items()},
        })

    n_cls = sum(u["cls"] for u in units)
    n_rgs = sum(u["rgs"] for u in units)
    target_cls = round(n_cls * args.test_frac)
    target_rgs = round(n_rgs * args.test_frac)
    totals = {k: sum(v.values()) for k, v in volumes.items()}

    def greedy(seed: int):
        rnd = random.Random(seed)
        pool = units[:]
        rnd.shuffle(pool)
        c = r = 0
        test = []
        for u in pool:
            if u["cls"] + c <= target_cls and u["rgs"] + r <= target_rgs and (u["cls"] or u["rgs"]):
                test.append(u)
                c += u["cls"]
                r += u["rgs"]
        return (c, r), test

    def deviation(test) -> float:
        return sum(
            abs(sum(u[k] for u in test) / totals[k] - args.test_frac)
            for k in totals if totals[k]
        )

    if args.seed is not None:
        (c, r), test = greedy(args.seed)
        if (c, r) != (target_cls, target_rgs):
            raise SystemExit(f"forced seed {args.seed} misses stratification target: got {(c, r)} want {(target_cls, target_rgs)}")
        best_seed, best_test = args.seed, test
    else:
        best_seed = best_test = None
        best_dev = float("inf")
        for seed in range(args.seeds):
            (c, r), test = greedy(seed)
            if (c, r) != (target_cls, target_rgs):
                continue
            dev = deviation(test)
            if dev < best_dev:
                best_dev, best_seed, best_test = dev, seed, test
        if best_test is None:
            raise SystemExit("no seed satisfied the stratification target")

    test_ds = sorted(m for u in best_test for m in u["members"])
    test_set = set(test_ds)
    train_ds = sorted(ds for ds in tt if ds not in test_set)

    # ---- hard invariants ----
    assert not (set(train_ds) & test_set), "dataset appears in both splits"
    assert len(train_ds) + len(test_ds) == len(tt)
    for grp in SIBLING_GROUPS:
        present = [g for g in grp if g in tt]
        assert len({g in test_set for g in present}) == 1, f"sibling group split apart: {grp}"

    # ---- emit split files ----
    train_recs = [r for r in source_records if r["metadata"]["dataset"] not in test_set]
    test_recs = [r for r in source_records if r["metadata"]["dataset"] in test_set]
    train_path = d / f"{args.out_prefix}_train_{len(train_ds)}.jsonl"
    test_path = d / f"{args.out_prefix}_test_{len(test_ds)}.jsonl"
    write_jsonl(train_path, train_recs)
    write_jsonl(test_path, test_recs)

    # ---- manifest ----
    def share(key, ds_list):
        return sum(volumes[key][x] for x in ds_list)

    manifest = {
        "split": "dataset_level_80_20",
        "seed": best_seed,
        "seeds_searched": None if args.seed is not None else args.seeds,
        "test_frac": args.test_frac,
        "sibling_groups": [list(g) for g in SIBLING_GROUPS],
        "sibling_group_side": {
            "+".join(g): ("test" if g[0] in test_set else "train") for g in SIBLING_GROUPS
        },
        "n_units": len(units),
        "counts": {
            "train_datasets": len(train_ds),
            "test_datasets": len(test_ds),
            "train_cls": sum(1 for x in train_ds if tt[x] == "cls"),
            "test_cls": sum(1 for x in test_ds if tt[x] == "cls"),
            "train_rgs": sum(1 for x in train_ds if tt[x] == "rgs"),
            "test_rgs": sum(1 for x in test_ds if tt[x] == "rgs"),
            "train_records": len(train_recs),
            "test_records": len(test_recs),
        },
        "balance": {
            k: {
                "train": share(k, train_ds),
                "test": share(k, test_ds),
                "total": totals[k],
                "test_pct": round(100.0 * share(k, test_ds) / totals[k], 3) if totals[k] else None,
            }
            for k in totals
        },
        "train_datasets": train_ds,
        "test_datasets": test_ds,
        "test_dataset_detail": [
            {"dataset": x, "task_type": tt[x], "task_id": task_id[x], "n_algos": len(algos[x])}
            for x in test_ds
        ],
        "files": {"train": str(train_path), "test": str(test_path)},
    }
    man_path = d / f"{args.out_prefix}_split_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"seed={best_seed}  units={len(units)}  (sibling groups kept atomic)")
    print(f"train {len(train_ds)} datasets / {len(train_recs)} records -> {train_path}")
    print(f"test  {len(test_ds)} datasets / {len(test_recs)} records -> {test_path}")
    for k in sorted(totals):
        b = manifest["balance"][k]
        print(f"  {k:<12} train={b['train']:<6} test={b['test']:<6} test_pct={b['test_pct']}%")
    print(f"manifest -> {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
