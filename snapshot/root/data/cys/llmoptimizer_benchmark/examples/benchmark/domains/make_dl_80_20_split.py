#!/usr/bin/env python3
"""Re-balance the DL SFT split from 85/15 to 80/20, preserving task spread.

Approach: INCREMENTAL, train -> test only
----------------------------------------
Rather than reshuffling all 858 records, this moves the minimum number of
records from train to test to hit 20% per task family. Consequences:

  * the existing 143 test records stay in test, so nothing ever moves
    test -> train (no risk of a task the model was evaluated on becoming
    training data);
  * only ~29 train tasks change side, so BO runs on the other ~686 train tasks
    remain valid. A fresh stratified reshuffle would invalidate far more.

Spread is preserved on two axes: the per-family test share is driven to 20%, and
within each family the moved records are drawn round-robin across
optimizer_family groups so no single optimizer label is over-drained. This
mirrors the original stratification in scripts/split_85_15.py, which stratified
on (task_family, optimizer_family, core_type).

Nothing is overwritten: outputs are dl_train_80.jsonl / dl_test_20.jsonl beside
the originals.

Usage:
  make_dl_80_20_split.py --dry-run          # report the plan, write nothing
  make_dl_80_20_split.py                    # write the new split
  make_dl_80_20_split.py --bo-run-dir /root/data/logs/taskset_bo_train_1000steps
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = Path("/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data")
TASK_CONFIGS = Path("/root/data/cys/google-research/task_set/tasks/configs")
SRC_TRAIN, SRC_TEST = "dl_train_85.jsonl", "dl_test_15.jsonl"
OUT_TRAIN, OUT_TEST = "dl_train_80.jsonl", "dl_test_20.jsonl"

FAMILY_RE = re.compile(r"^(.*?_family)_seed\d+$")


def _hash(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def _canonical_spec(spec: Any) -> Any:
    """losg records store [problem, params, seed] as a dict; configs use a list."""
    if isinstance(spec, dict) and "losg_problem" in spec:
        return [spec["losg_problem"], spec.get("params"), spec.get("init_seed")]
    return spec


def build_spec_index() -> Dict[str, str]:
    index: Dict[str, str] = {}
    for path in sorted(TASK_CONFIGS.glob("*.json")):
        for task_id, value in json.loads(path.read_text()).items():
            spec = value[0] if isinstance(value, list) else value
            index[_hash(spec)] = task_id
    return index


def load_side(name: str, index: Dict[str, str]) -> List[Dict[str, Any]]:
    """Return [{record, task_id, task_family, optimizer_family}] for one jsonl side."""
    out = []
    for line in (DATA / name).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        user = json.loads(rec["messages"][1]["content"])
        asst = json.loads(rec["messages"][2]["content"])
        out.append({
            "record": rec,
            "task_id": index.get(_hash(_canonical_spec(user["task"]))),
            "task_family": user["task_family"],
            "optimizer_family": asst.get("optimizer_family", "unknown"),
        })
    return out


def plan_moves(train: List[Dict[str, Any]], test: List[Dict[str, Any]],
               test_frac: float, seed: int) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    """Choose which train records move to test to hit `test_frac` per family."""
    by_fam_train: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in train:
        by_fam_train[r["task_family"]].append(r)
    test_counts = Counter(r["task_family"] for r in test)

    rng = random.Random(seed)
    targets: Dict[str, int] = {}
    moves: List[Dict[str, Any]] = []

    for fam in sorted(by_fam_train):
        total = len(by_fam_train[fam]) + test_counts[fam]
        want_test = round(total * test_frac)
        need = max(0, want_test - test_counts[fam])
        targets[fam] = need
        if need == 0:
            continue

        # Round-robin across optimizer_family groups so the moved set keeps the
        # family's optimizer_family mix instead of draining one label.
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in by_fam_train[fam]:
            groups[r["optimizer_family"]].append(r)
        for g in groups.values():
            rng.shuffle(g)
        order = sorted(groups, key=lambda k: (-len(groups[k]), k))

        picked: List[Dict[str, Any]] = []
        while len(picked) < need:
            progressed = False
            for k in order:
                if len(picked) >= need:
                    break
                if groups[k]:
                    picked.append(groups[k].pop())
                    progressed = True
            if not progressed:
                break
        moves.extend(picked)
    return targets, moves


def balance_table(train: List[Dict[str, Any]], test: List[Dict[str, Any]], key: str) -> str:
    ftr, fte = Counter(r[key] for r in train), Counter(r[key] for r in test)
    lines = [f"  {key:<38} {'train':>6} {'test':>5} {'test%':>7}"]
    for k in sorted(set(ftr) | set(fte)):
        tot = ftr[k] + fte[k]
        lines.append(f"  {k:<38} {ftr[k]:>6} {fte[k]:>5} {100*fte[k]/tot if tot else 0:>6.1f}%")
    tot = len(train) + len(test)
    lines.append(f"  {'TOTAL':<38} {len(train):>6} {len(test):>5} {100*len(test)/tot:>6.1f}%")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-frac", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42, help="matches split_85_15.py's seed")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--bo-run-dir", type=Path, default=None,
                   help="report which completed BO runs land on the new test side")
    args = p.parse_args(argv)

    index = build_spec_index()
    train = load_side(SRC_TRAIN, index)
    test = load_side(SRC_TEST, index)
    unmapped = [r for r in train + test if not r["task_id"]]
    if unmapped:
        print(f"! {len(unmapped)} records could not be mapped to a task_id", file=sys.stderr)

    print(f"=== BEFORE ({len(train)}/{len(test)} = "
          f"{100*len(test)/(len(train)+len(test)):.1f}% test) ===")
    print(balance_table(train, test, "task_family"))

    targets, moves = plan_moves(train, test, args.test_frac, args.seed)
    move_ids = {r["task_id"] for r in moves}
    new_train = [r for r in train if r["task_id"] not in move_ids]
    new_test = test + moves

    print(f"\n=== MOVES train -> test ({len(moves)} records) ===")
    for fam in sorted(targets):
        if targets[fam]:
            got = [r for r in moves if r["task_family"] == fam]
            mix = dict(Counter(r["optimizer_family"] for r in got))
            print(f"  {fam:<38} move {targets[fam]:>2}   optimizer mix: {mix}")

    print(f"\n=== AFTER ({len(new_train)}/{len(new_test)} = "
          f"{100*len(new_test)/(len(new_train)+len(new_test)):.1f}% test) ===")
    print(balance_table(new_train, new_test, "task_family"))
    print()
    print(balance_table(new_train, new_test, "optimizer_family"))

    # sanity: no record on both sides, none lost
    assert len(new_train) + len(new_test) == len(train) + len(test)
    ids_tr = {r["task_id"] for r in new_train}
    ids_te = {r["task_id"] for r in new_test}
    overlap = ids_tr & ids_te
    print(f"\n  integrity: total preserved={len(new_train)+len(new_test)}  "
          f"train/test task_id overlap={len(overlap)} (must be 0)")
    assert not overlap, f"overlap: {sorted(overlap)[:5]}"

    if args.bo_run_dir:
        done = {h.parents[2].name for h in args.bo_run_dir.glob("*/*/*/history.json")}
        invalid = sorted(done & move_ids)
        print(f"\n=== impact on BO runs in {args.bo_run_dir.name} ===")
        print(f"  completed BO tasks           : {len(done)}")
        print(f"  now on the TEST side         : {len(invalid)}  <- no longer train-side runs")
        for t in invalid:
            print(f"     {t}")
        print(f"  still valid as train-side    : {len(done) - len(invalid)}")

    if args.dry_run:
        print("\nDRY RUN - no files written")
        return 0

    for name, rows in ((OUT_TRAIN, new_train), (OUT_TEST, new_test)):
        path = DATA / name
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r["record"], ensure_ascii=False) + "\n")
        print(f"\nwrote {path}  ({len(rows)} records)")

    ids_path = DATA / "dl_split_80_20_task_ids.json"
    ids_path.write_text(json.dumps({
        "test_frac": args.test_frac, "seed": args.seed,
        "n_train": len(new_train), "n_test": len(new_test),
        "moved_train_to_test": sorted(move_ids),
        "train": sorted(ids_tr), "test": sorted(ids_te),
    }, indent=2))
    print(f"wrote {ids_path}")
    print(f"\noriginals untouched: {SRC_TRAIN}, {SRC_TEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
