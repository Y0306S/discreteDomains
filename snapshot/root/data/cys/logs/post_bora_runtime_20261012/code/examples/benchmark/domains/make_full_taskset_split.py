#!/usr/bin/env python3
"""Extend the 80/20 split to cover ALL scoreable TaskSet tasks, not just the 858
the SFT jsonl happens to cover.

The SFT files (dl_train_85/dl_test_15, and the dl_train_80/dl_test_20 derived from
them) only contain records for 9 of TaskSet's 12 sampled families -- 858 tasks. So
an 80/20 over those gives 686 train, when 80% of the full 1157-task suite is 926.

The 299 uncovered tasks are mlp_vae_family (55), nvp_family (66),
word_rnn_language_model_family (69) and the Fixed*/one-off tasks (109). All 299
have a paper best_val in the baseline cache, so all are scoreable -- there is no
technical reason to exclude them from a BO corpus.

This is additive and order-preserving:
  * every task already assigned by the SFT-derived split keeps its side, so no task
    ever moves test -> train (which would leak evaluation tasks into training);
  * the 299 new tasks are assigned 80/20 within their own group, so each group is
    independently balanced.

Note the SFT records and the BO task split are now different sets: the 299 new
tasks have no SFT record. That is fine for generating BO trajectories (only task
ids are needed) but means the older winner-only SFT corpus covers a subset.

Usage:
  make_full_taskset_split.py --dry-run
  make_full_taskset_split.py -o /path/to/full_split.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains.dlhpo.taskset_space import list_taskset_ids
from mindware.domains.paths import TASKSET_BASELINE_CACHE, TASKSET_EVAL_DIR

SFT_SPLIT = Path(
    "/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/dl_split_80_20_task_ids.json"
)
DEFAULT_OUT = Path(
    "/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/taskset_full_split_80_20.json"
)
FAMILY_RE = re.compile(r"^(.*?_family)_seed\d+$")


def group_of(task_id: str) -> str:
    m = FAMILY_RE.match(task_id)
    return m.group(1) if m else "fixed_or_other"


def scoreable(task_ids):
    eval_dir = str(TASKSET_EVAL_DIR)
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    from taskset_paper_scoring import get_task_best_val, load_baseline_cache  # type: ignore

    cache = load_baseline_cache(TASKSET_BASELINE_CACHE)
    ok, missing = [], []
    for t in task_ids:
        (ok if get_task_best_val(t, cache) is not None else missing).append(t)
    return ok, missing


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-frac", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    all_ids = list_taskset_ids()
    ok, missing = scoreable(all_ids)
    if missing:
        print(f"  ! {len(missing)} tasks have no best_val and are excluded: {missing[:3]}", file=sys.stderr)

    prior = json.loads(SFT_SPLIT.read_text())
    prior_train, prior_test = set(prior["train"]), set(prior["test"])
    assigned = prior_train | prior_test

    new_tasks = [t for t in ok if t not in assigned]
    rng = random.Random(args.seed)

    train, test = set(prior_train), set(prior_test)
    added = defaultdict(lambda: [0, 0])
    by_group = defaultdict(list)
    for t in new_tasks:
        by_group[group_of(t)].append(t)
    for g in sorted(by_group):
        pool = sorted(by_group[g])
        rng.shuffle(pool)
        n_test = round(len(pool) * args.test_frac)
        test.update(pool[:n_test])
        train.update(pool[n_test:])
        added[g] = [len(pool) - n_test, n_test]

    print("=== inherited from the SFT-derived split (unchanged) ===")
    print(f"  train {len(prior_train)}   test {len(prior_test)}   total {len(assigned)}")
    print(f"\n=== newly assigned ({len(new_tasks)} tasks that had no side) ===")
    print(f"  {'group':<34} {'train':>6} {'test':>5}")
    for g in sorted(added):
        print(f"  {g:<34} {added[g][0]:>6} {added[g][1]:>5}")

    print(f"\n=== FULL split ===")
    print(f"  {'group':<34} {'train':>6} {'test':>5} {'test%':>7}")
    gt, ge = Counter(map(group_of, train)), Counter(map(group_of, test))
    for g in sorted(set(gt) | set(ge)):
        tot = gt[g] + ge[g]
        print(f"  {g:<34} {gt[g]:>6} {ge[g]:>5} {100*ge[g]/tot if tot else 0:>6.1f}%")
    tot = len(train) + len(test)
    print(f"  {'TOTAL':<34} {len(train):>6} {len(test):>5} {100*len(test)/tot:>6.1f}%")

    # integrity
    assert not (train & test), f"overlap: {sorted(train & test)[:5]}"
    assert prior_test <= test, "a prior TEST task leaked into train"
    assert prior_train <= train, "a prior TRAIN task moved to test"
    print(f"\n  integrity: overlap=0   prior test preserved={prior_test <= test}   "
          f"prior train preserved={prior_train <= train}   total={tot}/{len(ok)}")

    if args.dry_run:
        print("\nDRY RUN - nothing written")
        return 0

    args.output.write_text(json.dumps({
        "test_frac": args.test_frac,
        "seed": args.seed,
        "source": "extends dl_split_80_20_task_ids.json to all scoreable TaskSet tasks",
        "n_train": len(train),
        "n_test": len(test),
        "n_inherited": len(assigned),
        "n_newly_assigned": len(new_tasks),
        "train": sorted(train),
        "test": sorted(test),
    }, indent=2))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
