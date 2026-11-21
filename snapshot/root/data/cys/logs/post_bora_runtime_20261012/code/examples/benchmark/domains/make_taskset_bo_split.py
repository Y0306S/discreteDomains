#!/usr/bin/env python3
"""Build a family-balanced TaskSet task-id split for BO runs.

Why this exists
---------------
`bolt_qwen9b_finetune/data/dl_train_85.jsonl` / `dl_test_15.jsonl` are LLM
fine-tuning datasets (chat `messages` records), not task-id lists: a record
carries `task_family` + `task` (the raw spec) but **no task_id**. BO runs need
task ids. This script recovers them by matching each record's `task` spec against
`task_set/tasks/configs/*.json`, which store `{task_id: [spec, family]}`.

That recovery matters for more than convenience: it is what lets a BO run stay on
the correct side of the LLM's train/test boundary. Warm-started BO
(`--optimizer ptllm_tpe`) evaluated on a task that was in `dl_train_85.jsonl` is
leakage, because the LLM was fine-tuned on that task's answer.

Coverage caveat (measured, see --report)
----------------------------------------
The SFT split covers exactly the 858 tasks of **9 of TaskSet's 12** sampled
families. `mlp_vae_family` (55), `nvp_family` (66) and
`word_rnn_language_model_family` (69) are absent entirely, as are the 109
`Fixed*`/one-off tasks. Those 190 family tasks are therefore a *clean*
generalization set: the LLM has never seen any of them. Use `--source heldout`.

Usage
-----
  # what's available, and the coverage report
  python make_taskset_bo_split.py --report

  # 5 tasks per family from the LLM's train side (BO development)
  python make_taskset_bo_split.py --source train --per-family 5

  # 5 per family from the LLM's test side (fair warm-start evaluation)
  python make_taskset_bo_split.py --source test --per-family 5 -o /tmp/bo_test.json

  # families the LLM never saw (strongest generalization test)
  python make_taskset_bo_split.py --source heldout --per-family 5

Feed the result to the BO runner:
  python run_taskset_bo.py --task-ids "$(python make_taskset_bo_split.py \
      --source test --per-family 3 --print-ids)" --no-dummy ...
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
from typing import Any, Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains.dlhpo.taskset_space import list_taskset_ids
from mindware.domains.paths import TASKSET_BASELINE_CACHE, TASKSET_EVAL_DIR

SFT_DATA_DIR = Path(
    "/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data"
)
TASK_CONFIGS = Path("/root/data/cys/google-research/task_set/tasks/configs")
SFT_TRAIN = "dl_train_85.jsonl"
SFT_TEST = "dl_test_15.jsonl"

FAMILY_RE = re.compile(r"^(.*?_family)_seed\d+$")


def family_of(task_id: str) -> str:
    m = FAMILY_RE.match(task_id)
    return m.group(1) if m else "fixed_or_other"


def _spec_hash(spec: Any) -> str:
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()


def _canonical_spec(spec: Any) -> Any:
    """Normalize a record's `task` spec to the shape used in the config files.

    losg tasks are the one family that differs: the configs store a positional
    list ``[problem, params, init_seed]`` while the SFT records store a dict
    ``{losg_problem, params, init_seed}``. Without this, all 100 losg tasks fail
    to match.
    """
    if isinstance(spec, dict) and "losg_problem" in spec:
        return [spec["losg_problem"], spec.get("params"), spec.get("init_seed")]
    return spec


def build_spec_index() -> Dict[str, List[str]]:
    if not TASK_CONFIGS.is_dir():
        raise SystemExit(f"TaskSet configs not found: {TASK_CONFIGS}")
    index: Dict[str, List[str]] = {}
    for path in sorted(TASK_CONFIGS.glob("*.json")):
        for task_id, value in json.loads(path.read_text()).items():
            spec = value[0] if isinstance(value, list) else value
            index.setdefault(_spec_hash(spec), []).append(task_id)
    return index


def recover_task_ids(jsonl: Path, index: Dict[str, List[str]]) -> Dict[str, Any]:
    """Map every SFT record in `jsonl` back to its TaskSet task_id."""
    if not jsonl.is_file():
        raise SystemExit(f"SFT split not found: {jsonl}")
    ids: List[str] = []
    unmatched: List[str] = []
    ambiguous: List[str] = []
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        user = json.loads(record["messages"][1]["content"])
        hits = index.get(_spec_hash(_canonical_spec(user["task"])), [])
        if not hits:
            unmatched.append(user.get("task_family", "?"))
        elif len(hits) > 1:
            ambiguous.append(hits[0])
            ids.append(hits[0])
        else:
            ids.append(hits[0])
    return {"ids": ids, "unmatched": unmatched, "ambiguous": ambiguous}


def usable_ids(candidates: Set[str]) -> Dict[str, Any]:
    """Keep only ids that are real TaskSet tasks AND have a paper best_val.

    A task with no best_val cannot be scored: `evaluate_taskset_config` returns
    objective=1e10, feasible=False, so it silently wastes a BO trial.
    """
    known = set(list_taskset_ids())
    not_a_task = sorted(candidates - known)

    eval_dir = str(TASKSET_EVAL_DIR)
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    from taskset_paper_scoring import get_task_best_val, load_baseline_cache  # type: ignore

    cache = load_baseline_cache(TASKSET_BASELINE_CACHE)
    no_baseline = sorted(
        t for t in (candidates & known) if get_task_best_val(t, cache) is None
    )
    return {
        "usable": sorted((candidates & known) - set(no_baseline)),
        "not_a_task": not_a_task,
        "no_baseline": no_baseline,
    }


def stratified_sample(ids: List[str], per_family: int, seed: int) -> Dict[str, List[str]]:
    """Take exactly `per_family` ids from each family (fewer only if a family is short)."""
    by_family: Dict[str, List[str]] = defaultdict(list)
    for t in ids:
        by_family[family_of(t)].append(t)
    rng = random.Random(seed)
    out: Dict[str, List[str]] = {}
    for fam in sorted(by_family):
        pool = sorted(by_family[fam])
        rng.shuffle(pool)
        out[fam] = sorted(pool[:per_family])
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--source",
        default="train",
        choices=["train", "test", "heldout"],
        help="train/test = the LLM's own split (recovered task ids). "
        "heldout = the 3 sampled families absent from the SFT data, which the LLM never saw.",
    )
    p.add_argument("--per-family", type=int, default=5, help="tasks per family (equal across families)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-o", "--output", type=Path, default=None, help="write the split as JSON")
    p.add_argument("--print-ids", action="store_true", help="print only a comma-separated id list (for --task-ids)")
    p.add_argument("--report", action="store_true", help="print the full validation/coverage report and exit")
    args = p.parse_args(argv)

    index = build_spec_index()
    train = recover_task_ids(SFT_DATA_DIR / SFT_TRAIN, index)
    test = recover_task_ids(SFT_DATA_DIR / SFT_TEST, index)
    train_ids, test_ids = set(train["ids"]), set(test["ids"])

    all_known = set(list_taskset_ids())
    sampled_families = {family_of(t) for t in all_known if family_of(t) != "fixed_or_other"}
    covered_families = {family_of(t) for t in (train_ids | test_ids)}
    heldout_families = sorted(sampled_families - covered_families)
    heldout_ids = {t for t in all_known if family_of(t) in heldout_families}

    if args.report:
        print("=== recovery of task ids from the SFT splits ===")
        for name, res in (("dl_train_85", train), ("dl_test_15", test)):
            print(f"  {name:<12} matched={len(res['ids']):>4}  unmatched={len(res['unmatched']):>3}  "
                  f"ambiguous={len(res['ambiguous']):>3}  unique={len(set(res['ids'])):>4}")
            if res["unmatched"]:
                print(f"     unmatched by family: {dict(Counter(res['unmatched']))}")
        print(f"  train/test task_id overlap: {len(train_ids & test_ids)}  (must be 0)")

        print("\n=== coverage vs the full TaskSet suite ===")
        print(f"  TaskSet tasks total          : {len(all_known)}")
        print(f"  sampled families             : {len(sampled_families)}")
        print(f"  families covered by the SFT  : {len(covered_families)}")
        print(f"  families NEVER in the SFT    : {len(heldout_families)} -> {heldout_families}")
        print(f"  tasks in those families      : {len(heldout_ids)}")
        n_fixed = sum(1 for t in all_known if family_of(t) == "fixed_or_other")
        print(f"  Fixed*/one-off tasks (also unrepresented): {n_fixed}")

        print("\n=== usability for the BO oracle (valid id + has best_val) ===")
        for name, s in (("train", train_ids), ("test", test_ids), ("heldout", heldout_ids)):
            u = usable_ids(s)
            print(f"  {name:<8} usable={len(u['usable']):>4}  not_a_task={len(u['not_a_task']):>3}  "
                  f"no_baseline={len(u['no_baseline']):>3}")
            if u["no_baseline"]:
                print(f"     no baseline by family: {dict(Counter(map(family_of, u['no_baseline'])))}")
        print("\n=== per-family counts ===")
        for name, s in (("train", train_ids), ("test", test_ids), ("heldout", heldout_ids)):
            print(f"  {name:<8} {dict(sorted(Counter(map(family_of, s)).items()))}")
        return 0

    pool = {"train": train_ids, "test": test_ids, "heldout": heldout_ids}[args.source]
    checked = usable_ids(pool)
    if checked["no_baseline"] or checked["not_a_task"]:
        print(f"# dropped {len(checked['no_baseline'])} without best_val, "
              f"{len(checked['not_a_task'])} not-a-task", file=sys.stderr)

    sample = stratified_sample(checked["usable"], args.per_family, args.seed)
    flat = sorted(t for ids in sample.values() for t in ids)

    # Guard the leakage boundary explicitly rather than trusting the caller.
    if args.source in {"test", "heldout"}:
        bad = sorted(set(flat) & train_ids)
        if bad:
            raise SystemExit(f"LEAKAGE: {len(bad)} sampled ids are in the LLM train split: {bad[:5]}")

    if args.print_ids:
        print(",".join(flat))
        return 0

    short = {f: len(v) for f, v in sample.items() if len(v) < args.per_family}
    payload = {
        "source": args.source,
        "per_family": args.per_family,
        "seed": args.seed,
        "n_families": len(sample),
        "n_tasks": len(flat),
        "families_short_of_quota": short,
        "by_family": sample,
        "task_ids": flat,
        "llm_train_overlap": len(set(flat) & train_ids),
        "llm_test_overlap": len(set(flat) & test_ids),
    }
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.output}")
    print(json.dumps({k: v for k, v in payload.items() if k != "by_family"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
