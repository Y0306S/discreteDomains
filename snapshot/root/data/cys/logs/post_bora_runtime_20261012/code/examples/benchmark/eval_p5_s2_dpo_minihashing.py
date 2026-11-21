#!/usr/bin/env python3
"""Mini-eval of P5 S2 DPO pairs: Hamming distance + semantic + novel-diff checks.

Loads existing p5_s2_dpo.jsonl (with metadata), reports Hamming ≤1 / ≤2 / full,
spot-checks Hamming-1 and Hamming-2 examples, re-validates rejected configs
against BO history, and checks whether differing HP keys were already mentioned in:

  - P5 run-summary block (top vs worst HP hints in the human prompt)
  - P16 analysis_report (when available for the same improvement event)

Optionally writes filtered train files:
  - p5_s2_dpo_hamming_le2_train.jsonl       (Hamming ≤ 2)
  - p5_s2_dpo_hamming_le1_train.jsonl       (Hamming ≤ 1, if enough pairs)
  - p5_s2_dpo_hamming_le2_novel_train.jsonl (h≤2 + novel diff keys)

Does NOT start training.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BENCHMARK = Path(__file__).resolve().parent
REPO = BENCHMARK.parents[1]
sys.path.insert(0, str(BENCHMARK))

from p5_sft_common import (  # noqa: E402
    DEFAULT_OUT,
    TaskKey,
    config_for_optimizer_json,
    history_path,
    load_task_keys,
    perf_from_obs,
    successful_observations,
)

DEFAULT_DATA = Path("/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data")
DEFAULT_TASKS = DEFAULT_DATA / "ml_train_80_fixed.jsonl"
DEFAULT_P16 = DEFAULT_DATA / "p16_improvement_events_with_analysis.jsonl"
DEFAULT_DATASET_INFO = Path(
    "/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/llamafactory_data/dataset_info.json"
)

RUN_SUMMARY_HEADER = "Hyperparameters that differ most between top and worst trials:"
RUN_SUMMARY_HP_RE = re.compile(r"^\s*-\s+([^:]+):\s+top trials tend", re.MULTILINE)
SEARCH_SPACE_HP_RE = re.compile(r"^-\s+([^:]+):\s+", re.MULTILINE)


def hamming(a: dict, b: dict) -> int:
    keys = set(a) | set(b)
    return sum(1 for k in keys if a.get(k) != b.get(k))


def diff_keys(a: dict, b: dict) -> list[str]:
    keys = set(a) | set(b)
    return sorted(k for k in keys if a.get(k) != b.get(k))


def human_prompt(row: dict) -> str:
    for turn in row.get("conversations", []):
        if turn.get("from") == "human":
            return turn.get("value", "")
    return ""


def extract_search_space_hps(user_text: str) -> list[str]:
    block = user_text.split("Search space:", 1)
    if len(block) < 2:
        return []
    rest = block[1].split("\n\nPrevious validation trials", 1)[0]
    return [m.group(1).strip() for m in SEARCH_SPACE_HP_RE.finditer(rest)]


def extract_run_summary_hps(user_text: str) -> set[str]:
    if RUN_SUMMARY_HEADER not in user_text:
        return set()
    section = user_text.split(RUN_SUMMARY_HEADER, 1)[1]
    section = section.split("\n\nReturn only", 1)[0]
    return {m.group(1).strip() for m in RUN_SUMMARY_HP_RE.finditer(section)}


def extract_mentioned_hps_from_text(text: str, search_hps: list[str]) -> set[str]:
    if not text or not search_hps:
        return set()
    mentioned: set[str] = set()
    for hp in search_hps:
        if hp in text:
            mentioned.add(hp)
            continue
        short = hp.split(":", 1)[-1]
        if len(short) >= 4 and re.search(rf"(?<![\w:]){re.escape(short)}(?![\w:])", text):
            mentioned.add(hp)
    return mentioned


def load_p16_analysis(path: Path) -> dict[tuple[str, str, str, int], str]:
    out: dict[tuple[str, str, str, int], str] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            meta = row.get("metadata", {})
            key = (
                meta.get("task_type", ""),
                meta.get("dataset", ""),
                meta.get("algorithm", ""),
                int(meta.get("improvement_trial", -1)),
            )
            report = row.get("analysis_report") or ""
            if key[3] >= 0 and report:
                out[key] = report
    return out


@dataclass
class NovelCheck:
    run_summary_hps: set[str] = field(default_factory=set)
    analysis_hps: set[str] = field(default_factory=set)
    mentioned_hps: set[str] = field(default_factory=set)
    diff_key_set: set[str] = field(default_factory=set)
    novel_diff_keys: set[str] = field(default_factory=set)
    overlap_diff_keys: set[str] = field(default_factory=set)
    all_diffs_mentioned: bool = False
    zero_overlap: bool = False
    has_p16: bool = False


def novel_check_pair(
    row: dict,
    p16_analysis: dict[tuple[str, str, str, int], str],
) -> NovelCheck:
    meta = row.get("metadata", {})
    chosen = meta.get("chosen_config") or {}
    rejected = meta.get("rejected_config") or {}
    diffs = set(diff_keys(chosen, rejected))

    user = human_prompt(row)
    search_hps = extract_search_space_hps(user)
    run_hps = extract_run_summary_hps(user)

    akey = (
        meta.get("task_type", ""),
        meta.get("dataset", ""),
        meta.get("algorithm", ""),
        int(meta.get("improvement_trial", -1)),
    )
    analysis_text = p16_analysis.get(akey, "")
    analysis_hps = extract_mentioned_hps_from_text(analysis_text, search_hps)
    mentioned = set(run_hps) | set(analysis_hps)
    overlap = diffs & mentioned
    novel = diffs - mentioned

    return NovelCheck(
        run_summary_hps=run_hps,
        analysis_hps=analysis_hps,
        mentioned_hps=mentioned,
        diff_key_set=diffs,
        novel_diff_keys=novel,
        overlap_diff_keys=overlap,
        all_diffs_mentioned=bool(diffs) and diffs.issubset(mentioned),
        zero_overlap=not overlap,
        has_p16=bool(analysis_text),
    )


def format_novel_spot(idx: int, row: dict, nc: NovelCheck) -> str:
    meta = row.get("metadata", {})
    chosen = meta.get("chosen_config") or {}
    rejected = meta.get("rejected_config") or {}
    lines = [
        f"[{idx}] {meta.get('task_type')}/{meta.get('dataset')}/{meta.get('algorithm')} "
        f"trial={meta.get('improvement_trial')} hamming={meta.get('hamming')}",
        f"  run_summary_hps: {sorted(nc.run_summary_hps)}",
        f"  analysis_hps:    {sorted(nc.analysis_hps)} (p16={'yes' if nc.has_p16 else 'no'})",
        f"  diff_keys:       {sorted(nc.diff_key_set)}",
        f"  novel:           {sorted(nc.novel_diff_keys)}",
        f"  overlap:         {sorted(nc.overlap_diff_keys)}",
    ]
    for k in sorted(nc.diff_key_set):
        tag = "novel" if k in nc.novel_diff_keys else "mentioned"
        lines.append(f"    {k}: rejected={rejected.get(k)!r} -> chosen={chosen.get(k)!r} [{tag}]")
    return "\n".join(lines)


def cfg_key(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True, ensure_ascii=False)


def to_train_row(row: dict) -> dict:
    return {
        "conversations": row["conversations"],
        "chosen": row["chosen"],
        "rejected": row["rejected"],
    }


@dataclass
class ScoreCheck:
    found_in_prefix: bool
    rejected_score: float | None
    incumbent: float | None
    chosen_score: float | None
    rejected_le_incumbent: bool | None
    rejected_lt_chosen: bool | None
    notes: str = ""


def load_history_cache(
    tasks_jsonl: Path,
    benchmark_root: Path,
) -> dict[tuple[str, str, str], tuple[list[dict], list[float]]]:
    """Map (task_type, dataset, algorithm) -> (optimizer configs, perfs)."""
    cache: dict[tuple[str, str, str], tuple[list[dict], list[float]]] = {}
    tasks = load_task_keys(tasks_jsonl)
    for task in tasks:
        key = (task.task_type, task.dataset, task.algorithm)
        if key in cache:
            continue
        hpath = history_path(benchmark_root, task)
        if not hpath.is_file():
            continue
        history = json.loads(hpath.read_text(encoding="utf-8"))
        obs = successful_observations(history)
        configs = [config_for_optimizer_json(o["config"]) for o in obs]
        perfs = [perf_from_obs(o) for o in obs]
        cache[key] = (configs, perfs)
    return cache


def score_check_pair(
    meta: dict,
    hist: dict[tuple[str, str, str], tuple[list[dict], list[float]]],
) -> ScoreCheck:
    key = (meta.get("task_type", ""), meta.get("dataset", ""), meta.get("algorithm", ""))
    if key not in hist:
        return ScoreCheck(False, None, None, None, None, None, notes="history_missing")

    configs, perfs = hist[key]
    trial = int(meta.get("improvement_trial", -1))  # 1-based in export metadata
    if trial < 1 or trial > len(configs):
        return ScoreCheck(False, None, None, None, None, None, notes=f"bad_trial={trial}")

    t = trial - 1  # 0-based index of chosen / improvement
    prefix_cfgs, prefix_perfs = configs[:t], perfs[:t]
    if not prefix_cfgs:
        return ScoreCheck(False, None, None, None, None, None, notes="empty_prefix")

    chosen = meta.get("chosen_config") or {}
    rejected = meta.get("rejected_config") or {}
    chosen_score = perfs[t]
    incumbent = max(prefix_perfs)

    # Recompute Hamming sanity
    # Find rejected in prefix (exact match on optimizer-json config)
    rej_k = cfg_key(rejected)
    matches = [i for i, c in enumerate(prefix_cfgs) if cfg_key(c) == rej_k]
    if not matches:
        return ScoreCheck(
            False,
            None,
            incumbent,
            chosen_score,
            None,
            None,
            notes="rejected_not_in_prefix",
        )

    # If multiple identical configs, take the one used as near-miss intent:
    # any match with score <= incumbent (all should be).
    rejected_score = prefix_perfs[matches[0]]
    # Prefer a match that is non-improving if several exist
    for i in matches:
        if prefix_perfs[i] <= incumbent + 1e-12:
            rejected_score = prefix_perfs[i]
            break

    le_inc = rejected_score <= incumbent + 1e-12
    lt_ch = rejected_score < chosen_score - 1e-12
    notes = "ok"
    if not le_inc:
        notes = "rejected_gt_incumbent"
    elif not lt_ch:
        notes = "rejected_ge_chosen"
    return ScoreCheck(True, rejected_score, incumbent, chosen_score, le_inc, lt_ch, notes)


def format_spot(pe_idx: int, row: dict, sc: ScoreCheck) -> str:
    meta = row.get("metadata", {})
    chosen = meta.get("chosen_config") or {}
    rejected = meta.get("rejected_config") or {}
    diffs = diff_keys(chosen, rejected)
    lines = [
        f"[{pe_idx}] {meta.get('task_type')}/{meta.get('dataset')}/{meta.get('algorithm')} "
        f"trial={meta.get('improvement_trial')} hamming={meta.get('hamming')} "
        f"delta_rel={meta.get('delta_rel_score')}",
        f"  differing keys ({len(diffs)}): {diffs}",
    ]
    for k in diffs:
        lines.append(f"    {k}: rejected={rejected.get(k)!r}  chosen={chosen.get(k)!r}")
    lines.append(
        f"  scores: rejected={sc.rejected_score}  incumbent={sc.incumbent}  "
        f"chosen={sc.chosen_score}  "
        f"in_prefix={sc.found_in_prefix}  le_inc={sc.rejected_le_incumbent}  "
        f"lt_chosen={sc.rejected_lt_chosen}  ({sc.notes})"
    )
    return "\n".join(lines)


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100.0 * n / d:.1f}%)" if d else "0/0"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_train_row(row), ensure_ascii=False) + "\n")


def register_dataset(dataset_info: Path, name: str, file_name: str) -> None:
    info = json.loads(dataset_info.read_text(encoding="utf-8"))
    info[name] = {
        "file_name": file_name,
        "ranking": True,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "chosen": "chosen",
            "rejected": "rejected",
        },
    }
    dataset_info.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpo", type=Path, default=DEFAULT_DATA / "p5_s2_dpo.jsonl")
    parser.add_argument("--tasks-jsonl", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--benchmark-root", type=Path, default=REPO)
    parser.add_argument(
        "--out-le2",
        type=Path,
        default=DEFAULT_DATA / "p5_s2_dpo_hamming_le2_train.jsonl",
    )
    parser.add_argument(
        "--out-le1",
        type=Path,
        default=DEFAULT_DATA / "p5_s2_dpo_hamming_le1_train.jsonl",
    )
    parser.add_argument(
        "--out-le2-novel",
        type=Path,
        default=DEFAULT_DATA / "p5_s2_dpo_hamming_le2_novel_train.jsonl",
    )
    parser.add_argument("--p16-analysis", type=Path, default=DEFAULT_P16)
    parser.add_argument("--min-train-count", type=int, default=100)
    parser.add_argument("--spot-check", type=int, default=4)
    parser.add_argument("--register-dataset-info", type=Path, default=DEFAULT_DATASET_INFO)
    parser.add_argument("--no-write", action="store_true", help="Stats/spot-check only")
    args = parser.parse_args()

    print("Loading BO history cache for score validation...")
    hist = load_history_cache(args.tasks_jsonl, args.benchmark_root)
    print(f"  history tasks cached: {len(hist)}")

    p16 = load_p16_analysis(args.p16_analysis)
    print(f"Loading P16 analysis: {args.p16_analysis} ({len(p16)} rows)")

    rows: list[dict] = []
    checks: list[ScoreCheck] = []
    novel_checks: list[NovelCheck] = []
    with args.dpo.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            checks.append(score_check_pair(row.get("metadata", {}), hist))
            novel_checks.append(novel_check_pair(row, p16))

    total = len(rows)
    hammings = [int(r["metadata"]["hamming"]) for r in rows]
    hdist = Counter(hammings)
    idx_le1 = [i for i, h in enumerate(hammings) if h <= 1]
    idx_le2 = [i for i, h in enumerate(hammings) if h <= 2]
    idx_h1 = [i for i, h in enumerate(hammings) if h == 1]
    idx_h2 = [i for i, h in enumerate(hammings) if h == 2]

    # Semantic validation aggregates
    n_in_prefix = sum(1 for c in checks if c.found_in_prefix)
    n_le_inc = sum(1 for c in checks if c.rejected_le_incumbent is True)
    n_lt_ch = sum(1 for c in checks if c.rejected_lt_chosen is True)
    n_all_ok = sum(
        1
        for c in checks
        if c.found_in_prefix and c.rejected_le_incumbent and c.rejected_lt_chosen
    )
    fail_notes = Counter(c.notes for c in checks if c.notes != "ok")

    # Among h<=2
    le2_ok = sum(
        1
        for i in idx_le2
        if checks[i].found_in_prefix
        and checks[i].rejected_le_incumbent
        and checks[i].rejected_lt_chosen
    )

    print("=" * 72)
    print("P5 S2 DPO Hamming mini-eval")
    print("=" * 72)
    print(f"Input: {args.dpo}")
    print(f"Total pairs: {total}")
    print(f"Hamming mean/min/max: {sum(hammings)/total:.3f} / {min(hammings)} / {max(hammings)}")
    print()
    print("Hamming distance distribution:")
    for d in sorted(hdist):
        print(f"  h={d}: {hdist[d]}")
    print()
    print("Counts:")
    print(f"  full set: {total}")
    print(f"  h<=1:     {pct(len(idx_le1), total)}")
    print(f"  h<=2:     {pct(len(idx_le2), total)}")
    print(f"  h==1:     {pct(len(idx_h1), total)}")
    print(f"  h==2:     {pct(len(idx_h2), total)}")
    print()
    print("Semantic validation vs BO history (all pairs):")
    print(f"  rejected in prefix:     {pct(n_in_prefix, total)}")
    print(f"  rejected ≤ incumbent:   {pct(n_le_inc, total)}")
    print(f"  rejected < chosen:      {pct(n_lt_ch, total)}")
    print(f"  all three checks pass:  {pct(n_all_ok, total)}")
    print(f"  among h<=2, all pass:   {pct(le2_ok, len(idx_le2))}")
    if fail_notes:
        print(f"  failure notes: {dict(fail_notes)}")
    print()

    p16_hits = sum(1 for nc in novel_checks if nc.has_p16)
    idx_le2_novel = [i for i in idx_le2 if novel_checks[i].novel_diff_keys]
    idx_le2_zero = [i for i in idx_le2 if novel_checks[i].zero_overlap]
    idx_le2_all_mentioned = [i for i in idx_le2 if novel_checks[i].all_diffs_mentioned]

    print("Novel diff-key analysis (run-summary ∪ P16 analysis):")
    print(f"  P16 matched: {p16_hits}/{total} pairs")
    print(f"  among h<=2, all diff keys mentioned (BAD):     {pct(len(idx_le2_all_mentioned), len(idx_le2))}")
    print(f"  among h<=2, >=1 novel diff key (prefer):       {pct(len(idx_le2_novel), len(idx_le2))}")
    print(f"  among h<=2, zero overlap / all novel (strict): {pct(len(idx_le2_zero), len(idx_le2))}")
    print()

    print(f"--- Good examples (h<=2, >=1 novel diff), up to {args.spot_check} ---")
    good_pool = sorted(
        idx_le2_novel,
        key=lambda i: (hammings[i], -len(novel_checks[i].novel_diff_keys), i),
    )
    for i in good_pool[: args.spot_check]:
        print(format_novel_spot(i, rows[i], novel_checks[i]))
        print()

    print(f"--- Bad examples (h<=2, all diffs mentioned), up to {args.spot_check} ---")
    bad_pool = sorted(idx_le2_all_mentioned, key=lambda i: (hammings[i], i))
    for i in bad_pool[: args.spot_check]:
        print(format_novel_spot(i, rows[i], novel_checks[i]))
        print()

    # Spot-checks: prefer semantically OK examples
    def pick_spots(indices: list[int], n: int) -> list[int]:
        ok = [i for i in indices if checks[i].notes == "ok"]
        pool = ok if ok else indices
        # diversify by algorithm
        by_algo: dict[str, list[int]] = defaultdict(list)
        for i in pool:
            by_algo[str(rows[i]["metadata"].get("algorithm"))].append(i)
        picked: list[int] = []
        algos = sorted(by_algo)
        ai = 0
        while len(picked) < min(n, len(pool)) and algos:
            algo = algos[ai % len(algos)]
            if by_algo[algo]:
                picked.append(by_algo[algo].pop(0))
            else:
                algos = [a for a in algos if by_algo[a]]
                if not algos:
                    break
                continue
            ai += 1
        return picked

    print(f"--- Spot-check Hamming=1 (up to {args.spot_check}) ---")
    for i in pick_spots(idx_h1, args.spot_check):
        print(format_spot(i, rows[i], checks[i]))
        print()

    print(f"--- Spot-check Hamming=2 (up to {args.spot_check}) ---")
    for i in pick_spots(idx_h2, args.spot_check):
        print(format_spot(i, rows[i], checks[i]))
        print()

    # Diff-key frequency among h<=2 (are single-HP edits meaningful?)
    key_freq: Counter[str] = Counter()
    for i in idx_le2:
        for k in diff_keys(
            rows[i]["metadata"].get("chosen_config") or {},
            rows[i]["metadata"].get("rejected_config") or {},
        ):
            key_freq[k.split(":")[-1]] += 1
    print("Top differing HP short-names among h<=2:")
    for name, cnt in key_freq.most_common(12):
        print(f"  {name}: {cnt}")
    print()

    # Write filtered files if clean
    semantic_clean = n_all_ok >= 0.98 * total and le2_ok >= 0.98 * len(idx_le2)
    wrote: list[str] = []

    if args.no_write:
        print("Skipped writing (--no-write).")
    elif not semantic_clean:
        print(
            f"Skipped writing filtered files: semantic checks not clean enough "
            f"(all_ok={n_all_ok}/{total}, le2_ok={le2_ok}/{len(idx_le2)})."
        )
    else:
        if len(idx_le2) >= args.min_train_count:
            le2_rows = [rows[i] for i in idx_le2]
            write_jsonl(args.out_le2, le2_rows)
            man = {
                "source": str(args.dpo),
                "filter": "hamming<=2",
                "pairs": len(le2_rows),
                "total": total,
                "out": str(args.out_le2),
                "semantic_all_ok": n_all_ok,
                "semantic_le2_ok": le2_ok,
            }
            args.out_le2.with_suffix(".manifest.json").write_text(
                json.dumps(man, indent=2) + "\n", encoding="utf-8"
            )
            wrote.append(str(args.out_le2))
            if args.register_dataset_info and args.register_dataset_info.is_file():
                register_dataset(
                    args.register_dataset_info,
                    "p5_s2_dpo_hamming_le2",
                    "../data/p5_s2_dpo_hamming_le2_train.jsonl",
                )
                print(f"Registered dataset p5_s2_dpo_hamming_le2 in {args.register_dataset_info}")
            print(f"Wrote {args.out_le2} ({len(le2_rows)} pairs)")
        else:
            print(f"Skipped le2 write: {len(idx_le2)} < min {args.min_train_count}")

        if len(idx_le1) >= args.min_train_count:
            le1_rows = [rows[i] for i in idx_le1]
            write_jsonl(args.out_le1, le1_rows)
            man = {
                "source": str(args.dpo),
                "filter": "hamming<=1",
                "pairs": len(le1_rows),
                "total": total,
                "out": str(args.out_le1),
            }
            args.out_le1.with_suffix(".manifest.json").write_text(
                json.dumps(man, indent=2) + "\n", encoding="utf-8"
            )
            wrote.append(str(args.out_le1))
            if args.register_dataset_info and args.register_dataset_info.is_file():
                register_dataset(
                    args.register_dataset_info,
                    "p5_s2_dpo_hamming_le1",
                    "../data/p5_s2_dpo_hamming_le1_train.jsonl",
                )
                print(f"Registered dataset p5_s2_dpo_hamming_le1 in {args.register_dataset_info}")
            print(f"Wrote {args.out_le1} ({len(le1_rows)} pairs)")
        else:
            print(
                f"Skipped le1 write: only {len(idx_le1)} pairs "
                f"(need >= {args.min_train_count} for a dedicated train file)"
            )

        # Novel-filtered h<=2: prefer strict zero-overlap if enough pairs
        use_strict = len(idx_le2_zero) >= args.min_train_count
        novel_indices = idx_le2_zero if use_strict else idx_le2_novel
        novel_filter = "h<=2, zero-overlap" if use_strict else "h<=2, >=1 novel diff"
        if len(novel_indices) >= args.min_train_count:
            novel_rows = [rows[i] for i in novel_indices]
            write_jsonl(args.out_le2_novel, novel_rows)
            man = {
                "source": str(args.dpo),
                "filter": novel_filter,
                "pairs": len(novel_rows),
                "total": total,
                "hamming_le2": len(idx_le2),
                "novel_le2": len(idx_le2_novel),
                "zero_overlap_le2": len(idx_le2_zero),
                "out": str(args.out_le2_novel),
            }
            args.out_le2_novel.with_suffix(".manifest.json").write_text(
                json.dumps(man, indent=2) + "\n", encoding="utf-8"
            )
            wrote.append(str(args.out_le2_novel))
            if args.register_dataset_info and args.register_dataset_info.is_file():
                register_dataset(
                    args.register_dataset_info,
                    "p5_s2_dpo_hamming_le2_novel",
                    "../data/p5_s2_dpo_hamming_le2_novel_train.jsonl",
                )
                print(
                    f"Registered dataset p5_s2_dpo_hamming_le2_novel in {args.register_dataset_info}"
                )
            print(f"Wrote {args.out_le2_novel} ({len(novel_rows)} pairs, filter={novel_filter})")
        else:
            print(
                f"Skipped novel write: {len(novel_indices)} pairs < min {args.min_train_count} "
                f"(filter={novel_filter})"
            )

    print()
    print("Recommendation:")
    frac_le2 = len(idx_le2) / total if total else 0
    frac_novel = len(idx_le2_novel) / total if total else 0
    if frac_le2 >= 0.25 and semantic_clean:
        print(
            f"  Hamming ≤2 retains {len(idx_le2)} pairs ({100*frac_le2:.1f}% of {total}); "
            f"rejected near-misses are semantically clean (prefix / ≤incumbent / <chosen)."
        )
        if len(idx_le2_novel) >= args.min_train_count:
            print(
                f"  Prefer h≤2 + novel diff filter: {len(idx_le2_novel)} pairs "
                f"({100*frac_novel:.1f}% of all) teach preferences beyond run-summary/analysis hints."
            )
            if len(idx_le2_zero) >= args.min_train_count:
                print(
                    f"  Stricter zero-overlap filter keeps {len(idx_le2_zero)} h≤2 pairs "
                    f"({100*len(idx_le2_zero)/max(len(idx_le2),1):.1f}% of h≤2)."
                )
        else:
            print(
                f"  Novel diff filter is thin ({len(idx_le2_novel)} pairs); "
                f"use h≤2 alone ({len(idx_le2)} pairs) for volume."
            )
        if len(idx_le1) < args.min_train_count:
            print(
                f"  Hamming ≤1 alone is thin ({len(idx_le1)} pairs) — diagnostic subset only."
            )
    elif frac_le2 >= 0.25:
        print(
            f"  Hamming ≤2 retains {len(idx_le2)} pairs, but semantic validation "
            f"had issues ({n_all_ok}/{total} fully OK) — inspect failure notes before filtering."
        )
    else:
        print(
            f"  Only {len(idx_le2)} h≤2 pairs ({100*frac_le2:.1f}%); prefer keeping all "
            f"or re-exporting with a hard Hamming cap in pick_near_miss."
        )
    if wrote:
        print("  Wrote: " + ", ".join(wrote))
    print("  Do NOT start S2 DPO training from this script.")


if __name__ == "__main__":
    main()
