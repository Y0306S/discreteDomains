#!/usr/bin/env python3
"""Build T2 / T3 attribution preference pairs from p5_s2_dpo.jsonl.

T2 (partial-edit):
  rejected = original bad config
  chosen   = bad config with *all* blamed HPs repaired toward good

T3 (soft per-HP via data):
  one pair per blamed HP (fix only that HP); duplicate ∝ blame strength
    (approx soft field weights without custom ORPO loss)

Blame:
  numeric  — search-space fraction (log distance iff log_scale=True)
  categorical — conservative history gate (n≥5, mean gap, prefer in top 25%
                and avoid not in top)
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

DEFAULT_SRC = Path(
    "/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/p5_s2_dpo.jsonl"
)
DEFAULT_DATA_DIR = Path("/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data")
DEFAULT_DATASET_INFO = Path(
    "/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/llamafactory_data/dataset_info.json"
)


def cfg_only(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k != "algorithm"}


def is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


SPACE_RE = re.compile(
    r"- ([^\n]+?): (categorical|float|integer), "
    r"(?:choices=(\[[^\]]+\])|range=\[([^\]]+)\])(?:, log_scale=(True|False))?"
)


def parse_search_space(prompt: str) -> dict[str, dict[str, Any]]:
    space: dict[str, dict[str, Any]] = {}
    for m in SPACE_RE.finditer(prompt or ""):
        name, typ, choices, rng, logsc = m.groups()
        rec: dict[str, Any] = {"type": typ, "log": logsc == "True"}
        if rng:
            lo, hi = [float(x.strip()) for x in rng.split(",")]
            rec["lo"], rec["hi"] = lo, hi
        if choices:
            rec["choices"] = choices
        space[name] = rec
    return space


def space_rel_diff(gv: Any, bv: Any, spec: dict[str, Any] | None) -> float:
    if not is_number(gv) or not is_number(bv):
        return 0.0
    g, b = float(gv), float(bv)
    if g == b:
        return 0.0
    if spec and spec.get("lo") is not None and spec.get("hi") is not None:
        lo, hi = float(spec["lo"]), float(spec["hi"])
        if spec.get("log") and g > 0 and b > 0 and lo > 0 and hi > lo:
            den = abs(math.log(hi) - math.log(lo)) or 1e-12
            return float(min(1.0, abs(math.log(g) - math.log(b)) / den))
        if hi > lo:
            return float(min(1.0, abs(g - b) / (hi - lo)))
    scale = max(abs(g), abs(b), abs(g - b), 1e-12)
    return float(min(1.0, abs(g - b) / scale))


def load_history_trials(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    trials: list[dict[str, Any]] = []
    for i, obs in enumerate(data.get("observations") or []):
        cfg = cfg_only(obs.get("config") or {})
        objs = obs.get("objectives") or [None]
        if objs[0] is None:
            continue
        trials.append({"i": i, "cfg": cfg, "score": -float(objs[0])})
    return trials


def _json_eq(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def cat_history_supported(
    trials: list[dict[str, Any]],
    hp: str,
    good_v: Any,
    bad_v: Any,
    *,
    min_n: int = 5,
    score_delta: float = 0.03,
    top_frac: float = 0.25,
) -> bool:
    """Conservative categorical gate: only blame if history is sure."""
    g_sc = [t["score"] for t in trials if _json_eq(t["cfg"].get(hp), good_v)]
    b_sc = [t["score"] for t in trials if _json_eq(t["cfg"].get(hp), bad_v)]
    if len(g_sc) < min_n or len(b_sc) < min_n:
        return False
    if (sum(g_sc) / len(g_sc)) - (sum(b_sc) / len(b_sc)) < score_delta:
        return False
    if len(trials) < 8:
        return False
    ordered = sorted(trials, key=lambda t: t["score"], reverse=True)
    ktop = max(2, int(math.ceil(top_frac * len(ordered))))
    top = ordered[:ktop]
    good_in_top = any(_json_eq(t["cfg"].get(hp), good_v) for t in top)
    bad_in_top = any(_json_eq(t["cfg"].get(hp), bad_v) for t in top)
    return bool(good_in_top and not bad_in_top)


def blame_hps(
    good: dict[str, Any],
    bad: dict[str, Any],
    *,
    rel_diff: float = 0.15,
    space: dict[str, dict[str, Any]] | None = None,
    trials: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Return hp -> blame strength in (0,1].

    Numerics: fraction of search-space span; log distance iff log_scale=True.
    Categoricals: blamed only if history gate is sure (n≥5, mean gap, prefer
    in top 25% and avoid not in top). Without history, categoricals are skipped.
    """
    g = cfg_only(good)
    b = cfg_only(bad)
    keys = sorted(set(g) & set(b))
    if not keys:
        return {}
    space = space or {}
    trials = trials or []

    blamed: dict[str, float] = {}
    for k in keys:
        gv, bv = g[k], b[k]
        if is_number(gv) and is_number(bv):
            rel = space_rel_diff(gv, bv, space.get(k))
            if rel >= rel_diff:
                blamed[k] = float(min(1.0, max(rel_diff, rel)))
        else:
            if _json_eq(gv, bv):
                continue
            if cat_history_supported(trials, k, gv, bv):
                blamed[k] = 1.0
    return blamed


def repair(bad: dict[str, Any], good: dict[str, Any], hps: list[str]) -> dict[str, Any]:
    out = dict(cfg_only(bad))
    g = cfg_only(good)
    for k in hps:
        if k in g:
            out[k] = g[k]
    return out


def format_answer(cfg: dict[str, Any], blamed: list[str], *, repaired: bool) -> str:
    names = ", ".join(blamed) if blamed else "key hyperparameters"
    if repaired:
        rat = (
            f"Repair blamed hyperparameter(s) ({names}) toward values associated with "
            f"better validation scores while keeping other hyperparameters unchanged."
        )
    else:
        rat = (
            f"This configuration keeps problematic value(s) on ({names}) that are "
            f"associated with worse validation scores."
        )
    body = json.dumps({"config": cfg}, ensure_ascii=True, sort_keys=True, default=str)
    return f"Rationale: {rat}\n\nJSON:\n{body}"


def sharegpt_row(
    conversations: list[dict[str, Any]],
    chosen_cfg: dict[str, Any],
    rejected_cfg: dict[str, Any],
    blamed: list[str],
    meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "conversations": conversations,
        "chosen": {"from": "gpt", "value": format_answer(chosen_cfg, blamed, repaired=True)},
        "rejected": {
            "from": "gpt",
            "value": format_answer(rejected_cfg, blamed, repaired=False),
        },
        "metadata": meta,
    }


def register_dataset(dataset_info: Path, dataset_id: str, rel_file: str) -> None:
    info = json.loads(dataset_info.read_text(encoding="utf-8"))
    info[dataset_id] = {
        "file_name": rel_file,
        "formatting": "sharegpt",
        "ranking": True,
        "columns": {
            "messages": "conversations",
            "chosen": "chosen",
            "rejected": "rejected",
        },
    }
    dataset_info.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--dataset-info", type=Path, default=DEFAULT_DATASET_INFO)
    ap.add_argument("--rel-diff", type=float, default=0.15)
    ap.add_argument("--max-blame-frac", type=float, default=0.5)
    ap.add_argument("--t3-upsample", type=float, default=3.0, help="max copies ∝ blame*this")
    args = ap.parse_args()

    t2_path = args.data_dir / "p5_attr_t2_train.jsonl"
    t3_path = args.data_dir / "p5_attr_t3_train.jsonl"
    t2_full = args.data_dir / "p5_attr_t2.jsonl"
    t3_full = args.data_dir / "p5_attr_t3.jsonl"

    n_src = n_skip_empty = n_skip_frac = 0
    t2_rows: list[dict] = []
    t3_rows: list[dict] = []

    with args.src.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            n_src += 1
            meta = row.get("metadata") or {}
            good = meta.get("chosen_config") or {}
            bad = meta.get("rejected_config") or {}
            if not good or not bad:
                n_skip_empty += 1
                continue
            prompt = ""
            for m in row.get("conversations") or []:
                if m.get("from") == "human":
                    prompt = m.get("value") or ""
                    break
            space = parse_search_space(prompt)
            trials_all = load_history_trials(meta.get("history_path"))
            imp = meta.get("improvement_trial")
            if isinstance(imp, int) and imp > 0:
                trials = [t for t in trials_all if t["i"] < imp]
                if len(trials) < 5:
                    trials = trials_all
            else:
                trials = trials_all
            blamed = blame_hps(
                good, bad, rel_diff=args.rel_diff, space=space, trials=trials
            )
            if not blamed:
                n_skip_empty += 1
                continue
            n_hp = len(cfg_only(bad)) or len(blamed)
            if len(blamed) / max(n_hp, 1) > args.max_blame_frac:
                n_skip_frac += 1
                continue

            conv = row["conversations"]
            blamed_names = sorted(blamed.keys())

            # T2: fix all blamed HPs at once
            chosen_t2 = repair(bad, good, blamed_names)
            rejected_t2 = cfg_only(bad)
            if chosen_t2 == rejected_t2:
                n_skip_empty += 1
                continue
            t2_meta = {
                **{k: meta[k] for k in meta if k not in ("chosen_config", "rejected_config")},
                "attribution": "t2_partial_edit",
                "blamed_hps": blamed_names,
                "blame_strength": blamed,
                "chosen_config": chosen_t2,
                "rejected_config": rejected_t2,
                "source_chosen_config": cfg_only(good),
                "source_rejected_config": cfg_only(bad),
            }
            t2_rows.append(
                sharegpt_row(conv, chosen_t2, rejected_t2, blamed_names, t2_meta)
            )

            # T3: one pair per blamed HP, upsample by strength
            for hp, strength in blamed.items():
                chosen_t3 = repair(bad, good, [hp])
                rejected_t3 = cfg_only(bad)
                if chosen_t3 == rejected_t3:
                    continue
                copies = max(1, min(int(args.t3_upsample), int(math.ceil(strength * args.t3_upsample))))
                t3_meta = {
                    **{k: meta[k] for k in meta if k not in ("chosen_config", "rejected_config")},
                    "attribution": "t3_per_hp_soft",
                    "blamed_hps": [hp],
                    "blame_strength": {hp: strength},
                    "upsample_copies": copies,
                    "chosen_config": chosen_t3,
                    "rejected_config": rejected_t3,
                    "source_chosen_config": cfg_only(good),
                    "source_rejected_config": cfg_only(bad),
                }
                one = sharegpt_row(conv, chosen_t3, rejected_t3, [hp], t3_meta)
                for _ in range(copies):
                    t3_rows.append(one)

    def _write(path: Path, rows: list[dict], strip_meta: bool) -> None:
        with path.open("w", encoding="utf-8") as out:
            for r in rows:
                if strip_meta:
                    out.write(
                        json.dumps(
                            {
                                "conversations": r["conversations"],
                                "chosen": r["chosen"],
                                "rejected": r["rejected"],
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )
                else:
                    out.write(json.dumps(r, ensure_ascii=True) + "\n")

    _write(t2_full, t2_rows, strip_meta=False)
    _write(t3_full, t3_rows, strip_meta=False)
    _write(t2_path, t2_rows, strip_meta=True)
    _write(t3_path, t3_rows, strip_meta=True)

    register_dataset(args.dataset_info, "p5_attr_t2", "../data/p5_attr_t2_train.jsonl")
    register_dataset(args.dataset_info, "p5_attr_t3", "../data/p5_attr_t3_train.jsonl")

    manifest = {
        "src": str(args.src),
        "n_src": n_src,
        "n_skip_empty_or_no_blame": n_skip_empty,
        "n_skip_blame_frac": n_skip_frac,
        "n_t2": len(t2_rows),
        "n_t3": len(t3_rows),
        "t2_train": str(t2_path),
        "t3_train": str(t3_path),
        "rel_diff": args.rel_diff,
        "max_blame_frac": args.max_blame_frac,
        "t3_upsample": args.t3_upsample,
    }
    (args.data_dir / "p5_attr_t2_t3.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
