#!/usr/bin/env python3
"""Measure how often P16 optimizer configs follow analysis-agent recommendations."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]


def _norm_hp(name: str) -> str:
    return name.strip().strip("`").strip()


def _norm_val(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.6g}"
    return str(val).strip().strip("`").strip('"\'')


def parse_primary_recommendation(analysis: str) -> tuple[str | None, str | None]:
    if not analysis:
        return None, None
    text = analysis.replace("**", "")
    sec4 = re.search(
        r"4\.\s*SPECIFIC RECOMMENDATION:\s*(.+?)(?:\n\s*5\.|\Z)",
        text,
        flags=re.I | re.S,
    )
    chunk = sec4.group(1) if sec4 else text
    m = re.search(
        r"Set\s+([^\s]+(?::[^\s]+)?)\s+to\s+([^\n\.]+)",
        chunk,
        flags=re.I,
    )
    if m:
        return _norm_hp(m.group(1)), _norm_val(m.group(2))
    m = re.search(
        r"Change\s+([^\s]+(?::[^\s]+)?)\s+(?:first|to)",
        text,
        flags=re.I,
    )
    if m:
        return _norm_hp(m.group(1)), None
    sec3 = re.search(
        r"3\.\s*PRIMARY ACTION:\s*(.+?)(?:\n\s*4\.|\Z)",
        text,
        flags=re.I | re.S,
    )
    if sec3:
        hp = re.search(r"([A-Za-z0-9_]+:[A-Za-z0-9_]+)", sec3.group(1))
        if hp:
            return _norm_hp(hp.group(1)), None
    return None, None


def parse_latest_config_from_tcs(user_prompt: str) -> dict[str, str]:
    m = re.search(
        r"=== LATEST EXPERIMENT \(trial \d+\) ===\nConfig: (\{.*?\})\n",
        user_prompt,
        flags=re.S,
    )
    if not m:
        return {}
    try:
        cfg = json.loads(m.group(1))
        return {k: _norm_val(v) for k, v in cfg.items()}
    except json.JSONDecodeError:
        return {}


def values_close(a: str, b: str, *, rtol: float = 0.15) -> bool:
    if a == b:
        return True
    try:
        fa, fb = float(a), float(b)
        if fa == fb:
            return True
        denom = max(abs(fa), abs(fb), 1e-9)
        return abs(fa - fb) / denom <= rtol
    except ValueError:
        return a.lower() == b.lower()


def direction_match(prev: str, new: str, rec: str) -> bool:
    """Heuristic: recommended numeric between prev and rec, or exact."""
    if values_close(new, rec):
        return True
    try:
        p, n, r = float(prev), float(new), float(rec)
    except ValueError:
        return _norm_val(new).lower() == _norm_val(rec).lower()
    if r < p and n <= p:
        return True
    if r > p and n >= p:
        return True
    return abs(n - r) < abs(p - r)


def load_trial(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def analyze_file(path: Path) -> dict | None:
    row = load_trial(path)
    if not row:
        return None
    inp = row.get("llm_input") or {}
    if not inp.get("tcs_analysis"):
        return None
    analysis = inp.get("analysis_text") or ""
    if not analysis.strip():
        return None
    attempts = inp.get("attempts") or []
    if not attempts:
        return None
    cfg = (attempts[0].get("parsed_payload") or {}).get("config") or {}
    if not cfg:
        return None

    primary_hp, primary_val = parse_primary_recommendation(analysis)
    prev_cfg = parse_latest_config_from_tcs(inp.get("user_prompt") or "")
    opt_cfg = {k: _norm_val(v) for k, v in cfg.items()}

    out = {
        "path": str(path),
        "trial": row.get("trial"),
        "primary_hp": primary_hp,
        "primary_val_rec": primary_val,
        "optimizer_primary_val": opt_cfg.get(primary_hp) if primary_hp else None,
        "prev_primary_val": prev_cfg.get(primary_hp) if primary_hp else None,
    }

    if primary_hp and primary_hp in opt_cfg:
        out["primary_hp_changed"] = (
            primary_hp not in prev_cfg or opt_cfg[primary_hp] != prev_cfg[primary_hp]
        )
        if primary_val is not None:
            out["primary_value_exact"] = values_close(opt_cfg[primary_hp], primary_val)
            if primary_hp in prev_cfg:
                out["primary_direction_ok"] = direction_match(
                    prev_cfg[primary_hp], opt_cfg[primary_hp], primary_val
                )
            else:
                out["primary_direction_ok"] = out["primary_value_exact"]
        else:
            out["primary_value_exact"] = None
            out["primary_direction_ok"] = out.get("primary_hp_changed")
    else:
        out["primary_hp_changed"] = None
        out["primary_value_exact"] = None
        out["primary_direction_ok"] = None

    changed_hps = [
        hp
        for hp in set(prev_cfg) | set(opt_cfg)
        if prev_cfg.get(hp) != opt_cfg.get(hp)
    ]
    out["n_changed_hps"] = len(changed_hps)
    out["changed_hps"] = changed_hps
    out["only_primary_changed"] = (
        primary_hp is not None and changed_hps == [primary_hp]
    )
    out["full_config_exact_rec"] = (
        primary_hp is not None
        and primary_val is not None
        and out.get("primary_value_exact")
        and out.get("only_primary_changed")
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        default=str(REPO / "benchmark_data" / "real_hpo_v2"),
    )
    parser.add_argument(
        "--variant",
        default="ptllm_tpe__P16__H2",
    )
    parser.add_argument(
        "--out-csv",
        default=str(
            REPO
            / "benchmark_data"
            / "real_hpo_v2"
            / "analysis"
            / "eval8"
            / "p16_analysis_optimizer_alignment.csv"
        ),
    )
    args = parser.parse_args()
    base = Path(args.base_dir)
    rows = []
    for path in sorted(base.rglob(f"**/{args.variant}/**/llm_conversations/trial_*.json")):
        rec = analyze_file(path)
        if rec:
            rows.append(rec)

    if not rows:
        print("[Warn] no TCS analysis trials found")
        return

    n = len(rows)
    parsed_primary = sum(1 for r in rows if r["primary_hp"])
    with_rec_val = sum(1 for r in rows if r["primary_val_rec"] is not None)
    exact = sum(1 for r in rows if r.get("primary_value_exact") is True)
    direction = sum(1 for r in rows if r.get("primary_direction_ok") is True)
    only_primary = sum(1 for r in rows if r.get("only_primary_changed"))
    full_exact = sum(1 for r in rows if r.get("full_config_exact_rec"))
    changed_primary = sum(1 for r in rows if r.get("primary_hp_changed") is True)

    by_dataset: dict[str, list] = defaultdict(list)
    for r in rows:
        parts = Path(r["path"]).parts
        try:
            idx = parts.index(args.variant.split("__")[0] if False else args.variant)
        except ValueError:
            idx = -1
        # path: .../cls/yeast/ptllm_tpe__P16__H2/...
        for i, p in enumerate(parts):
            if p in ("cls", "rgs") and i + 1 < len(parts):
                by_dataset[parts[i + 1]].append(r)
                break

    print(f"Variant: {args.variant}")
    print(f"TCS analysis trials analyzed: {n}")
    print(f"Parsed primary HP: {parsed_primary}/{n} ({100*parsed_primary/n:.1f}%)")
    print(f"Parsed primary value: {with_rec_val}/{n} ({100*with_rec_val/n:.1f}%)")
    print(f"Optimizer changed primary HP: {changed_primary}/{parsed_primary} ({100*changed_primary/max(parsed_primary,1):.1f}%)")
    print(f"Primary value EXACT match: {exact}/{with_rec_val} ({100*exact/max(with_rec_val,1):.1f}%)")
    print(f"Primary value direction/close match: {direction}/{with_rec_val} ({100*direction/max(with_rec_val,1):.1f}%)")
    print(f"Only primary HP changed (single-knob): {only_primary}/{n} ({100*only_primary/n:.1f}%)")
    print(f"Full config = exact single-knob rec: {full_exact}/{n} ({100*full_exact/n:.1f}%)")

    changed_counts = Counter(r["n_changed_hps"] for r in rows)
    print("\n# HPs changed vs previous trial:")
    for k in sorted(changed_counts):
        print(f"  {k} HPs changed: {changed_counts[k]} trials ({100*changed_counts[k]/n:.1f}%)")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    fields = [
        "path",
        "trial",
        "primary_hp",
        "primary_val_rec",
        "optimizer_primary_val",
        "prev_primary_val",
        "primary_hp_changed",
        "primary_value_exact",
        "primary_direction_ok",
        "n_changed_hps",
        "only_primary_changed",
        "full_config_exact_rec",
        "changed_hps",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["changed_hps"] = "|".join(r.get("changed_hps") or [])
            w.writerow(r)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
