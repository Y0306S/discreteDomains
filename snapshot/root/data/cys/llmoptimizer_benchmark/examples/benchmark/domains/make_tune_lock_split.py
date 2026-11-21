#!/usr/bin/env python3
"""Frozen TUNE/LOCK split for a domain (hash-based, never fit on LOCK)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples" / "benchmark"))

from dataset_split import make_split  # noqa: E402
from mindware.domains.paths import domain_history  # noqa: E402


def _task_ids_from_histories(root: Path) -> list[str]:
    ids = []
    skip = {"tune_lock_split.json", "experience_bank.jsonl", "sequence_vae.meta.json", "library_pairs.json"}
    skip_stems = {"synthetic_tpe", "library_pairs"}
    skip_tasks = {"apexgo_gramneg", "apexgo_mean", "bolt_peptides"}
    for path in Path(root).rglob("*.json"):
        if path.name in skip or path.stem in skip_stems or "vae" in path.parts:
            continue
        if path.parent.name in skip_tasks:
            continue
        if path.parent.parent.name not in {"fitness", "cost", "val_loss"}:
            continue
        ids.append(path.parent.name)
    return sorted(set(ids))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=["protein", "query", "dl"], required=True)
    p.add_argument("--history-root", type=Path, default=None)
    p.add_argument("--n-lock", type=int, default=None)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)
    root = args.history_root or domain_history(args.domain)
    names = _task_ids_from_histories(root)
    if not names:
        raise SystemExit(f"No tasks under {root}")
    n_lock = args.n_lock
    if n_lock is None:
        n_lock = max(1, int(round(0.25 * len(names))))
        n_lock = min(n_lock, max(1, len(names) - 1))
    payload = make_split(names, n_lock=n_lock)
    payload["domain"] = args.domain
    out = args.output or (root / "tune_lock_split.json")
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}: n_total={payload['n_total']} tune={payload['n_tune']} lock={payload['n_lock']}")
    print("LOCK:", ", ".join(payload["lock"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
