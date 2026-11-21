#!/usr/bin/env python3
"""Query histories: CEB SQL + dummy-cost join-order sweeps (BAO LFS if present)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains.history import is_git_lfs_pointer, observation, write_openbox_history
from mindware.domains.paths import QUERY_ROOT, domain_history
from mindware.domains.queryopt.oracle import dummy_plan_cost
from mindware.domains.queryopt.sql_parse import default_join_order, parse_tables, repair_order
from mindware.domains.sequence_vae import train_sequence_vae

CEB_SQL_ROOT = QUERY_ROOT / "query_plan_optimization/workload/ceb-3k"
CEB_TASKS = QUERY_ROOT / "tasks/ceb_tasks.txt"
BAO_INIT = QUERY_ROOT / "query_plan_optimization/initialization_data/ceb_3k_bao_initialization.csv"


def _sql_for_task(task_id: str) -> Path | None:
    # CEB_8A94 → 8a/8a94.sql (folder lower, file lower without leading zeros issues)
    raw = task_id.replace("CEB_", "").replace("ceb_", "")
    # e.g. 8A94 → 8a / 8a94.sql
    folder = "".join(ch.lower() if ch.isalpha() else ch for ch in raw if ch.isalpha() or (ch.isdigit() and raw.index(ch) < 2))
    # Better: split letters prefix + rest
    i = 0
    while i < len(raw) and raw[i].isdigit():
        i += 1
    num, rest = raw[:i], raw[i:]
    letter = rest[:1].lower()
    suffix = rest[1:]
    folder = f"{num}{letter}"
    fname = f"{folder}{suffix}.sql"
    cand = CEB_SQL_ROOT / folder / fname
    if cand.is_file():
        return cand
    # glob fallback
    hits = list(CEB_SQL_ROOT.glob(f"**/{folder}*.sql"))
    # prefer exact suffix
    for h in hits:
        if h.stem.lower().endswith(suffix.lower()) or h.stem.lower() == f"{folder}{suffix}".lower():
            return h
    return hits[0] if hits else None


def _load_task_ids(limit: int) -> list[str]:
    if CEB_TASKS.is_file():
        ids = [ln.strip() for ln in CEB_TASKS.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return ids[:limit]
    return []


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=domain_history("query"))
    p.add_argument("--limit", type=int, default=24)
    p.add_argument("--n-fake", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if BAO_INIT.is_file() and is_git_lfs_pointer(BAO_INIT):
        print(f"BAO init is a Git LFS pointer (not materialized): {BAO_INIT}")
        print("Skipping BAO pairs. Run `git lfs pull` in the BOLT repo to enable.")
    elif BAO_INIT.is_file():
        print(f"BAO init present ({BAO_INIT.stat().st_size} bytes); dummy-cost still used unless Postgres DSN is set.")

    task_ids = _load_task_ids(args.limit)
    rng = np.random.RandomState(args.seed)
    join_seqs = []
    written = []
    for tid in task_ids:
        sql_path = _sql_for_task(tid)
        if sql_path is None:
            continue
        sql = sql_path.read_text(encoding="utf-8", errors="replace")
        pairs = parse_tables(sql)
        aliases = default_join_order(pairs)
        if len(aliases) < 2:
            continue
        tables = [t for t, _ in pairs]
        join_seqs.append(tables)
        rows = []
        # default order
        orders = [aliases]
        for _ in range(args.n_fake - 1):
            perm = list(aliases)
            rng.shuffle(perm)
            orders.append(perm)
        for order in orders:
            order = repair_order(order, aliases)
            cost = dummy_plan_cost(order, aliases)
            cfg = {
                "n_tables": len(aliases),
                "starts_with": order[0],
                "join_order": "|".join(order),
                "join_skeleton": "fact_first" if order and order[0] == aliases[0] else "probe_first",
            }
            rows.append(
                observation(
                    cfg,
                    float(cost),
                    extra={"source": "synthetic_join_tpe", "sql_path": str(sql_path)},
                )
            )
        dest = args.out / "cost" / tid / "synthetic_tpe.json"
        write_openbox_history(
            dest,
            task_id=tid,
            observations=rows,
            meta={"source": "query_dummy_cost", "sql_path": str(sql_path), "n_tables": len(aliases)},
        )
        written.append(dest)

    if join_seqs:
        vae_path = args.out / "vae/sequence_vae.pkl"
        alphabet = sorted({tok for seq in join_seqs for tok in seq})
        train_sequence_vae(join_seqs, alphabet, out_path=vae_path, max_len=16, steps=30, latent_dim=16)
        print(f"Wrote query VAE {vae_path}")

    print(f"Wrote {len(written)} query histories under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
