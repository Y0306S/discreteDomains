#!/usr/bin/env python3
"""Convert CEB pickle query reps (learnedsystems/ceb) into BOLT-style SQL trees.

Expects extracted CEB-IMDb-full layout: <src>/<template>/<template><id>.pkl
Writes: <dst>/<template>/<stem>.sql
Also writes tasks/ceb_tasks.txt (CEB_<STEM_UPPER>).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


def _sql_from_obj(obj) -> str:
    if obj is None:
        return ""
    if isinstance(obj, (bytes, bytearray)):
        try:
            obj = obj.decode("utf-8")
        except Exception:
            obj = pickle.loads(obj)
    if isinstance(obj, str):
        return obj.strip()
    if isinstance(obj, dict):
        for key in ("sql", "SQL", "query", "query_str"):
            if key in obj and obj[key]:
                return _sql_from_obj(obj[key])
        nested = obj.get("subset") or obj.get("qrep") or obj.get("query_rep")
        if nested:
            return _sql_from_obj(nested)
        join_graph = obj.get("join_graph")
        if join_graph is not None and hasattr(join_graph, "graph"):
            sql = join_graph.graph.get("sql") if hasattr(join_graph.graph, "get") else None
            if sql:
                return str(sql).strip()
    return ""


def convert(src: Path, dst: Path) -> tuple[int, int]:
    n_ok = n_fail = 0
    dst.mkdir(parents=True, exist_ok=True)
    pkls = sorted(src.rglob("*.pkl"))
    if not pkls:
        # already SQL?
        sqls = list(src.rglob("*.sql"))
        for p in sqls:
            rel = p.relative_to(src)
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.resolve() != p.resolve():
                out.write_text(p.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
            n_ok += 1
        return n_ok, n_fail
    for p in pkls:
        try:
            with p.open("rb") as f:
                obj = pickle.load(f)
            sql = _sql_from_obj(obj)
        except Exception:
            sql = ""
        if not sql or "select" not in sql.lower():
            n_fail += 1
            continue
        rel = p.relative_to(src)
        out = dst / rel.with_suffix(".sql")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(sql.rstrip() + "\n", encoding="utf-8")
        n_ok += 1
    return n_ok, n_fail


def write_task_list(sql_root: Path, out: Path) -> int:
    ids = []
    for p in sorted(sql_root.rglob("*.sql")):
        ids.append("CEB_" + p.stem.upper())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
    return len(ids)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--tasks-out", type=Path, default=None)
    args = p.parse_args(argv)
    n_ok, n_fail = convert(args.src, args.dst)
    n_tasks = 0
    if args.tasks_out:
        n_tasks = write_task_list(args.dst, args.tasks_out)
    print(f"[convert_ceb] sql={n_ok} fail={n_fail} tasks={n_tasks} dst={args.dst}")
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
