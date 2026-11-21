"""Query-plan cost oracles: dummy hash, optional Postgres EXPLAIN."""

from __future__ import annotations

import os
from typing import List, Optional


def dummy_plan_cost(order: List[str], aliases: List[str]) -> float:
    size = {a: 10.0 + (sum(ord(c) for c in a) % 50) for a in aliases}
    acc = 0.0
    running = 1.0
    for i, a in enumerate(order):
        running *= size.get(a, 20.0)
        acc += running * (0.15 + 0.02 * i)
    if order:
        acc += size.get(order[0], 20.0) * 8.0
    return float(acc)


def postgres_explain_cost(
    sql: str,
    hint: str,
    *,
    dsn: Optional[str] = None,
    timeout_ms: int = 30000,
    join_order: Optional[List[str]] = None,
) -> Optional[float]:
    dsn = dsn or os.environ.get("QUERYOPT_PG_DSN")
    if not dsn:
        return None
    try:
        import psycopg2  # type: ignore
    except ImportError:
        return None
    from mindware.domains.queryopt.sql_parse import reorder_from

    q = reorder_from(sql, join_order) if join_order else sql
    conn = psycopg2.connect(dsn)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {int(timeout_ms)}")
            cur.execute("SET join_collapse_limit = 1")
            cur.execute("SET from_collapse_limit = 1")
            cur.execute("SET geqo = off")
            try:
                cur.execute("LOAD 'pg_hint_plan'")
                cur.execute("SET pg_hint_plan.enable_hint = on")
            except Exception:
                pass
            cur.execute("EXPLAIN (FORMAT JSON) " + (hint or "") + q)
            row = cur.fetchone()
        payload = row[0] if row else None
        if isinstance(payload, list) and payload:
            plan = payload[0].get("Plan") or payload[0]
            cost = plan.get("Total Cost")
            return float(cost) if cost is not None else None
        return None
    finally:
        conn.close()
