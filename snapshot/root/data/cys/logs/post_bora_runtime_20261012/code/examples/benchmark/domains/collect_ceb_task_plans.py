#!/usr/bin/env python3
"""Build ceb_task_plans.csv: connected join encodings for all CEB SQLs.

Fast path (default): no Postgres — default FROM order + random connected perms.
Probe path: also EXPLAIN-score extra perms for selected tasks.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains.bolt_vae import left_deep_flat
from mindware.domains.queryopt.ceb import IMDB_TABLES, load_sql, load_task_ids
from mindware.domains.queryopt.sql_parse import (
    alias_join_graph,
    parse_tables,
    random_connected_order,
)

PROBE_TASKS = [
    "CEB_10A35",
    "CEB_10A9",
    "CEB_1A1229",
    "CEB_1A2517",
    "CEB_11A102",
    "CEB_2B118",
    "CEB_2A412",
    "CEB_8A94",
    "CEB_8A121",
    "CEB_6A209",
    "CEB_7A119",
    "CEB_7A10",
]


def _flat_for_order(pairs, order, table_to_id):
    alias_to_table = {a: t for t, a in pairs}
    ids = [table_to_id.get(alias_to_table.get(a, a), 0) for a in order]
    if len(ids) < 2:
        return None
    return left_deep_flat(ids)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/root/data/cys/BOLT/optimization/query_plans/"
            "query_plan_optimization/training_data/encoded_ceb/ceb_task_plans.csv"
        ),
    )
    p.add_argument("--n-perm", type=int, default=20)
    p.add_argument("--probe-n-perm", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="0 = all CEB task ids")
    p.add_argument("--explain-probes", action="store_true")
    args = p.parse_args(argv)

    table_to_id = {t: i for i, t in enumerate(IMDB_TABLES)}
    rng = random.Random(args.seed)
    ids = load_task_ids(limit=args.limit or None)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        for tid in ids:
            sql = load_sql(tid)
            if not sql:
                continue
            pairs = parse_tables(sql)
            aliases = [a for _, a in pairs]
            if len(aliases) < 2:
                continue
            graph = alias_join_graph(sql, aliases)
            n_perm = args.probe_n_perm if tid in PROBE_TASKS else args.n_perm
            seen = set()
            orders = [aliases]
            for _ in range(n_perm):
                orders.append(random_connected_order(aliases, graph, rng))
            for i, order in enumerate(orders):
                key = tuple(order)
                if key in seen:
                    continue
                seen.add(key)
                flat = _flat_for_order(pairs, order, table_to_id)
                if not flat:
                    continue
                kind = "default" if i == 0 else "connected"
                w.writerow((tid, kind, ",".join(str(x) for x in flat)))
                n_rows += 1

    print(f"[collect] wrote {n_rows} rows -> {args.out} tasks={len(ids)}")

    if args.explain_probes:
        dsn = os.environ.get(
            "QUERYOPT_PG_DSN",
            "host=127.0.0.1 port=5432 dbname=imdb user=imdb password=imdb",
        )
        from mindware.domains.queryopt.oracle import postgres_explain_cost
        from mindware.domains.queryopt.sql_parse import pg_leading_hint

        probe_out = args.out.with_name("ceb_probe_explained.csv")
        n_ok = 0
        with probe_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(("task_id", "kind", "cost", "order", "flat"))
            for tid in PROBE_TASKS:
                sql = load_sql(tid)
                pairs = parse_tables(sql)
                aliases = [a for _, a in pairs]
                graph = alias_join_graph(sql, aliases)
                orders = [aliases]
                for _ in range(args.probe_n_perm):
                    orders.append(random_connected_order(aliases, graph, rng))
                seen = set()
                for i, order in enumerate(orders):
                    key = tuple(order)
                    if key in seen:
                        continue
                    seen.add(key)
                    cost = postgres_explain_cost(
                        sql, pg_leading_hint(order), dsn=dsn, join_order=order
                    )
                    flat = _flat_for_order(pairs, order, table_to_id)
                    w.writerow(
                        (
                            tid,
                            "default" if i == 0 else "connected",
                            cost,
                            " ".join(order),
                            ",".join(str(x) for x in (flat or [])),
                        )
                    )
                    n_ok += 1
                    if n_ok % 20 == 0:
                        print(f"[collect] explained {n_ok} probe plans last={tid} cost={cost}")
        print(f"[collect] probe explained {n_ok} -> {probe_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Build ceb_task_plans.csv: connected join encodings for all CEB SQLs.

Fast path (default): no Postgres — default FROM order + random connected perms.
Probe path: also EXPLAIN-score extra perms for selected tasks.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains.bolt_vae import left_deep_flat
from mindware.domains.queryopt.ceb import IMDB_TABLES, load_sql, load_task_ids
from mindware.domains.queryopt.sql_parse import (
    alias_join_graph,
    parse_tables,
    random_connected_order,
)

PROBE_TASKS = [
    "CEB_10A35",
    "CEB_10A9",
    "CEB_1A1229",
    "CEB_1A2517",
    "CEB_11A102",
    "CEB_2B118",
    "CEB_2A412",
    "CEB_8A94",
    "CEB_8A121",
    "CEB_6A209",
    "CEB_7A119",
    "CEB_7A10",
]


def _flat_for_order(pairs, order, table_to_id):
    alias_to_table = {a: t for t, a in pairs}
    ids = [table_to_id.get(alias_to_table.get(a, a), 0) for a in order]
    if len(ids) < 2:
        return None
    return left_deep_flat(ids)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/root/data/cys/BOLT/optimization/query_plans/"
            "query_plan_optimization/training_data/encoded_ceb/ceb_task_plans.csv"
        ),
    )
    p.add_argument("--n-perm", type=int, default=20)
    p.add_argument("--probe-n-perm", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="0 = all CEB task ids")
    p.add_argument("--explain-probes", action="store_true")
    args = p.parse_args(argv)

    table_to_id = {t: i for i, t in enumerate(IMDB_TABLES)}
    rng = random.Random(args.seed)
    ids = load_task_ids(limit=args.limit or None)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        for tid in ids:
            sql = load_sql(tid)
            if not sql:
                continue
            pairs = parse_tables(sql)
            aliases = [a for _, a in pairs]
            if len(aliases) < 2:
                continue
            graph = alias_join_graph(sql, aliases)
            n_perm = args.probe_n_perm if tid in PROBE_TASKS else args.n_perm
            seen = set()
            orders = [aliases]
            for _ in range(n_perm):
                orders.append(random_connected_order(aliases, graph, rng))
            for i, order in enumerate(orders):
                key = tuple(order)
                if key in seen:
                    continue
                seen.add(key)
                flat = _flat_for_order(pairs, order, table_to_id)
                if not flat:
                    continue
                kind = "default" if i == 0 else "connected"
                w.writerow((tid, kind, ",".join(str(x) for x in flat)))
                n_rows += 1

    print(f"[collect] wrote {n_rows} rows -> {args.out} tasks={len(ids)}")

    if args.explain_probes:
        dsn = os.environ.get(
            "QUERYOPT_PG_DSN",
            "host=127.0.0.1 port=5432 dbname=imdb user=imdb password=imdb",
        )
        from mindware.domains.queryopt.oracle import postgres_explain_cost
        from mindware.domains.queryopt.sql_parse import pg_leading_hint

        probe_out = args.out.with_name("ceb_probe_explained.csv")
        n_ok = 0
        with probe_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(("task_id", "kind", "cost", "order", "flat"))
            for tid in PROBE_TASKS:
                sql = load_sql(tid)
                pairs = parse_tables(sql)
                aliases = [a for _, a in pairs]
                graph = alias_join_graph(sql, aliases)
                orders = [aliases]
                for _ in range(args.probe_n_perm):
                    orders.append(random_connected_order(aliases, graph, rng))
                seen = set()
                for i, order in enumerate(orders):
                    key = tuple(order)
                    if key in seen:
                        continue
                    seen.add(key)
                    cost = postgres_explain_cost(
                        sql, pg_leading_hint(order), dsn=dsn, join_order=order
                    )
                    flat = _flat_for_order(pairs, order, table_to_id)
                    w.writerow(
                        (
                            tid,
                            "default" if i == 0 else "connected",
                            cost,
                            " ".join(order),
                            ",".join(str(x) for x in (flat or [])),
                        )
                    )
                    n_ok += 1
                    if n_ok % 20 == 0:
                        print(f"[collect] explained {n_ok} probe plans last={tid} cost={cost}")
        print(f"[collect] probe explained {n_ok} -> {probe_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
