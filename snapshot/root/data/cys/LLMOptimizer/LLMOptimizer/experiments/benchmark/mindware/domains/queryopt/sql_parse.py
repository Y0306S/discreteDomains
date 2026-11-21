"""Parse JOIN/FROM tables from JOB/CEB-style SQL."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Set, Tuple

_FROM_CHUNK = re.compile(r"\bfrom\b(.+?)\bwhere\b", re.I | re.S)
_ITEM = re.compile(
    r"([A-Za-z_][\w.]*)\s+(?:as\s+)?([A-Za-z_]\w*)",
    re.I,
)
_EQ = re.compile(
    r"([A-Za-z_]\w*)\s*\.\s*[A-Za-z_]\w*\s*=\s*([A-Za-z_]\w*)\s*\.\s*[A-Za-z_]\w*",
    re.I,
)


def parse_tables(sql: str) -> List[Tuple[str, str]]:
    """Return (table, alias) in FROM-list order (left-deep skeleton)."""
    text = sql.strip()
    m = _FROM_CHUNK.search(text)
    chunk = m.group(1) if m else text
    pairs: List[Tuple[str, str]] = []
    seen_alias = set()
    for table, alias in _ITEM.findall(chunk):
        alias_l = alias.lower()
        table_l = table.split(".")[-1].lower()
        if alias_l in {"as", "on", "and", "or", "join", "inner", "left", "right"}:
            continue
        if alias_l in seen_alias:
            continue
        seen_alias.add(alias_l)
        pairs.append((table_l, alias_l))
    return pairs


def default_join_order(pairs: List[Tuple[str, str]]) -> List[str]:
    return [alias for _, alias in pairs]


def is_legal_order(order: List[str], aliases: List[str]) -> bool:
    return sorted(order) == sorted(aliases) and len(order) == len(set(order))


def alias_symbol_map(
    pairs: Sequence[Tuple[str, str]],
    table_to_id: Dict[str, int],
    num_aliases: int = 3,
) -> Tuple[Dict[str, Tuple[int, int]], Dict[Tuple[int, int], str]]:
    """SQL alias → (table_id, alias_num) using BOLT 1-based per-table counts."""
    alias_to_ta: Dict[str, Tuple[int, int]] = {}
    ta_to_alias: Dict[Tuple[int, int], str] = {}
    counts: Dict[str, int] = {}
    nmax = max(1, int(num_aliases))
    for table, alias in pairs:
        tid = table_to_id.get(str(table).lower())
        if tid is None:
            continue
        counts[table] = counts.get(table, 0) + 1
        anum = min(int(counts[table]), nmax)
        key = (int(tid), int(anum))
        alias_to_ta[str(alias)] = key
        ta_to_alias.setdefault(key, str(alias))
    return alias_to_ta, ta_to_alias


def connected_prefix_repair(
    prefix: Sequence[str],
    aliases: Sequence[str],
    graph: Dict[str, Set[str]],
) -> List[str]:
    """Keep the longest connected VAE prefix, then grow remaining aliases on the join graph."""
    alias_set = list(aliases)
    order: List[str] = []
    seen = set()
    for a in prefix:
        if a not in alias_set or a in seen:
            continue
        if not order:
            order.append(a)
            seen.add(a)
            continue
        if is_connected_order(order + [a], graph):
            order.append(a)
            seen.add(a)
        else:
            break
    remaining = [a for a in alias_set if a not in seen]
    while remaining:
        frontier = [a for a in remaining if any(a in graph.get(b, ()) for b in order)] if order else remaining
        nxt = frontier[0] if frontier else remaining[0]
        order.append(nxt)
        remaining.remove(nxt)
    return order


def repair_order(order: List[str], aliases: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    alias_set = set(aliases)
    for a in order:
        if a in alias_set and a not in seen:
            out.append(a)
            seen.add(a)
    for a in aliases:
        if a not in seen:
            out.append(a)
    return out


def alias_join_graph(sql: str, aliases: Sequence[str]) -> Dict[str, Set[str]]:
    """Undirected graph of aliases joined by equality predicates."""
    alias_set = {a.lower() for a in aliases}
    graph: Dict[str, Set[str]] = {a: set() for a in aliases}
    lower_to_orig = {a.lower(): a for a in aliases}
    for a1, a2 in _EQ.findall(sql or ""):
        u = lower_to_orig.get(a1.lower())
        v = lower_to_orig.get(a2.lower())
        if u and v and u != v and u.lower() in alias_set and v.lower() in alias_set:
            graph[u].add(v)
            graph[v].add(u)
    return graph


def is_connected_order(order: Sequence[str], graph: Dict[str, Set[str]]) -> bool:
    if not order:
        return False
    if not any(graph.get(a) for a in order):
        return True
    seen = {order[0]}
    for a in order[1:]:
        if any(a in graph.get(b, ()) for b in seen):
            seen.add(a)
        else:
            return False
    return True


def random_connected_order(
    aliases: Sequence[str],
    graph: Dict[str, Set[str]],
    rng,
) -> List[str]:
    """Grow a left-deep order that stays connected on `graph` when possible."""
    aliases = list(aliases)
    if not aliases:
        return []
    remaining = set(aliases)
    start = rng.choice(aliases)
    order = [start]
    remaining.remove(start)
    while remaining:
        frontier = [a for a in remaining if any(a in graph.get(b, ()) for b in order)]
        nxt = rng.choice(frontier) if frontier else rng.choice(list(remaining))
        order.append(nxt)
        remaining.remove(nxt)
    return order


def map_tables_to_aliases(
    table_names: Iterable[str],
    pairs: Sequence[Tuple[str, str]],
) -> List[str]:
    """Greedy map decoded table-name sequence onto this query's aliases."""
    unused: Dict[str, List[str]] = defaultdict(list)
    for table, alias in pairs:
        unused[table.lower()].append(alias)
    order: List[str] = []
    used = set()
    for tok in table_names:
        bucket = unused.get(str(tok).lower(), [])
        alias = next((a for a in bucket if a not in used), None)
        if alias is None:
            continue
        order.append(alias)
        used.add(alias)
    return order


def pg_leading_hint(order: List[str]) -> str:
    return "/*+ Leading(" + " ".join(order) + ") */\n"


def reorder_from(sql: str, aliases: List[str]) -> str:
    """Rewrite comma-FROM list so `from_collapse_limit=1` joins in `aliases` order.

    Needed when pg_hint_plan is not installed (conda Postgres).
    """
    pairs = parse_tables(sql)
    if not pairs or not aliases:
        return sql
    by_alias = {a: t for t, a in pairs}
    items = []
    seen = set()
    for a in aliases:
        t = by_alias.get(a)
        if t and a not in seen:
            items.append(f"{t} AS {a}")
            seen.add(a)
    for t, a in pairs:
        if a not in seen:
            items.append(f"{t} AS {a}")
            seen.add(a)
    m = _FROM_CHUNK.search(sql)
    if not m:
        return sql
    return sql[: m.start(1)] + "\n" + ",\n".join(items) + "\n" + sql[m.end(1) :]


QUERY_ROLE_HINTS = (
    ("n_tables", "join_width"),
    ("starts_with", "join_skeleton"),
    ("filter_early", "filter_early"),
    ("cartesian", "cartesian"),
    ("z_", "latent_coord"),
)
