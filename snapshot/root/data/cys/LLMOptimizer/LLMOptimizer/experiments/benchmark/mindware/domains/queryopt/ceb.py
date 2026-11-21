"""CEB-3k task id → SQL file helpers (shared by converters and BOLT BO)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from mindware.domains.paths import QUERY_ROOT

CEB_SQL_ROOT_3K = QUERY_ROOT / "query_plan_optimization/workload/ceb-3k"
CEB_SQL_ROOT_13K = QUERY_ROOT / "query_plan_optimization/workload/ceb-13k"
CEB_TASKS = QUERY_ROOT / "tasks/ceb_tasks.txt"


def _sql_root() -> Path:
    """Prefer CEB-13k once it has SQL; else CEB-3k."""
    if CEB_SQL_ROOT_13K.is_dir() and any(CEB_SQL_ROOT_13K.rglob("*.sql")):
        return CEB_SQL_ROOT_13K
    return CEB_SQL_ROOT_3K


CEB_SQL_ROOT = _sql_root()  # resolved at import; refresh via sql_path_for_task(sql_root=...)

# Canonical IMDB table list used when BOLT's job/schema.sql is missing.
# Sorted order matches sqlglot `get_all_table_names` (alphabetical).
IMDB_TABLES = [
    "aka_name",
    "aka_title",
    "cast_info",
    "char_name",
    "comp_cast_type",
    "company_name",
    "company_type",
    "complete_cast",
    "info_type",
    "keyword",
    "kind_type",
    "link_type",
    "movie_companies",
    "movie_info",
    "movie_info_idx",
    "movie_keyword",
    "movie_link",
    "name",
    "person_info",
    "role_type",
    "title",
]


def load_task_ids(limit: Optional[int] = None, offset: int = 0) -> List[str]:
    ids: List[str] = []
    if CEB_TASKS.is_file():
        ids = [ln.strip() for ln in CEB_TASKS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not ids:
        ids = ["CEB_" + p.stem.upper() for p in sorted(_sql_root().rglob("*.sql"))]
    ids = ids[int(offset) :]
    if limit is not None:
        ids = ids[: int(limit)]
    return ids


def sql_path_for_task(task_id: str, sql_root: Optional[Path] = None) -> Optional[Path]:
    """CEB_8A94 → ceb-3k/8a/8a94.sql (or ceb-13k when present)."""
    root = Path(sql_root) if sql_root is not None else _sql_root()
    raw = task_id.replace("CEB_", "").replace("ceb_", "")
    i = 0
    while i < len(raw) and raw[i].isdigit():
        i += 1
    num, rest = raw[:i], raw[i:]
    letter = rest[:1].lower()
    suffix = rest[1:]
    folder = f"{num}{letter}"
    fname = f"{folder}{suffix}.sql"
    cand = root / folder / fname
    if cand.is_file():
        return cand
    hits = list(root.glob(f"**/{folder}*.sql"))
    for h in hits:
        if h.stem.lower().endswith(suffix.lower()) or h.stem.lower() == f"{folder}{suffix}".lower():
            return h
    return hits[0] if hits else None


def load_sql(task_id: str) -> str:
    path = sql_path_for_task(task_id)
    if path is None:
        return ""
    return path.read_text(encoding="utf-8", errors="replace")
"""CEB-3k task id → SQL file helpers (shared by converters and BOLT BO)."""
