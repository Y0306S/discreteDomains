"""BOLT query execution oracle (runtime) helpers for TPE / arm-C benchmarks."""

from __future__ import annotations

import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mindware.domains.paths import QUERY_ROOT

INIT_DIR = QUERY_ROOT / "query_plan_optimization" / "initialization_data"
BAYESQO_INIT_DIR = Path(
    os.environ.get("BAYESQO_INIT_DIR") or "/root/data/cys/BayesQO/bayes_lqo/initialization_data"
)
BAO_CSV = INIT_DIR / "ceb_3k_bao_initialization.csv"
BAO_CENSORED_CSV = INIT_DIR / "bao_censored_initializations.csv"
BAO_CENSORED_FALLBACK = BAYESQO_INIT_DIR / "bao_censored_initializations.csv"


def bao_censored_csv_paths() -> List[Path]:
    """JOB init CSV: BOLT tree first, then BayesQO copy (often git-lfs stub only)."""
    out: List[Path] = []
    for p in (BAO_CENSORED_CSV, BAO_CENSORED_FALLBACK):
        if p not in out:
            out.append(p)
    return out


def runtime_to_lolbo_score(runtime_secs: float) -> float:
    """BayesQO maximizes f(x); DatabaseObjective returns score = -runtime."""
    return -float(runtime_secs)


def worst_runtime_from_lolbo_scores(scores: Sequence[float]) -> float:
    """Per-task worst_runtime_observed (BayesQO: min(score) * -1)."""
    if not scores:
        return 200.0
    return float(min(float(s) for s in scores) * -1.0)


def parse_pg_dsn(dsn: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for tok in str(dsn or "").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k.strip().lower()] = v.strip()
    return out


def configure_bolt_db_from_dsn(dsn: str) -> Dict[str, str]:
    """Map libpq-style DSN to BOLT oracle env vars (psycopg v3)."""
    p = parse_pg_dsn(dsn)
    cfg = {
        "DB_HOST": p.get("host", "127.0.0.1"),
        "DB_PORT": str(p.get("port", "5432")),
        "DB_USER": p.get("user", "imdb"),
        "DB_PASSWORD": p.get("password", "imdb"),
        "DB_NAME": p.get("dbname") or p.get("database") or p.get("user", "imdb"),
    }
    for k, v in cfg.items():
        os.environ[k] = str(v)
    return cfg


def _ensure_bolt_qp_path() -> str:
    qp = str(QUERY_ROOT / "query_plan_optimization")
    if qp not in sys.path:
        sys.path.insert(0, qp)
    return qp


def make_database_objective(
    workload_name: str,
    *,
    timeout: float = 100.0,
    worst_runtime_observed: float = 200.0,
) -> Any:
    configure_bolt_db_from_dsn(os.environ.get("QUERYOPT_PG_DSN", ""))
    _ensure_bolt_qp_path()
    import importlib.util

    mod_path = QUERY_ROOT / "query_plan_optimization/your_tasks/your_objective_functions.py"
    spec = importlib.util.spec_from_file_location("bolt_query_db_objective", mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load DatabaseObjective from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    DatabaseObjective = mod.DatabaseObjective

    return DatabaseObjective(
        workload_name=str(workload_name),
        worst_runtime_observed=float(worst_runtime_observed),
        timeout=float(timeout),
        which_language="aliases",
    )


def _csv_is_real(path: Path, min_bytes: int = 500) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    if p.stat().st_size < int(min_bytes):
        return False
    head = p.read_text(encoding="utf-8", errors="ignore")[:80]
    return "git-lfs" not in head


def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in re.split(r"[,;\s]+", str(s).strip()) if x.strip().lstrip("-").isdigit()]


def _valid_ceb_task_id(name: str) -> bool:
    return bool(re.match(r"^CEB_\d+[A-Z]\d+$", str(name or "").strip()))


def load_bao_init(
    task_id: str,
) -> Optional[List[Tuple[List[int], float, int]]]:
    """Return [(encoded_plan, runtime_secs, censored), ...] or None if data missing."""
    import csv

    if not _valid_ceb_task_id(task_id):
        return None

    rows: List[Tuple[List[int], float, int]] = []

    def _load_csv(
        path: Path,
        *,
        sep: str = ",",
        plan_key: str,
        cens_key: str,
        plan_id_key: Optional[str] = None,
    ) -> None:
        if not _csv_is_real(path):
            return
        seen_plan_ids: set[str] = set()
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=sep):
                if str(row.get("query_name", "")).strip() != task_id:
                    continue
                if plan_id_key:
                    pid = str(row.get(plan_id_key, "")).strip()
                    if not pid or pid in seen_plan_ids:
                        continue
                    seen_plan_ids.add(pid)
                plan = _parse_int_list(row.get(plan_key, ""))
                if not plan:
                    continue
                try:
                    rt = float(row["runtime_secs"])
                except (KeyError, TypeError, ValueError):
                    continue
                cens_raw = row.get(cens_key, 0)
                cens = 1 if str(cens_raw).strip() in {"1", "True", "true"} else 0
                rows.append((plan, rt, cens))

    for path in bao_censored_csv_paths():
        _load_csv(path, sep=";", plan_key="plan", cens_key="censored")
        if rows:
            break
    if not rows:
        _load_csv(
            BAO_CSV,
            plan_key="encoded_plan",
            cens_key="timed_out",
            plan_id_key="plan_id",
        )
    return rows or None


def bao_lolbo_init(
    adapter: Any,
    task_id: str,
) -> Optional[Dict[str, Any]]:
    """BayesQO-style LOLBO init: encoded plans, -runtime scores, censoring, z."""
    bao = load_bao_init(task_id)
    if not bao:
        return None
    vae = adapter.vae
    xs: List[List[int]] = []
    ys: List[float] = []
    zs: List[List[float]] = []
    censoring: List[int] = []
    seen: set[tuple[int, ...]] = set()
    for enc, rt, cens in bao:
        key = tuple(int(x) for x in enc)
        if len(key) < 2 or key in seen:
            continue
        try:
            z = vae.encode_flat(enc)
        except Exception:
            continue
        seen.add(key)
        runtime = float(rt)
        xs.append(list(key))
        ys.append(runtime_to_lolbo_score(runtime))
        zs.append([float(v) for v in z])
        censoring.append(int(cens))
    if not xs:
        return None
    worst = worst_runtime_from_lolbo_scores(ys)
    return {
        "xs": xs,
        "ys": ys,
        "zs": zs,
        "censoring": censoring,
        "worst_runtime_observed": worst,
    }


def plan_has_crossjoin(db_obj: Any, encoded: Sequence[int]) -> bool:
    _ensure_bolt_qp_path()
    from oracle.oracle import plan_has_crossjoin as _has_xj  # type: ignore

    return bool(_has_xj(db_obj.full_workload_spec, list(encoded)))


def execute_encoded(
    db_obj: Any,
    encoded: Sequence[int],
    *,
    timeout: Optional[float] = None,
) -> Tuple[float, int]:
    """Execute one encoded plan; return (runtime_secs, censored 0/1)."""
    flat = [int(x) for x in encoded]
    to = float(timeout if timeout is not None else db_obj.timeout)
    if plan_has_crossjoin(db_obj, flat):
        bad = min(float(db_obj.worst_runtime_observed), to * 2.0)
        return bad, 1
    scores, censoring = db_obj.query_black_box([flat], [to])
    score = float(scores[0])
    cens = int(censoring[0]) if censoring else 0
    if not (score == score):  # nan
        bad = min(float(db_obj.worst_runtime_observed), to * 2.0)
        return bad, 1
    runtime = abs(score)
    if runtime > 0:
        db_obj.worst_runtime_observed = max(float(db_obj.worst_runtime_observed), runtime)
    return runtime, cens


def runtime_warm_start_candidates(
    *,
    task_id: str,
    db_obj: Any,
    vae: Any,
    alias_to_ta: Dict[str, Tuple[int, int]],
    aliases: Sequence[str],
    join_graph: Dict[str, Sequence[str]],
    rng: random.Random,
    n_perm: int = 15,
    n_random_z: int = 12,
    latent_dim: int = 64,
) -> List[Tuple[List[int], float, int, Optional[List[float]]]]:
    """Collect init points as (encoded, runtime, censored, z_or_none)."""
    out: List[Tuple[List[int], float, int, Optional[List[float]]]] = []
    seen: set[tuple[int, ...]] = set()

    def _add(encoded: Sequence[int], z: Optional[List[float]] = None) -> None:
        key = tuple(int(x) for x in encoded)
        if len(key) < 5 or key in seen:
            return
        seen.add(key)
        rt, cens = execute_encoded(db_obj, key)
        out.append((list(key), rt, cens, z))

    bao = load_bao_init(task_id)
    if bao:
        for enc, rt, cens in bao:
            key = tuple(int(x) for x in enc)
            if key in seen:
                continue
            seen.add(key)
            z = None
            if hasattr(vae, "encode_flat"):
                try:
                    z = vae.encode_flat(enc)
                except Exception:
                    z = None
            out.append((list(enc), float(rt), int(cens), z))
        return out

    enc_pairs = getattr(vae, "table_alias_pairs_to_flat", None)
    if callable(enc_pairs) and alias_to_ta:
        default_pairs = [alias_to_ta[a] for a in aliases if a in alias_to_ta]
        if len(default_pairs) >= 2:
            _add(enc_pairs(default_pairs))

        from mindware.domains.queryopt.sql_parse import random_connected_order

        for _ in range(max(0, int(n_perm))):
            order = random_connected_order(list(aliases), join_graph, rng)
            pairs = [alias_to_ta[a] for a in order if a in alias_to_ta]
            if len(pairs) >= 2:
                _add(enc_pairs(pairs))

    z0 = [0.0] * int(latent_dim)
    for k in range(int(n_random_z)):
        z = [float(v) + 0.35 * rng.gauss(0, 1) for v in z0]
        if hasattr(vae, "sample_flat"):
            try:
                flat = vae.sample_flat(z, sample_seed=rng.randint(0, 2**30))
                _add(flat, z)
            except Exception:
                continue
    return out
