"""Resume TPE / LLM-TPE from the last flushed trajectory row.

The runner used to treat ``--resume`` as skip-if-complete: an incomplete CASH
directory was ignored and a new study started at trial 1. These helpers copy the
existing traj, replay configs into the Optuna study / hybrid selector, and let
the next CASH loop run only the remaining trials.
"""

from __future__ import annotations

import csv
import json
import os
import pickle
import shutil
from typing import Any, Dict, List, Optional, Tuple


def load_traj_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_row_config(
    row: Dict[str, Any], default_algorithm: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    for key in ("config_json", "config_for_prompt", "config"):
        raw = row.get(key)
        if raw is None or raw == "":
            continue
        if isinstance(raw, dict):
            cfg = dict(raw)
        else:
            try:
                cfg = json.loads(raw)
            except Exception:
                continue
        if not isinstance(cfg, dict) or not cfg:
            continue
        if default_algorithm and "algorithm" not in cfg:
            cfg["algorithm"] = default_algorithm
        return cfg
    return None


def row_success_perf(row: Dict[str, Any]) -> Tuple[bool, float]:
    try:
        perf = float(row.get("perf"))
    except Exception:
        return False, float("-inf")
    if perf != perf or perf <= -1e18:
        return False, float("-inf")
    return True, perf


def sibling_path(traj_path: str, suffix: str) -> str:
    if traj_path.endswith("_traj.csv"):
        return traj_path[: -len("_traj.csv")] + suffix
    return traj_path + suffix


def study_pickle_path(traj_path: str) -> str:
    return sibling_path(traj_path, "_study.pkl")


def observations_jsonl_path(traj_path: str) -> str:
    return sibling_path(traj_path, "_observations.jsonl")


def copy_if_exists(src: str, dest: str) -> None:
    if not src or not os.path.isfile(src):
        return
    if os.path.abspath(src) == os.path.abspath(dest):
        return
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    shutil.copy2(src, dest)


def copy_traj_bundle(src_traj: str, dest_traj: str) -> None:
    copy_if_exists(src_traj, dest_traj)
    copy_if_exists(study_pickle_path(src_traj), study_pickle_path(dest_traj))
    copy_if_exists(observations_jsonl_path(src_traj), observations_jsonl_path(dest_traj))


def load_jsonl_configs(
    path: str, default_algorithm: Optional[str] = None
) -> List[Optional[Dict[str, Any]]]:
    if not path or not os.path.isfile(path):
        return []
    rows: List[Optional[Dict[str, Any]]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                rows.append(None)
                continue
            try:
                rec = json.loads(line)
            except Exception:
                rows.append(None)
                continue
            cfg = rec.get("config") if isinstance(rec, dict) else None
            if isinstance(cfg, dict):
                if default_algorithm and "algorithm" not in cfg:
                    cfg = dict(cfg)
                    cfg["algorithm"] = default_algorithm
                rows.append(cfg)
            else:
                rows.append(None)
    return rows


def load_history_json_configs(
    path: str, default_algorithm: Optional[str] = None
) -> List[Optional[Dict[str, Any]]]:
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []
    out: List[Optional[Dict[str, Any]]] = []
    for rec in payload.get("observations") or []:
        cfg = rec.get("config") if isinstance(rec, dict) else None
        if isinstance(cfg, dict) and cfg:
            if default_algorithm and "algorithm" not in cfg:
                cfg = dict(cfg)
                cfg["algorithm"] = default_algorithm
            out.append(cfg)
        else:
            out.append(None)
    return out


def _iter_topk_pkls(traj_path: str) -> List[str]:
    """Find topk pkls in this CASH dir and sibling CASH dirs of the same slot."""
    cash_dir = os.path.dirname(os.path.abspath(traj_path))
    roots = [cash_dir]
    parent = os.path.dirname(cash_dir)
    if parent:
        roots.append(parent)
    seen = set()
    found: List[str] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if not fn.endswith("topk_config.pkl"):
                    continue
                fp = os.path.join(dirpath, fn)
                if fp in seen:
                    continue
                seen.add(fp)
                found.append(fp)
    return found


def load_topk_config_items(
    traj_path: str, default_algorithm: Optional[str] = None
) -> List[Tuple[Dict[str, Any], float]]:
    """Load unique (config, score) pairs from topk pkls, oldest CASH dir first."""
    items: List[Tuple[Dict[str, Any], float]] = []
    seen = set()
    for fp in _iter_topk_pkls(traj_path):
        try:
            with open(fp, "rb") as f:
                payload = pickle.load(f)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        buckets = []
        if default_algorithm and default_algorithm in payload:
            buckets.append(payload.get(default_algorithm) or [])
        else:
            buckets.extend(payload.values())
        for bucket in buckets:
            if not isinstance(bucket, list):
                continue
            for rec in bucket:
                if not isinstance(rec, (list, tuple)) or len(rec) < 2:
                    continue
                cfg, score = rec[0], rec[1]
                if not isinstance(cfg, dict) or not cfg:
                    continue
                try:
                    score_f = float(score)
                except Exception:
                    continue
                cfg_out = dict(cfg)
                if default_algorithm and "algorithm" not in cfg_out:
                    cfg_out["algorithm"] = default_algorithm
                key = json.dumps(cfg_out, sort_keys=True, default=str)
                if key in seen:
                    continue
                seen.add(key)
                items.append((cfg_out, score_f))
    return items


def pop_topk_for_perf(
    unused: List[Tuple[Dict[str, Any], float]], perf: float, atol: float = 1e-8
) -> Optional[Dict[str, Any]]:
    best_i = None
    best_d = None
    target = float(perf)
    for i, (_cfg, score) in enumerate(unused):
        d = abs(float(score) - target)
        if d <= atol and (best_d is None or d < best_d):
            best_i, best_d = i, d
    if best_i is None:
        return None
    cfg, _score = unused.pop(best_i)
    return dict(cfg)


def resolve_row_configs(
    rows: List[Dict[str, Any]],
    *,
    traj_path: str,
    default_algorithm: Optional[str] = None,
    jsonl_cfgs: Optional[List[Optional[Dict[str, Any]]]] = None,
    hist_cfgs: Optional[List[Optional[Dict[str, Any]]]] = None,
) -> Tuple[List[Optional[Dict[str, Any]]], int]:
    """Fill each traj row's config from CSV/jsonl/history/topk. Returns configs, n_from_topk."""
    jsonl_cfgs = jsonl_cfgs or []
    hist_cfgs = hist_cfgs or []
    unused = load_topk_config_items(traj_path, default_algorithm)
    recovered: List[Optional[Dict[str, Any]]] = []
    n_topk = 0
    for i, row in enumerate(rows):
        cfg = parse_row_config(row, default_algorithm=default_algorithm)
        if cfg is None and i < len(jsonl_cfgs):
            cfg = jsonl_cfgs[i]
        if cfg is None and i < len(hist_cfgs):
            cfg = hist_cfgs[i]
        if cfg is None:
            success, perf = row_success_perf(row)
            if success:
                cfg = pop_topk_for_perf(unused, perf)
                if cfg is not None:
                    n_topk += 1
        if cfg is not None:
            if default_algorithm and "algorithm" not in cfg:
                cfg = dict(cfg)
                cfg["algorithm"] = default_algorithm
            row["config_json"] = json.dumps(cfg, sort_keys=True, default=str)
        recovered.append(cfg)
    return recovered, n_topk


def rewrite_traj_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not path or not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    preferred = [
        "ts",
        "stage",
        "iter_idx",
        "perf",
        "best_so_far",
        "algorithm",
        "smac_time",
        "eval_time",
        "iter_time",
        "wall_elapsed_sec",
        "note",
        "fallback",
        "config_json",
        "config_for_prompt",
        "llm_cost",
    ]
    for key in preferred:
        if any(key in r for r in rows) or key == "config_json":
            fieldnames.append(key)
            seen.add(key)
    for r in rows:
        for key in r.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def rewrite_observations_jsonl(
    path: str,
    rows: List[Dict[str, Any]],
    configs: List[Optional[Dict[str, Any]]],
    default_algorithm: Optional[str] = None,
) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            cfg = configs[i] if i < len(configs) else None
            success, _perf = row_success_perf(row)
            rec = {
                "iter_idx": row.get("iter_idx", i),
                "perf": row.get("perf"),
                "success": success,
                "algorithm": row.get("algorithm") or default_algorithm,
                "config": cfg,
            }
            f.write(json.dumps(rec, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def persist_jsonl_row(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def persist_study_pickle(path: str, study: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(study, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def try_load_study_pickle(path: str) -> Any:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def cfg_as_json(cfg: Any) -> str:
    if hasattr(cfg, "get_dictionary"):
        d = cfg.get_dictionary()
    elif isinstance(cfg, dict):
        d = cfg
    else:
        d = {"repr": str(cfg)}
    return json.dumps(d, sort_keys=True, default=str)
