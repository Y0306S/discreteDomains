"""Convert zaxie ``tune_report.json`` Optuna TPE logs to OpenBox History JSON."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from mindware.domains.history import observation, scored_only, write_openbox_history

ZAXIE_KEYS = (
    "depth",
    "device_batch_size",
    "grad_accum_steps",
    "embedding_lr",
    "unembedding_lr",
    "matrix_lr",
    "scalar_lr",
    "weight_decay",
    "warmup_ratio",
    "warmdown_ratio",
    "final_lr_frac",
)

DL_ROLE_HINTS = (
    ("embedding_lr", "learning_rate"),
    ("unembedding_lr", "learning_rate"),
    ("matrix_lr", "learning_rate"),
    ("scalar_lr", "learning_rate"),
    ("learning_rate", "learning_rate"),
    ("weight_decay", "regularization"),
    ("warmup", "schedule"),
    ("warmdown", "schedule"),
    ("final_lr", "schedule"),
    ("grad_accum", "batch_or_budget"),
    ("device_batch", "batch_or_budget"),
    ("depth", "tree_depth"),
    ("beta1", "optimizer_choice"),
    ("beta2", "optimizer_choice"),
    ("epsilon", "optimizer_choice"),
)


def iter_tune_reports(root: Path) -> Iterator[Path]:
    root = Path(root)
    if root.is_file() and root.name == "tune_report.json":
        yield root
        return
    for path in root.rglob("tune_report.json"):
        yield path


def _finite_score(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return val


def extract_trials(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    phase_c = report.get("phase_c") or {}
    for stage in phase_c.get("stages") or []:
        method = str(stage.get("method") or "bo")
        for trial in stage.get("trials") or []:
            params = dict(trial.get("params") or {})
            if not params:
                continue
            score = _finite_score(trial.get("score"))
            status = str(trial.get("status") or "")
            if score is None:
                rows.append(
                    observation(
                        params,
                        1e10,
                        trial_state=1,
                        extra={"source": "zaxie_tpe", "status": status or "failed", "method": method},
                    )
                )
                continue
            rows.append(
                observation(
                    params,
                    score,
                    trial_state=0,
                    extra={"source": "zaxie_tpe", "status": status or "ok", "method": method},
                )
            )
    return rows


def task_id_from_path(path: Path) -> str:
    parts = [p for p in path.parent.parts if p not in {".", ""}]
    tail = parts[-4:] if len(parts) >= 2 else parts
    return "__".join(tail)


def convert_tune_report(
    path: Path,
    *,
    out_dir: Path,
    min_scored: int = 10,
) -> Optional[Path]:
    path = Path(path)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    rows = extract_trials(report)
    scored = scored_only(rows)
    if len(scored) < int(min_scored):
        return None
    tid = task_id_from_path(path)
    dest = Path(out_dir) / "val_loss" / tid / "tpe.json"
    space = None
    space_path = path.parent / "_search_space.json"
    if space_path.is_file():
        try:
            space = json.loads(space_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            space = None
    write_openbox_history(
        dest,
        task_id=tid,
        observations=rows,
        meta={
            "source": "zaxie_tune_report",
            "src_path": str(path),
            "n_scored": len(scored),
            "final_best_score": report.get("final_best_score"),
            "search_space": space,
        },
    )
    return dest


def convert_tree(
    root: Path,
    out_dir: Path,
    *,
    min_scored: int = 10,
    max_tasks: int = 0,
) -> List[Path]:
    written: List[Path] = []
    seen = set()
    for path in iter_tune_reports(root):
        dest = convert_tune_report(path, out_dir=out_dir, min_scored=min_scored)
        if dest is None or dest in seen:
            continue
        seen.add(dest)
        written.append(dest)
        if max_tasks and len(written) >= int(max_tasks):
            break
    return written
