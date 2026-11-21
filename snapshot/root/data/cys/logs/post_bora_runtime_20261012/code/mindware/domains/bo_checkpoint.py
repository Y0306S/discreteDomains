"""Atomic per-trial BO checkpoints: history.json + optimizer pickle + pending eval.

Used so a TaskSet (or any domain) run can move to another machine and continue
with the same TPE/SMAC state, without re-evaluating completed trials.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

from mindware.domains.history import write_rotating_openbox_history
from mindware.domains.protein.checkpoint_rotate import checkpoint_candidates, commit_two_deep_checkpoint

STUDY_NAME = "optuna_study.pkl"
SMAC_NAME = "smac_advisor.pkl"
PENDING_NAME = "pending.json"


def study_path(output_dir: Path) -> Path:
    return Path(output_dir) / STUDY_NAME


def smac_path(output_dir: Path) -> Path:
    return Path(output_dir) / SMAC_NAME


def pending_path(output_dir: Path) -> Path:
    return Path(output_dir) / PENDING_NAME


def _atomic_pickle(obj: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(obj, fh, protocol=4)
        fh.flush()
        os.fsync(fh.fileno())
    commit_two_deep_checkpoint(tmp, path)
    return path


def _load_pickle(path: Path) -> Any:
    for cand in checkpoint_candidates(Path(path)):
        try:
            with cand.open("rb") as fh:
                return pickle.load(fh)
        except Exception:
            continue
    return None


def save_optuna_study(study: Any, output_dir: Path) -> Optional[Path]:
    if study is None:
        return None
    try:
        return _atomic_pickle(study, study_path(output_dir))
    except Exception:
        return None


def load_optuna_study(output_dir: Path, *, n_complete: int) -> Any:
    """Return a pickled study if its COMPLETE trials match history length."""
    import optuna
    from optuna.trial import TrialState

    study = _load_pickle(study_path(output_dir))
    if study is None:
        return None
    try:
        n_ok = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
    except Exception:
        return None
    if n_ok != int(n_complete):
        return None
    return study


def save_smac_advisor(advisor: Any, output_dir: Path) -> Optional[Path]:
    if advisor is None:
        return None
    try:
        return _atomic_pickle(advisor, smac_path(output_dir))
    except Exception:
        return None


def load_smac_advisor(output_dir: Path, *, n_complete: int) -> Any:
    advisor = _load_pickle(smac_path(output_dir))
    if advisor is None:
        return None
    hist = getattr(advisor, "history", None)
    n_ok = None
    if hist is not None:
        try:
            n_ok = len(getattr(hist, "observations", hist) or [])
        except Exception:
            n_ok = None
        if n_ok is None:
            try:
                n_ok = len(list(hist))
            except Exception:
                n_ok = None
    if n_ok is None:
        try:
            n_ok = len(advisor.history.observations)
        except Exception:
            return None
    if int(n_ok) != int(n_complete):
        return None
    return advisor


def write_pending(output_dir: Path, config: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Path:
    path = pending_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": dict(config), **(extra or {})}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_pending(output_dir: Path) -> Optional[Dict[str, Any]]:
    path = pending_path(output_dir)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not payload.get("config"):
        return None
    return payload


def clear_pending(output_dir: Path) -> None:
    path = pending_path(output_dir)
    if path.is_file():
        path.unlink()


def persist_search_checkpoint(
    output_dir: Path,
    *,
    task_id: str,
    observations: List[Dict[str, Any]],
    meta: Dict[str, Any],
    study: Any = None,
    smac_advisor: Any = None,
) -> Path:
    hist = write_rotating_openbox_history(
        Path(output_dir) / "history.json",
        task_id=task_id,
        observations=observations,
        meta=meta,
    )
    save_optuna_study(study, output_dir)
    save_smac_advisor(smac_advisor, output_dir)
    clear_pending(output_dir)
    return hist
