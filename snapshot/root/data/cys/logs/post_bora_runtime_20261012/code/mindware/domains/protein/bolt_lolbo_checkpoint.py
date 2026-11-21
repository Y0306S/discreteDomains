"""Save / restore BOLT LOLBO checkpoints for multi-phase runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from mindware.domains.base import z_to_config
from mindware.domains.protein.checkpoint_rotate import (
    checkpoint_candidates,
    commit_two_deep_checkpoint,
    prev_checkpoint_path,
)


def checkpoint_path(hist_path: Path) -> Path:
    return Path(hist_path).parent / "checkpoint.pt"


def _config_to_z(cfg: Dict[str, Any]) -> List[float]:
    keys = sorted(
        (k for k in cfg if str(k).startswith("z_")),
        key=lambda x: int(str(x).split("_", 1)[1]),
    )
    return [float(cfg[k]) for k in keys]


def tensors_from_observations(
    observations: List[Dict[str, Any]],
    *,
    minimize: bool = False,
) -> Tuple[List[str], torch.Tensor, torch.Tensor, Dict[str, float]]:
    train_x: List[str] = []
    ys: List[float] = []
    zs: List[List[float]] = []
    xs_dict: Dict[str, float] = {}
    for obs in observations:
        extra = obs.get("extra_info") or {}
        seq = str(extra.get("artifact") or "")
        if not seq:
            continue
        y = float((obs.get("objectives") or [0.0])[0])
        cfg = obs.get("config") or {}
        z = _config_to_z(cfg)
        if not z:
            continue
        train_x.append(seq)
        ys.append(y)
        zs.append(z)
        xs_dict[seq] = y
    train_y = torch.tensor(ys, dtype=torch.float32).unsqueeze(-1)
    train_z = torch.tensor(zs, dtype=torch.float32)
    if minimize:
        train_y = train_y * -1
    return train_x, train_y, train_z, xs_dict


def vae_state_dict_from_objective(objective: Any) -> Optional[Dict[str, Any]]:
    """Extract finetuned VAE weights from a BOLT LOLBO objective, if present."""
    from mindware.domains.protein.bolt_peptide_e2e import capture_vae_state_dict

    return capture_vae_state_dict(getattr(objective, "vae", None))


def restore_vae_to_objective(objective: Any, state_dict: Optional[Dict[str, Any]]) -> bool:
    from mindware.domains.protein.bolt_peptide_e2e import restore_vae_state_dict

    return restore_vae_state_dict(getattr(objective, "vae", None), state_dict)


def save_checkpoint(
    path: Path,
    *,
    train_x: List[str],
    train_y: torch.Tensor,
    train_z: torch.Tensor,
    train_c: Optional[torch.Tensor],
    num_calls: int,
    xs_to_scores_dict: Dict[str, float],
    n_init_loaded: int,
    tot_num_e2e_updates: int,
    best_x_seen: Any,
    best_score_seen: Any,
    minimize: bool,
    vae_state_dict: Optional[Dict[str, Any]] = None,
    n_acq_calls: int = 0,
    n_e2e_calls: int = 0,
    n_rounds: int = 0,
    spans: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spans": list(spans or []),
        "train_x": list(train_x),
        "train_y": train_y.detach().cpu(),
        "train_z": train_z.detach().cpu(),
        "train_c": train_c.detach().cpu() if train_c is not None else None,
        "num_calls": int(num_calls),
        "xs_to_scores_dict": dict(xs_to_scores_dict),
        "n_init_loaded": int(n_init_loaded),
        "tot_num_e2e_updates": int(tot_num_e2e_updates),
        # Cumulative acquisition accounting, so a resumed run continues UP TO the budget
        # instead of doing a further `budget` calls on top of what it already spent.
        "n_acq_calls": int(n_acq_calls),
        "n_e2e_calls": int(n_e2e_calls),
        "n_rounds": int(n_rounds),
        "best_x_seen": best_x_seen,
        "best_score_seen": float(best_score_seen) if best_score_seen is not None else None,
        "minimize": bool(minimize),
    }
    if vae_state_dict:
        payload["vae_state_dict"] = vae_state_dict
    tmp = path.with_name(path.name + ".tmp")
    latest_meta = path.with_name("checkpoint.meta.json")
    torch.save(payload, tmp)
    commit_two_deep_checkpoint(tmp, path)
    prev_meta = prev_checkpoint_path(latest_meta)
    if prev_meta.is_file():
        prev_meta.unlink(missing_ok=True)
    if latest_meta.is_file():
        latest_meta.replace(prev_meta)
    latest_meta.write_text(
        json.dumps(
            {
                "num_calls": int(num_calls),
                "tot_num_e2e_updates": int(tot_num_e2e_updates),
                "has_vae_state_dict": bool(vae_state_dict),
                "role": "latest",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def load_checkpoint(path: Path) -> Dict[str, Any]:
    path = Path(path)
    last_err: Optional[Exception] = None
    for cand in checkpoint_candidates(path):
        try:
            return torch.load(cand, map_location="cpu", weights_only=False)
        except Exception as exc:
            last_err = exc
            continue
    if last_err is not None:
        raise last_err
    raise FileNotFoundError(f"LOLBO checkpoint missing: {path}")


def build_checkpoint_from_history(hist_path: Path, *, out_path: Optional[Path] = None) -> Path:
    hist_path = Path(hist_path)
    payload = json.loads(hist_path.read_text(encoding="utf-8"))
    meta = payload.get("meta_info") or payload.get("meta") or {}
    observations = list(payload.get("observations") or [])
    if not observations:
        raise ValueError(f"empty history: {hist_path}")
    minimize = False
    train_x, train_y, train_z, xs_dict = tensors_from_observations(observations, minimize=minimize)
    out = out_path or checkpoint_path(hist_path)
    return save_checkpoint(
        out,
        train_x=train_x,
        train_y=train_y,
        train_z=train_z,
        train_c=None,
        num_calls=int(meta.get("n_oracle_calls") or 0),
        xs_to_scores_dict=xs_dict,
        n_init_loaded=int(meta.get("n_init_points") or 1000),
        tot_num_e2e_updates=int(meta.get("e2e_updates") or 0),
        best_x_seen=None,
        best_score_seen=None,
        minimize=minimize,
    )


def ensure_checkpoint(hist_path: Path) -> Path:
    ckpt = checkpoint_path(hist_path)
    if ckpt.is_file():
        return ckpt
    hist_path = Path(hist_path)
    if not hist_path.is_file():
        raise FileNotFoundError(f"LOLBO history missing: {hist_path}")
    return build_checkpoint_from_history(hist_path, out_path=ckpt)
