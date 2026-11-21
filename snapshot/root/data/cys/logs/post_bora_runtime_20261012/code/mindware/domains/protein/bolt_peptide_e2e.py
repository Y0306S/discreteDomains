"""BOLT-style end-to-end VAE finetune + latent recenter for peptide TPE/LOLBO-lite paths."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from mindware.domains.base import z_to_config
from mindware.domains.protein.bolt_seed0_init import BOLT_SEED0_HPARAMS

from mindware.domains.protein.checkpoint_rotate import (
    checkpoint_candidates,
    commit_two_deep_checkpoint,
)

PEPTIDE_VAE_CHECKPOINT_NAME = "vae_checkpoint.pt"
TPE_RUN_CHECKPOINT_INTERVAL = 1000
TPE_PROGRESS_LOG_INTERVAL = 100


def peptide_vae_checkpoint_path(output_dir: Path) -> Path:
    return Path(output_dir) / PEPTIDE_VAE_CHECKPOINT_NAME


def _vae_torch_module(vae: Any) -> Any:
    if vae is None:
        return None
    inner = getattr(vae, "vae", None)
    if inner is not None and callable(getattr(inner, "state_dict", None)):
        return inner
    if callable(getattr(vae, "state_dict", None)):
        return vae
    return None


def capture_vae_state_dict(vae: Any) -> Optional[Dict[str, Any]]:
    """Extract finetuned VAE weights (PeptideBoltVAE or raw nn.Module)."""
    module = _vae_torch_module(vae)
    if module is None:
        return None
    try:
        return {k: v.detach().cpu() for k, v in module.state_dict().items()}
    except Exception:
        return None


def restore_vae_state_dict(vae: Any, state_dict: Optional[Dict[str, Any]]) -> bool:
    module = _vae_torch_module(vae)
    if module is None or not state_dict:
        return False
    try:
        module.load_state_dict(state_dict, strict=True)
        return True
    except Exception:
        try:
            module.load_state_dict(state_dict, strict=False)
            return True
        except Exception:
            return False


def save_peptide_vae_checkpoint(
    vae: Any,
    path: Path,
    *,
    e2e_updates: int = 0,
    n_observations: int = 0,
) -> Optional[Path]:
    state_dict = capture_vae_state_dict(vae)
    if not state_dict:
        return None
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    payload = {
        "vae_state_dict": state_dict,
        "e2e_updates": int(e2e_updates),
        "n_observations": int(n_observations),
    }
    torch.save(payload, tmp)
    commit_two_deep_checkpoint(tmp, path)
    return path


def load_peptide_vae_checkpoint(vae: Any, path: Path) -> Dict[str, Any]:
    """Restore finetuned VAE; returns checkpoint meta (empty dict if missing)."""
    import torch

    path = Path(path)
    last_err: Optional[Exception] = None
    for cand in checkpoint_candidates(path):
        try:
            payload = torch.load(cand, map_location="cpu", weights_only=False)
            state_dict = payload.get("vae_state_dict")
            ok = restore_vae_state_dict(vae, state_dict)
            return {
                "e2e_updates": int(payload.get("e2e_updates") or 0),
                "n_observations": int(payload.get("n_observations") or 0),
                "vae_restored": bool(ok),
                "checkpoint_path": str(cand),
            }
        except Exception as exc:
            last_err = exc
            continue
    if last_err is not None:
        return {"vae_restored": False, "load_error": str(last_err)}
    return {}


def _obs_sequence(row: Dict[str, Any]) -> Optional[str]:
    extra = row.get("extra_info") or {}
    seq = extra.get("artifact")
    if seq:
        return str(seq)
    info = extra.get("info") or {}
    seq = info.get("sequence")
    return str(seq) if seq else None


def sequences_for_peptide_e2e(
    observations: Sequence[Dict[str, Any]],
    *,
    recent: int = 50,
    top_k: int = 50,
) -> List[str]:
    """Recent batch + top-k scored sequences (BOLT e2e training set, no GP)."""
    scored: List[tuple[float, str]] = []
    for row in observations:
        seq = _obs_sequence(row)
        if not seq:
            continue
        try:
            obj = float((row.get("objectives") or [1e10])[0])
        except (TypeError, ValueError):
            obj = 1e10
        if not math.isfinite(obj) or obj >= 1e9:
            continue
        scored.append((obj, seq))
    scored.sort(key=lambda x: x[0])
    top = [s for _, s in scored[: max(1, int(top_k))]]

    recent_seqs: List[str] = []
    for row in reversed(observations):
        seq = _obs_sequence(row)
        if not seq:
            continue
        recent_seqs.append(seq)
        if len(recent_seqs) >= int(recent):
            break
    recent_seqs.reverse()

    out: List[str] = []
    seen: set[str] = set()
    for seq in recent_seqs + top:
        if seq in seen:
            continue
        seen.add(seq)
        out.append(seq)
    return out


def finetune_peptide_bolt_vae(
    vae: Any,
    sequences: Sequence[str],
    *,
    num_update_epochs: Optional[int] = None,
    learning_rte: Optional[float] = None,
) -> int:
    """VAE-only e2e update (no GP). Returns number of optimizer steps run."""
    if not sequences:
        return 0
    forward = getattr(vae, "_vae_forward", None)
    model = getattr(vae, "vae", None)
    dataobj = getattr(vae, "dataobj", None)
    if forward is None or model is None or dataobj is None:
        return 0

    import torch

    n_epochs = int(num_update_epochs or BOLT_SEED0_HPARAMS.get("num_update_epochs") or 2)
    lr = float(learning_rte or BOLT_SEED0_HPARAMS.get("learning_rte") or 1e-3)
    seqs = [str(s) for s in sequences if str(s)]
    if not seqs:
        return 0

    was_training = bool(model.training)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    max_string_length = max(len(s) for s in seqs)
    bsz = max(1, int(2560 / max(1, max_string_length)))
    steps = 0
    for _ in range(n_epochs):
        for start in range(0, len(seqs), bsz):
            batch = seqs[start : start + bsz]
            z, vae_loss = forward(batch, dataobj, model)
            if vae_loss is None:
                continue
            loss = vae_loss if hasattr(vae_loss, "backward") else torch.as_tensor(vae_loss)
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            steps += 1
    model.eval()
    if was_training:
        model.train()
    return steps


def _artifact_key(x: Any) -> Any:
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return str(x)


def reencode_lolbo_state_latents(state: Any) -> int:
    """Replace train_z with encoder(train_x). Keep x/y. No decode, no oracle.

    LOLBO's native recenter() samples x' = decode(encode(x)) and oracles new
    strings, which inflates history far past init+acquisition. After an E2E
    VAE update we only need new latent coordinates for the same scored x.
    """
    import torch

    xs = list(getattr(state, "train_x", None) or [])
    objective = getattr(state, "objective", None)
    old_z = getattr(state, "train_z", None)
    if not xs or objective is None or old_z is None:
        return 0
    forward = getattr(objective, "vae_forward", None)
    if not callable(forward):
        return 0

    max_len = 1
    for x in xs:
        if isinstance(x, str):
            max_len = max(max_len, len(x) or 1)
        elif isinstance(x, (list, tuple)):
            max_len = max(max_len, len(x) or 1)
    bsz = max(1, int(2560 / max_len))
    vae = getattr(objective, "vae", None)
    was_training = bool(getattr(vae, "training", False)) if vae is not None else False
    if vae is not None and hasattr(vae, "eval"):
        vae.eval()
    chunks = []
    try:
        with torch.no_grad():
            for start in range(0, len(xs), bsz):
                z, _loss = forward(xs[start : start + bsz])
                z = z.detach().cpu().float()
                if z.dim() == 1:
                    z = z.unsqueeze(0)
                chunks.append(z)
    finally:
        if was_training and vae is not None and hasattr(vae, "train"):
            vae.train()
    if not chunks:
        return 0
    new_z = torch.cat(chunks, dim=0).reshape(len(xs), -1)
    old_flat = old_z.detach().cpu().float().reshape(int(old_z.shape[0]), -1)
    feat = int(old_flat.shape[-1])
    if int(new_z.shape[-1]) > feat:
        new_z = new_z[:, :feat]
    elif int(new_z.shape[-1]) < feat:
        pad = torch.zeros(new_z.shape[0], feat - int(new_z.shape[-1]), dtype=new_z.dtype)
        new_z = torch.cat([new_z, pad], dim=-1)
    if old_z.dim() == 2:
        state.train_z = new_z.to(dtype=old_z.dtype)
    else:
        try:
            state.train_z = new_z.reshape(tuple(old_z.shape)).to(dtype=old_z.dtype)
        except RuntimeError:
            state.train_z = new_z.to(dtype=old_z.dtype)

    top_xs = list(getattr(state, "top_k_xs", None) or [])
    if top_xs:
        index = {_artifact_key(x): i for i, x in enumerate(xs)}
        tz = state.train_z.detach().cpu()
        new_top = []
        for x in top_xs:
            i = index.get(_artifact_key(x))
            if i is None:
                continue
            row = tz[i]
            new_top.append(row.reshape(-1).unsqueeze(-2))
        if new_top:
            state.top_k_zs = new_top
    return int(len(xs))


def refit_lolbo_surrogate_on_all(state: Any) -> None:
    """Retrain the GP on every remapped (z, y) after an encode-only recenter."""
    if not hasattr(state, "update_surrogate_model"):
        return
    saved_epochs = getattr(state, "init_n_epochs", None)
    n_fast = max(1, int(getattr(state, "num_update_epochs", None) or 2))
    state.initial_model_training_complete = False
    if saved_epochs is not None:
        state.init_n_epochs = n_fast
    try:
        state.update_surrogate_model()
    finally:
        if saved_epochs is not None:
            state.init_n_epochs = saved_epochs


def reencode_observation_latents(
    vae: Any,
    observations: Sequence[Dict[str, Any]],
    *,
    latent_dim: int,
) -> int:
    """Re-encode peptide artifacts into z_* configs after VAE finetune (recenter)."""
    encode = getattr(vae, "encode", None)
    if encode is None:
        return 0
    n_ok = 0
    for row in observations:
        seq = _obs_sequence(row)
        if not seq:
            continue
        try:
            z = encode(seq)
        except Exception:
            continue
        row["config"] = z_to_config(z)
        n_ok += 1
    return n_ok
