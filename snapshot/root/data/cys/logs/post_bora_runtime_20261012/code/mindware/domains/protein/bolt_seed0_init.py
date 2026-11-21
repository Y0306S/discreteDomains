"""BOLT seed_0 init pool (seed_0_init.txt + seed_0_scores.csv) for TPE and LOLBO."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mindware.domains.base import z_to_config
from mindware.domains.history import observation
from mindware.domains.paths import BOLT_PEPTIDES

SEED0_INIT = BOLT_PEPTIDES / "apex_oracle/init_data/seed_0_init.txt"
SEED0_SCORES = BOLT_PEPTIDES / "apex_oracle/init_data/seed_0_scores.csv"

# Matches lolbo_scripts/run_single_seed_0.sh
BOLT_SEED0_HPARAMS: Dict[str, Any] = {
    "max_string_length": 30,
    "bsz": 50,
    "num_initialization_points": 1000,
    "init_n_update_epochs": 20,
    "num_update_epochs": 2,
    "e2e_freq": 10,
    "update_e2e": True,
    "k": 1000,
    "acq_func": "ts",
    "learning_rte": 0.001,
    "surrogate_type": "gp_dkl",
    "divf_id": "edit_dist",
    "constraint_function_ids": ["similarity"],
    "constraint_thresholds": [0.75],
    "constraint_types": [0],
    "task_specific_args": ["bacteria_0"],
    "minimize": False,
    "init_data_path": "apex_oracle/init_data/seed_0_init.txt",
    "init_scores_path": "apex_oracle/init_data/seed_0_scores.csv",
}

# BOLT run_single_seed_0.sh post-init oracle budget
BOLT_FULL_ORACLE_BUDGET = 200_000
CHECKPOINT_ORACLE_CALLS = 500


def load_seed0_pool(
    *,
    init_path: Optional[Path] = None,
    scores_path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Tuple[List[str], List[float]]:
    """Return (sequences, BOLT maximize scores) from seed_0 files."""
    init_path = Path(init_path) if init_path is not None else SEED0_INIT
    scores_path = Path(scores_path) if scores_path is not None else SEED0_SCORES
    if not init_path.is_file():
        raise FileNotFoundError(f"seed_0 init missing: {init_path}")
    if not scores_path.is_file():
        raise FileNotFoundError(f"seed_0 scores missing: {scores_path}")

    seqs = [
        ln.strip()
        for ln in init_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith(">")
    ]
    score_lines = [
        ln.strip()
        for ln in scores_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    scores = [float(x) for x in score_lines]
    n = min(len(seqs), len(scores))
    seqs, scores = seqs[:n], scores[:n]
    if limit is not None:
        n = min(int(limit), n)
        seqs, scores = seqs[:n], scores[:n]
    return seqs, scores


def bolt_maximize_score_to_mic(score: float) -> float:
    """BOLT CSV stores maximize objective (-MIC for bacteria_0)."""
    return -float(score)


def bolt_maximize_score_to_min_objective(score: float) -> float:
    """TPE path minimizes MIC."""
    return bolt_maximize_score_to_mic(score)


def encode_sequences_batch(vae: Any, seqs: Sequence[str], *, batch_size: int = 64) -> List[List[float]]:
    """Batch-encode peptide strings with PeptideBoltVAE."""
    import torch

    if not seqs:
        return []
    if hasattr(vae, "_vae_forward") and hasattr(vae, "dataobj") and hasattr(vae, "vae"):
        dim = int(getattr(vae, "latent_dim", 256))
        out: List[List[float]] = []
        for i in range(0, len(seqs), int(batch_size)):
            batch = [str(s) for s in seqs[i : i + int(batch_size)]]
            with torch.no_grad():
                z, _ = vae._vae_forward(batch, vae.dataobj, vae.vae)
            z = z.reshape(-1, dim).detach().cpu()
            out.extend(z[j].tolist() for j in range(z.shape[0]))
        return out
    out = []
    for seq in seqs:
        out.append(list(vae.encode(seq)))
    return out


def build_seed0_observations(
    vae: Any,
    *,
    limit: Optional[int] = None,
    batch_size: int = 64,
) -> List[Dict[str, Any]]:
    """Pre-scored startup observations for TPE (no oracle calls)."""
    seqs, scores = load_seed0_pool(limit=limit)
    zs = encode_sequences_batch(vae, seqs, batch_size=batch_size)
    obs: List[Dict[str, Any]] = []
    for seq, score, z in zip(seqs, scores, zs):
        mic = bolt_maximize_score_to_mic(score)
        obs.append(
            observation(
                z_to_config(z),
                float(mic),
                trial_state=0,
                elapsed_time=0.0,
                extra={
                    "source": "bolt_seed0_init",
                    "artifact": seq,
                    "info": {
                        "sequence": seq,
                        "oracle": "bolt_bacteria_0",
                        "raw_cost": float(mic),
                        "bacteria0_score": float(score),
                        "mic_bacteria_0": float(mic),
                        "precomputed": True,
                    },
                },
            )
        )
    return obs


def total_trial_budget(*, n_init_points: int, n_post_init_trials: int) -> int:
    return int(n_init_points) + int(n_post_init_trials)
