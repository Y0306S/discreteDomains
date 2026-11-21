"""APEXGo paper-faithful objective, constraints, and diverse-set selection."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

APEXGO_OPT = Path("/root/data/cys/APEXGo/optimization")


def _load_apex_objective(score_version: str = "gramnegonly_mean"):
    if str(APEXGO_OPT) not in sys.path:
        sys.path.insert(0, str(APEXGO_OPT))
    from your_tasks.your_objective_functions import ApexObjective  # type: ignore

    return ApexObjective(score_version=score_version)


def _levenshtein(a: str, b: str) -> int:
    try:
        from Levenshtein import distance as lev  # type: ignore

        return int(lev(a, b))
    except Exception:
        la, lb = len(a), len(b)
        dp = list(range(lb + 1))
        for i, cha in enumerate(a, 1):
            prev, dp[0] = dp[0], i
            for j, chb in enumerate(b, 1):
                cur = dp[j]
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (cha != chb))
                prev = cur
        return dp[lb]


def template_similarity(seq: str, template: str) -> float:
    """Paper: (l - d) / l with l = template length and d = Levenshtein distance."""
    template = str(template or "")
    if not template:
        return 0.0
    return (len(template) - _levenshtein(seq, template)) / len(template)


def passes_similarity_constraint(seq: str, template: str, *, min_similarity: float = 0.75) -> bool:
    return template_similarity(seq, template) >= float(min_similarity)


def gramneg_mean_mic(mic11: np.ndarray) -> float:
    return float(np.mean(np.asarray(mic11, dtype=float).reshape(11)[:7]))


def all_mean_mic(mic11: np.ndarray) -> float:
    return float(np.mean(np.asarray(mic11, dtype=float).reshape(11)))


def log2_fold_change(new_mic: float, old_mic: float, *, eps: float = 1e-6) -> float:
    return float(np.log2((float(new_mic) + eps) / (float(old_mic) + eps)))


def select_diverse_set(
    candidates: Sequence[Tuple[str, float]],
    *,
    m: int = 20,
    tau: int = 1,
) -> List[Tuple[str, float]]:
    """Greedy diverse top-M selection used by APEXGo ROBOT (edit distance >= tau)."""
    ranked = sorted(candidates, key=lambda x: x[1], reverse=True)
    picked: List[Tuple[str, float]] = []
    for seq, score in ranked:
        if any(_levenshtein(seq, prev) < int(tau) for prev, _ in picked):
            continue
        picked.append((seq, score))
        if len(picked) >= int(m):
            break
    return picked


def score_sequences(
    seqs: List[str],
    *,
    objective=None,
    apex_wrapper=None,
    batch_size: int = 64,
) -> Dict[str, dict]:
    """Return per-sequence APEXGo and MIC summaries."""
    if objective is None:
        objective = _load_apex_objective()
    if apex_wrapper is None:
        apex_dir = APEXGO_OPT / "apex_oracle"
        if str(apex_dir) not in sys.path:
            sys.path.insert(0, str(apex_dir))
        from APEX_predict import apex_wrapper as _aw  # type: ignore

        apex_wrapper = _aw

    trimmed = [str(s or "")[:150] for s in seqs]
    mic_rows: List[np.ndarray] = []
    for i in range(0, len(trimmed), int(batch_size)):
        chunk = trimmed[i : i + int(batch_size)]
        scores = apex_wrapper(chunk)
        for row in scores:
            mic_rows.append(np.asarray(row, dtype=float).reshape(11))

    apexgo_scores = objective.query_black_box(trimmed)
    out: Dict[str, dict] = {}
    for seq, mic, apexgo in zip(seqs, mic_rows, apexgo_scores):
        out[seq] = {
            "apexgo_score": float(apexgo),
            "gramneg_mean_mic": gramneg_mean_mic(mic),
            "all_mean_mic": all_mean_mic(mic),
            "mic11": mic.tolist(),
        }
    return out
