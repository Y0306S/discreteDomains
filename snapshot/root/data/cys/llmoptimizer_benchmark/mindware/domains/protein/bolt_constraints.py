"""BOLT seed_0 peptide constraints (similarity + bacteria_0), shared by TPE and LOLBO."""

from __future__ import annotations

import ast
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from mindware.domains.paths import BOLT_PEPTIDES

_AA = set("ACDEFGHIKLMNPQRSTVWY")
_REF_CACHE: Optional[List[str]] = None


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
    template = str(template or "")
    if not template:
        return 0.0
    return (len(template) - _levenshtein(seq, template)) / len(template)


def ensure_one_cysteine(seq: str) -> str:
    while seq.count("C") > 1:
        idx = seq.index("C")
        for i, ch in enumerate(seq):
            if ch == "C" and i != idx:
                return seq[:i] + "A" + seq[i + 1 :]
    return seq


def load_reference_sequences(path: Optional[Path] = None) -> List[str]:
    """Load BOLT REFERENCE_SEQUENCE list from apex_oracle/refseqs.py."""
    global _REF_CACHE
    if _REF_CACHE is not None:
        return _REF_CACHE
    ref_path = Path(path) if path is not None else (BOLT_PEPTIDES / "apex_oracle/refseqs.py")
    text = ref_path.read_text(encoding="utf-8")
    match = re.search(r"REFERENCE_SEQUENCE\s*=\s*(\[.*?\])\s*\n\s*ALL_AMINO_ACIDS", text, re.S)
    if not match:
        raise RuntimeError(f"Could not parse REFERENCE_SEQUENCE from {ref_path}")
    refs = ast.literal_eval(match.group(1))
    _REF_CACHE = [str(x) for x in refs]
    return _REF_CACHE


def bolt_reference_sequence(*, constraint_index: int = 0) -> str:
    """BOLT seed_0 uses constraint_types[0] as index into REFERENCE_SEQUENCE."""
    refs = load_reference_sequences()
    idx = int(constraint_index)
    if idx < 0 or idx >= len(refs):
        raise IndexError(f"constraint_index {idx} out of range (n={len(refs)})")
    return refs[idx]


def constraint_value(seq: str, reference: str, *, min_similarity: float = 0.75) -> float:
    """BOLT form c(x) <= 0: threshold - similarity."""
    return float(min_similarity) - template_similarity(seq, reference)


def is_feasible(seq: str, reference: str, *, min_similarity: float = 0.75) -> bool:
    return constraint_value(seq, reference, min_similarity=min_similarity) <= 0.0


def _align_length(seq: str, reference: str) -> str:
    seq = ensure_one_cysteine(str(seq or "").strip().upper())
    ref = str(reference or "")
    if not ref:
        return seq
    if len(seq) < len(ref):
        seq = seq + ref[len(seq) :]
    elif len(seq) > len(ref):
        seq = seq[: len(ref)]
    return "".join(ch if ch in _AA else "A" for ch in seq)


def repair_to_feasible(
    seq: str,
    reference: str,
    *,
    min_similarity: float = 0.75,
    max_steps: int = 64,
) -> str:
    """Greedy edit toward reference until similarity threshold is met."""
    seq = _align_length(seq, reference)
    ref = str(reference)
    if is_feasible(seq, ref, min_similarity=min_similarity):
        return seq
    for _ in range(int(max_steps)):
        if is_feasible(seq, ref, min_similarity=min_similarity):
            return seq
        best_i = None
        best_gain = 0.0
        cur = template_similarity(seq, ref)
        for i in range(len(ref)):
            if seq[i] == ref[i]:
                continue
            trial = seq[:i] + ref[i] + seq[i + 1 :]
            gain = template_similarity(trial, ref) - cur
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i is None:
            break
        seq = seq[:best_i] + ref[best_i] + seq[best_i + 1 :]
    return ensure_one_cysteine(seq)


def random_feasible_mutant(
    base: str,
    reference: str,
    *,
    min_similarity: float = 0.75,
    rng: Optional[random.Random] = None,
    max_edits: int = 3,
) -> str:
    """Single-point (or few-point) mutant of base that stays feasible w.r.t. reference."""
    rng = rng or random.Random(0)
    base = _align_length(base, reference)
    ref = str(reference)
    if not is_feasible(base, ref, min_similarity=min_similarity):
        base = repair_to_feasible(base, ref, min_similarity=min_similarity)
    n_edits = rng.randint(1, max(1, int(max_edits)))
    seq = list(base)
    alphabet = sorted(_AA)
    for _ in range(n_edits):
        if not is_feasible("".join(seq), ref, min_similarity=min_similarity):
            seq = list(repair_to_feasible("".join(seq), ref, min_similarity=min_similarity))
        i = rng.randrange(len(seq))
        choices = [aa for aa in alphabet if aa != seq[i]]
        seq[i] = rng.choice(choices)
        if not is_feasible("".join(seq), ref, min_similarity=min_similarity):
            seq[i] = ref[i]
    out = ensure_one_cysteine("".join(seq))
    if not is_feasible(out, ref, min_similarity=min_similarity):
        out = repair_to_feasible(out, ref, min_similarity=min_similarity)
    return out


def propose_bolt_feasible_seq(
    *,
    decoded: str,
    reference: str,
    anchor: str,
    best: Optional[str],
    min_similarity: float = 0.75,
    rng: Optional[random.Random] = None,
    seen: Optional[Set[str]] = None,
    max_novel_attempts: int = 64,
) -> str:
    """Map a raw VAE decode into a BOLT-feasible candidate sequence.

    If ``seen`` is set and the repaired sequence was already evaluated, sample a
    novel feasible mutant instead (when possible).
    """
    rng = rng or random.Random(0)
    ref = str(reference)
    seq = _align_length(decoded, ref)
    if is_feasible(seq, ref, min_similarity=min_similarity):
        candidate = ensure_one_cysteine(seq)
    else:
        repaired = repair_to_feasible(seq, ref, min_similarity=min_similarity)
        if is_feasible(repaired, ref, min_similarity=min_similarity):
            candidate = repaired
        else:
            base = best or anchor or ref
            candidate = random_feasible_mutant(
                base,
                ref,
                min_similarity=min_similarity,
                rng=rng,
                max_edits=3,
            )

    if not seen or candidate not in seen:
        return candidate

    bases = [b for b in (best, anchor, ref, candidate) if b]
    for attempt in range(int(max_novel_attempts)):
        base = bases[attempt % len(bases)]
        max_edits = 1 + (attempt % 6)
        cand = random_feasible_mutant(
            base,
            ref,
            min_similarity=min_similarity,
            rng=rng,
            max_edits=max_edits,
        )
        if cand not in seen:
            return cand
    return candidate


def feasible_startup_sequences(
    start_seq: str,
    reference: str,
    *,
    min_similarity: float = 0.75,
    n_extra: int = 10,
    seed: int = 0,
) -> List[str]:
    """Startup pool: task init + random feasible mutants (BOLT init-data analogue)."""
    rng = random.Random(int(seed))
    ref = str(reference)
    start = _align_length(start_seq, ref)
    if not is_feasible(start, ref, min_similarity=min_similarity):
        start = repair_to_feasible(start, ref, min_similarity=min_similarity)
    out = [start]
    seen = {start}
    attempts = 0
    while len(out) < int(n_extra) + 1 and attempts < max(200, int(n_extra) * 100):
        attempts += 1
        cand = random_feasible_mutant(start, ref, min_similarity=min_similarity, rng=rng, max_edits=4)
        if cand in seen:
            continue
        seen.add(cand)
        out.append(cand)
    return out


def seed_bolt_startup_observations(
    adapter: Any,
    vae: Any,
    *,
    start_seq: str,
    reference: str,
    min_similarity: float = 0.75,
    n_extra: int = 10,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Pre-evaluate feasible startup sequences before trust-region TPE."""
    from mindware.domains.base import z_to_config
    from mindware.domains.history import observation

    seqs = feasible_startup_sequences(
        start_seq,
        reference,
        min_similarity=min_similarity,
        n_extra=n_extra,
        seed=seed,
    )
    obs: List[Dict[str, Any]] = []
    for seq in seqs:
        try:
            z = vae.encode(seq)
        except Exception:
            continue
        cfg = z_to_config(z)
        res = adapter.evaluate(seq)
        if not res.feasible:
            continue
        obs.append(
            observation(
                cfg,
                float(res.objective),
                trial_state=0,
                elapsed_time=0.0,
                extra={
                    "source": "bolt_startup",
                    "artifact": seq,
                    "info": dict(res.info or {}),
                },
            )
        )
    return obs
