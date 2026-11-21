"""Protein oracles: DMS lookup, APEX MIC, optional ESMFold mean pLDDT."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


def load_seq_score_pairs(seq_path: Path, score_path: Path) -> Dict[str, float]:
    seqs = [
        line.strip()
        for line in Path(seq_path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(">")
    ]
    scores = []
    for line in Path(score_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            scores.append(float(line.split(",")[0]))
        except ValueError:
            continue
    n = min(len(seqs), len(scores))
    return {seqs[i]: scores[i] for i in range(n)}


def load_dms_json(path: Path) -> Dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "scores" in payload:
        payload = payload["scores"]
    return {str(k): float(v) for k, v in dict(payload).items()}


def hash_score(seq: str) -> float:
    acc = 0.0
    for i, c in enumerate(seq):
        acc += (ord(c) * (i + 3) % 97) * 0.11
    return float(acc)


class LookupOracle:
    """Map sequence → score. Missing keys fall back to a deterministic hash."""

    def __init__(self, table: Optional[Dict[str, float]] = None, *, minimize: bool = True):
        self.table = dict(table or {})
        self.minimize = bool(minimize)

    def __call__(self, seq: str) -> Tuple[float, bool]:
        key = str(seq).strip()
        if key in self.table:
            raw = float(self.table[key])
            obj = raw if self.minimize else -raw
            return float(obj), True
        dummy = hash_score(key)
        return float(dummy), False

    def __len__(self) -> int:
        return len(self.table)


class ESMFoldOracle:
    """Live ESMFold mean pLDDT. Cached. Infeasible (1e6) on fold failure.

    Runs on the process CUDA device (set CUDA_VISIBLE_DEVICES=0).
    Higher pLDDT is better; BO minimizes, so we return -mean_plddt.
    """

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        *,
        device: str = "cuda:0",
        timeout_s: float = 120.0,
    ):
        self.cache_path = Path(cache_path) if cache_path else None
        self.device = str(device)
        self.timeout_s = float(timeout_s)
        self._cache: Dict[str, float] = {}
        self._model = None
        self._tokenizer = None
        if self.cache_path and self.cache_path.is_file():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    self._cache[str(row["seq"])] = float(row["score"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, EsmForProteinFolding

        model_id = os.environ.get("ESMFOLD_MODEL", "facebook/esmfold_v1")
        os.environ.setdefault("HF_ENDPOINT", os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = EsmForProteinFolding.from_pretrained(model_id)
        dev = self.device if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(dev).eval()
        self.device = dev

    def _append_cache(self, seq: str, score: float) -> None:
        self._cache[seq] = float(score)
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"seq": seq, "score": float(score)}) + "\n")

    def __call__(self, seq: str) -> Tuple[float, bool]:
        key = str(seq or "").strip().upper()
        key = "".join(c for c in key if c in "ACDEFGHIKLMNPQRSTVWY")
        if not key:
            return 1e6, False
        if key in self._cache:
            return float(self._cache[key]), True
        try:
            import torch

            self._load()
            tok = self._tokenizer([key], return_tensors="pt", add_special_tokens=False)
            tok = {k: v.to(self.device) for k, v in tok.items()}
            with torch.no_grad():
                out = self._model(**tok)
            plddt = out.plddt
            mean_plddt = float(plddt.mean().detach().cpu())
            if mean_plddt <= 1.5:
                mean_plddt *= 100.0
            obj = -mean_plddt
            self._append_cache(key, obj)
            return float(obj), True
        except Exception:
            return 1e6, False
