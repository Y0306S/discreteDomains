"""ESM-2 frozen embeddings + MLP fitness head for ProteinGym landscapes."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class ESMFitnessSurrogate:
    """Maximize predicted DMS fitness. BO minimizes → return -pred.

    Unknown / failed encode → large penalty (1e6) and hit=False.
    """

    def __init__(
        self,
        *,
        model_id: str = "facebook/esm2_t12_35M_UR50D",
        device: str = "cuda:0",
        cache_path: Optional[Path] = None,
        head_path: Optional[Path] = None,
        max_len: int = 1024,
    ):
        self.model_id = model_id
        self.device = device
        self.cache_path = Path(cache_path) if cache_path else None
        self.head_path = Path(head_path) if head_path else None
        self.max_len = int(max_len)
        self._tokenizer = None
        self._model = None
        self._head = None
        self._dim: Optional[int] = None
        self._cache: Dict[str, np.ndarray] = {}
        if self.cache_path and self.cache_path.is_file():
            self._load_cache()

    def _load_cache(self) -> None:
        import torch

        blob = torch.load(self.cache_path, map_location="cpu", weights_only=False)
        for k, v in dict(blob.get("embeddings") or {}).items():
            self._cache[str(k)] = np.asarray(v, dtype=np.float32)
        self._dim = int(blob.get("dim") or (next(iter(self._cache.values())).shape[0] if self._cache else 0))

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import os
        import torch

        # huggingface.co is often unreachable here; prefer mirror + local snapshot.
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")
        # Import after endpoint env so hub client picks up the mirror.
        from transformers import AutoTokenizer, EsmModel

        load_id = self.model_id
        local = os.environ.get("PROTEINGYM_ESM_LOCAL")
        if not local:
            # Auto-detect HF hub snapshot for offline loads.
            hub = Path.home() / ".cache/huggingface/hub/models--facebook--esm2_t12_35M_UR50D/snapshots"
            if hub.is_dir():
                snaps = sorted(hub.iterdir())
                if snaps:
                    local = str(snaps[-1])
        if local and Path(local).is_dir():
            load_id = local
        self._tokenizer = AutoTokenizer.from_pretrained(load_id, local_files_only=True)
        self._model = EsmModel.from_pretrained(load_id, local_files_only=True)
        self._model = self._model.to(self.device).eval()
        self._dim = int(self._model.config.hidden_size)

    def _ensure_head(self) -> None:
        if self._head is not None:
            return
        import torch
        import torch.nn as nn

        if self._dim is None:
            self._ensure_model()
        assert self._dim is not None
        head = nn.Sequential(
            nn.Linear(self._dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        if self.head_path and self.head_path.is_file():
            state = torch.load(self.head_path, map_location=self.device, weights_only=False)
            head.load_state_dict(state["state_dict"])
            self._dim = int(state.get("dim", self._dim))
        self._head = head.to(self.device).eval()

    def embed_batch(self, seqs: Sequence[str], batch_size: int = 8) -> List[np.ndarray]:
        import torch

        self._ensure_model()
        out_vecs: List[Optional[np.ndarray]] = [None] * len(seqs)
        missing = []
        for i, seq in enumerate(seqs):
            key = "".join(c for c in str(seq).upper() if c in "ACDEFGHIKLMNPQRSTVWY")
            if key in self._cache:
                out_vecs[i] = self._cache[key]
            else:
                missing.append((i, key))
        for s in range(0, len(missing), batch_size):
            chunk = missing[s : s + batch_size]
            keys = [k for _, k in chunk]
            with torch.no_grad():
                tok = self._tokenizer(
                    keys,
                    return_tensors="pt",
                    truncation=True,
                    max_length=min(self.max_len, 1022),
                    padding=True,
                    add_special_tokens=True,
                )
                tok = {k: v.to(self.device) for k, v in tok.items()}
                hidden = self._model(**tok).last_hidden_state
                mask = tok["attention_mask"].bool()
                for j, (idx, key) in enumerate(chunk):
                    m = mask[j]
                    vec = hidden[j, m].mean(0).detach().cpu().numpy().astype(np.float32)
                    self._cache[key] = vec
                    out_vecs[idx] = vec
        return [v if v is not None else np.zeros(self._dim or 480, dtype=np.float32) for v in out_vecs]

    def embed(self, seq: str) -> np.ndarray:
        return self.embed_batch([seq], batch_size=1)[0]

    def predict_raw(self, seq: str) -> float:
        import torch

        self._ensure_head()
        vec = self.embed(seq)
        with torch.no_grad():
            t = torch.tensor(vec, dtype=torch.float32, device=self.device).view(1, -1)
            pred = float(self._head(t).view(-1)[0].item())
        return pred

    def __call__(self, seq: str) -> Tuple[float, bool]:
        try:
            raw = self.predict_raw(seq)
            # BO minimizes; higher fitness better
            return float(-raw), True
        except Exception:
            return 1e6, False

    def save_cache(self, path: Optional[Path] = None) -> None:
        import torch

        path = Path(path or self.cache_path)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"embeddings": self._cache, "dim": self._dim, "model_id": self.model_id}, path)

    def save_head(self, path: Path) -> None:
        import torch

        self._ensure_head()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self._head.state_dict(), "dim": self._dim, "model_id": self.model_id}, path)


def load_scored_split(csv_path: Path, split: str) -> List[Tuple[str, float]]:
    rows = []
    with Path(csv_path).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split") == split:
                rows.append((row["sequence"], float(row["score"])))
    return rows


def train_surrogate_head(
    landscape_dir: Path,
    *,
    device: str = "cuda:0",
    model_id: str = "facebook/esm2_t12_35M_UR50D",
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> Dict:
    import torch
    import torch.nn as nn

    scored = landscape_dir / "sequences_scored.csv"
    # Cap embed cost for large landscapes while keeping quantile coverage.
    if len(train) > 12000:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(train), size=12000, replace=False)
        train = [train[int(i)] for i in idx]
    if len(val) > 3000:
        rng = np.random.RandomState(1)
        idx = rng.choice(len(val), size=3000, replace=False)
        val = [val[int(i)] for i in idx]
    if len(test) > 3000:
        rng = np.random.RandomState(2)
        idx = rng.choice(len(test), size=3000, replace=False)
        test = [test[int(i)] for i in idx]
        seqs = [s for s, _ in pairs]
        ys = np.asarray([score for _, score in pairs], dtype=np.float32)
        xs_list = []
        bs = 16 if max((len(s) for s in seqs), default=1) < 400 else 4
        for i in range(0, len(seqs), 512):
            chunk = seqs[i : i + 512]
            xs_list.extend(sur.embed_batch(chunk, batch_size=bs))
            print(json.dumps({"embed": tag, "done": min(i + 512, len(seqs)), "total": len(seqs)}))
        return np.stack(xs_list), ys

    print(json.dumps({"phase": "embed_train", "n": len(train)}))
    Xtr, ytr = embed_all(train, "train")
    print(json.dumps({"phase": "embed_val", "n": len(val)}))
    Xva, yva = embed_all(val, "val")
    print(json.dumps({"phase": "embed_test", "n": len(test)}))
    Xte, yte = embed_all(test, "test")
    sur.save_cache(cache_path)

    dim = Xtr.shape[1]
    head = nn.Sequential(
        nn.Linear(dim, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    def batches(X, y):
        n = len(X)
        idx = np.random.permutation(n)
        for s in range(0, n, batch_size):
            sl = idx[s : s + batch_size]
            yield (
                torch.tensor(X[sl], dtype=torch.float32, device=device),
                torch.tensor(y[sl], dtype=torch.float32, device=device).view(-1, 1),
            )

    best_val = float("inf")
    best_state = None
    for epoch in range(epochs):
        head.train()
        tr_loss = 0.0
        n_b = 0
        for xb, yb in batches(Xtr, ytr):
            pred = head(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item())
            n_b += 1
        head.eval()
        with torch.no_grad():
            pv = head(torch.tensor(Xva, dtype=torch.float32, device=device)).view(-1).cpu().numpy()
            pt = head(torch.tensor(Xte, dtype=torch.float32, device=device)).view(-1).cpu().numpy()
        val_mse = float(np.mean((pv - yva) ** 2))
        # spearman
        def spearman(a, b):
            ra = a.argsort().argsort().astype(np.float64)
            rb = b.argsort().argsort().astype(np.float64)
            ra -= ra.mean()
            rb -= rb.mean()
            return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12))

        metrics = {
            "epoch": epoch,
            "train_mse": tr_loss / max(n_b, 1),
            "val_mse": val_mse,
            "val_spearman": spearman(pv, yva),
            "test_spearman": spearman(pt, yte),
        }
        print(json.dumps(metrics))
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)
    torch.save({"state_dict": head.state_dict(), "dim": dim, "model_id": model_id}, head_path)
    report = {
        "landscape": landscape_dir.name,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "best_val_mse": best_val,
        "cache": str(cache_path),
        "head": str(head_path),
        "model_id": model_id,
    }
    (landscape_dir / "surrogate_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
