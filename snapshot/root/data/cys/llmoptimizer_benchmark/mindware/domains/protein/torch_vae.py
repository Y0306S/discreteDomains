"""PyTorch CNN sequence VAE for ProteinGym-length proteins (GFP/AAV).

Encode AA string ↔ z ∈ R^{latent_dim}. Compatible with PeptideBoltVAE-style
encode/decode_string API used by ProteinAdapter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"


def build_vocab() -> Dict[str, int]:
    tokens = [PAD, BOS, EOS, UNK] + AA_ALPHABET
    return {t: i for i, t in enumerate(tokens)}


def _import_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


class CNNSeqVAEModule:
    """Built lazily so importing this file does not require CUDA init."""

    @staticmethod
    def build(vocab_size: int, latent_dim: int, embed_dim: int, hidden: int, max_len: int):
        torch, nn, F = _import_torch()

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.vocab_size = vocab_size
                self.latent_dim = latent_dim
                self.max_len = max_len
                self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.enc = nn.Sequential(
                    nn.Conv1d(embed_dim, hidden, 5, padding=2),
                    nn.ReLU(),
                    nn.Conv1d(hidden, hidden, 5, padding=2),
                    nn.ReLU(),
                    nn.Conv1d(hidden, hidden, 5, padding=2),
                    nn.ReLU(),
                )
                self.fc_mu = nn.Linear(hidden, latent_dim)
                self.fc_lv = nn.Linear(hidden, latent_dim)
                self.fc_dec = nn.Linear(latent_dim, hidden)
                self.dec = nn.Sequential(
                    nn.Conv1d(hidden, hidden, 5, padding=2),
                    nn.ReLU(),
                    nn.Conv1d(hidden, hidden, 5, padding=2),
                    nn.ReLU(),
                )
                self.out = nn.Conv1d(hidden, vocab_size, 1)

            def encode(self, ids, mask):
                x = self.embed(ids).transpose(1, 2)
                h = self.enc(x)
                w = mask.unsqueeze(1)
                pooled = (h * w).sum(-1) / w.sum(-1).clamp_min(1.0)
                return self.fc_mu(pooled), self.fc_lv(pooled)

            def reparam(self, mu, lv):
                std = (0.5 * lv).exp()
                return mu + std * torch.randn_like(std)

            def decode_logits(self, z, tlen):
                h = self.fc_dec(z).unsqueeze(-1).expand(-1, -1, tlen)
                h = self.dec(h)
                return self.out(h).transpose(1, 2)

            def forward(self, ids, mask):
                mu, lv = self.encode(ids, mask)
                z = self.reparam(mu, lv.clamp(-8, 8))
                logits = self.decode_logits(z, ids.size(1))
                return logits, mu, lv

        return Module()


class ProteinGymVAE:
    """Landscape-specific sequence VAE wrapper."""

    def __init__(
        self,
        *,
        max_len: int = 238,
        latent_dim: int = 32,
        embed_dim: int = 64,
        hidden: int = 256,
        device: str = "cuda:0",
        beta: float = 0.01,
    ):
        torch, _, _ = _import_torch()
        self.vocab = build_vocab()
        self.inv_vocab = {i: t for t, i in self.vocab.items()}
        self.max_len = int(max_len)
        self.latent_dim = int(latent_dim)
        self.device = str(device)
        self.beta = float(beta)
        self.module = CNNSeqVAEModule.build(
            len(self.vocab), self.latent_dim, embed_dim, hidden, self.max_len
        ).to(self.device)
        self.module.eval()
        self.trained = False

    def tokenize_batch(self, sequences: Sequence[str]):
        torch, _, _ = _import_torch()
        rows = []
        for seq in sequences:
            s = "".join(c for c in str(seq).upper() if c in AA_ALPHABET)
            ids = [self.vocab[BOS]]
            for c in s[: self.max_len - 2]:
                ids.append(self.vocab.get(c, self.vocab[UNK]))
            ids.append(self.vocab[EOS])
            rows.append(ids)
        tlen = min(self.max_len, max((len(r) for r in rows), default=2))
        ids = torch.full((len(rows), tlen), self.vocab[PAD], dtype=torch.long)
        mask = torch.zeros((len(rows), tlen), dtype=torch.float32)
        for i, r in enumerate(rows):
            n = min(len(r), tlen)
            ids[i, :n] = torch.tensor(r[:n], dtype=torch.long)
            mask[i, :n] = 1.0
        return ids.to(self.device), mask.to(self.device)

    def encode(self, seq: Sequence[str] | str) -> List[float]:
        torch, _, _ = _import_torch()
        if not isinstance(seq, str):
            seq = "".join(str(t) for t in seq)
        self.module.eval()
        with torch.no_grad():
            ids, mask = self.tokenize_batch([seq])
            mu, _ = self.module.encode(ids, mask)
            z = mu[0].detach().cpu().tolist()
        return [float(x) for x in z]

    def decode_string(self, z: Sequence[float], *, temperature: float = 0.8, max_len: Optional[int] = None, **kwargs: Any) -> str:
        torch, _, F = _import_torch()
        self.module.eval()
        tlen = int(max_len or self.max_len)
        arr = [float(v) for v in z]
        if len(arr) < self.latent_dim:
            arr = arr + [0.0] * (self.latent_dim - len(arr))
        zt = torch.tensor(arr[: self.latent_dim], dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            logits = self.module.decode_logits(zt, tlen)[0]  # [T,V]
            banned = {self.vocab[PAD], self.vocab[BOS], self.vocab[UNK]}
            chars: List[str] = []
            for t in range(tlen):
                row = logits[t].clone()
                for b in banned:
                    row[b] = -1e9
                if temperature and temperature > 0:
                    probs = F.softmax(row / max(1e-3, float(temperature)), dim=-1)
                    idx = int(torch.multinomial(probs, 1).item())
                else:
                    idx = int(row.argmax().item())
                if idx == self.vocab[EOS]:
                    break
                tok = self.inv_vocab.get(idx, UNK)
                if tok in AA_ALPHABET:
                    chars.append(tok)
        return "".join(chars) if chars else "G" * min(12, tlen)

    def decode(self, z: Sequence[float], **kwargs: Any) -> List[str]:
        return list(self.decode_string(z, **kwargs))

    def save(self, path: Path, meta: Optional[Dict[str, Any]] = None) -> None:
        torch, _, _ = _import_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.module.state_dict(),
            "max_len": self.max_len,
            "latent_dim": self.latent_dim,
            "beta": self.beta,
            "trained": self.trained,
            "meta": meta or {},
        }
        torch.save(payload, path)
        path.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "max_len": self.max_len,
                    "latent_dim": self.latent_dim,
                    "beta": self.beta,
                    "trained": self.trained,
                    **(meta or {}),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path, device: str = "cuda:0") -> "ProteinGymVAE":
        torch, _, _ = _import_torch()
        payload = torch.load(path, map_location=device, weights_only=False)
        obj = cls(
            max_len=int(payload.get("max_len", 238)),
            latent_dim=int(payload.get("latent_dim", 32)),
            device=device,
            beta=float(payload.get("beta", 0.01)),
        )
        obj.module.load_state_dict(payload["state_dict"])
        obj.module.to(device).eval()
        obj.trained = bool(payload.get("trained", True))
        return obj


def train_proteingym_vae(
    sequences: Sequence[str],
    *,
    max_len: int,
    latent_dim: int = 32,
    device: str = "cuda:0",
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    beta: float = 0.01,
    out_path: Optional[Path] = None,
    seed: int = 0,
) -> ProteinGymVAE:
    torch, _, F = _import_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    vae = ProteinGymVAE(max_len=max_len, latent_dim=latent_dim, device=device, beta=beta)
    vae.module.train()
    opt = torch.optim.AdamW(vae.module.parameters(), lr=lr, weight_decay=1e-4)
    seqs = [s for s in sequences if s]
    n = len(seqs)
    if n == 0:
        raise ValueError("no sequences")
    bs = max(1, min(batch_size, n))
    steps_per = max(1, (n + bs - 1) // bs)
    for epoch in range(max(1, epochs)):
        order = np.random.permutation(n)
        total = 0.0
        count = 0
        for s in range(steps_per):
            idx = order[s * bs : (s + 1) * bs]
            batch = [seqs[int(i)] for i in idx]
            ids, mask = vae.tokenize_batch(batch)
            logits, mu, lv = vae.module(ids, mask)
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_ids = ids.reshape(-1)
            flat_mask = mask.reshape(-1)
            ce = F.cross_entropy(flat_logits, flat_ids, reduction="none")
            recon = (ce * flat_mask).sum() / flat_mask.sum().clamp_min(1.0)
            kl = 0.5 * (mu.pow(2) + lv.exp() - lv - 1.0).mean()
            loss = recon + beta * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.module.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            count += 1
            if s % max(1, steps_per // 5) == 0:
                print(json.dumps({"epoch": epoch, "step": s, "steps": steps_per, "loss": round(float(loss.item()), 5)}), flush=True)
        print(json.dumps({"epoch": epoch, "loss": round(total / max(count, 1), 5), "n": n}), flush=True)
    vae.module.eval()
    vae.trained = True
    if out_path is not None:
        vae.save(Path(out_path), meta={"n_train": n, "epochs": epochs})
    return vae
