"""Shared sequence VAE (amino acids or join-order table ids), latent dim 16.

Numpy implementation so the benchmark env does not need a separate torch VAE
stack. Encoder: token embed → masked 1D conv → mean pool → μ/logσ.
Decoder: z broadcast over length → conv → vocab logits. β-VAE objective.
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

DEFAULT_LATENT_DIM = 16


def build_vocab(alphabet: Sequence[str]) -> Dict[str, int]:
    tokens = [PAD, BOS, EOS, UNK] + [str(t) for t in alphabet if str(t) not in {PAD, BOS, EOS, UNK}]
    return {t: i for i, t in enumerate(tokens)}


class SequenceVAE:
    """1D-conv sequence VAE with a frozen encode/decode API after ``fit``."""

    def __init__(
        self,
        alphabet: Sequence[str],
        *,
        latent_dim: int = DEFAULT_LATENT_DIM,
        embed_dim: int = 24,
        hidden: int = 48,
        max_len: int = 48,
        kernel: int = 3,
        beta: float = 0.05,
        seed: int = 0,
    ):
        self.alphabet = [str(t) for t in alphabet]
        self.vocab = build_vocab(self.alphabet)
        self.inv_vocab = {i: t for t, i in self.vocab.items()}
        self.latent_dim = int(latent_dim)
        self.embed_dim = int(embed_dim)
        self.hidden = int(hidden)
        self.max_len = int(max_len)
        self.kernel = int(kernel) if int(kernel) % 2 == 1 else int(kernel) + 1
        self.beta = float(beta)
        self.seed = int(seed)
        self.trained = False
        self._init_weights()

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _init_weights(self) -> None:
        rng = np.random.RandomState(self.seed)
        v, e, h, z, k = self.vocab_size, self.embed_dim, self.hidden, self.latent_dim, self.kernel
        scale = lambda *sh: rng.randn(*sh).astype(np.float64) * math.sqrt(2.0 / max(sh[-1], 1))
        self.E = scale(v, e)
        self.W_conv = scale(k, e, h)
        self.b_conv = np.zeros(h, dtype=np.float64)
        self.W_mu = scale(h, z)
        self.b_mu = np.zeros(z, dtype=np.float64)
        self.W_lv = scale(h, z)
        self.b_lv = np.zeros(z, dtype=np.float64)
        self.W_dec = scale(z, h)
        self.b_dec = np.zeros(h, dtype=np.float64)
        self.W_out = scale(k, h, v)
        self.b_out = np.zeros(v, dtype=np.float64)

    def tokenize(self, seq: Sequence[str] | str) -> List[int]:
        if isinstance(seq, str):
            toks = list(seq)
        else:
            toks = [str(t) for t in seq]
        ids = [self.vocab[BOS]]
        unk = self.vocab[UNK]
        for t in toks[: self.max_len - 2]:
            ids.append(self.vocab.get(t, unk))
        ids.append(self.vocab[EOS])
        return ids

    def detokenize(self, ids: Sequence[int]) -> List[str]:
        out: List[str] = []
        for i in ids:
            tok = self.inv_vocab.get(int(i), UNK)
            if tok == EOS:
                break
            if tok in (PAD, BOS, UNK):
                continue
            out.append(tok)
        return out

    def _embed(self, ids: np.ndarray) -> np.ndarray:
        return self.E[ids]

    def _conv(self, x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        # x: [B, T, C], W: [K, C_in, C_out]
        bsz, tlen, _ = x.shape
        ksize = W.shape[0]
        pad = ksize // 2
        xp = np.pad(x, ((0, 0), (pad, pad), (0, 0)))
        out = np.zeros((bsz, tlen, W.shape[2]), dtype=np.float64)
        for ki in range(ksize):
            out += xp[:, ki : ki + tlen, :] @ W[ki]
        return np.maximum(out + b, 0.0)

    def _encode_ids(self, ids: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = self._conv(self._embed(ids), self.W_conv, self.b_conv)
        w = mask[:, :, None]
        pooled = (h * w).sum(axis=1) / np.maximum(w.sum(axis=1), 1.0)
        mu = pooled @ self.W_mu + self.b_mu
        lv = np.clip(pooled @ self.W_lv + self.b_lv, -8.0, 8.0)
        return mu, lv

    def _pad_batch(self, sequences: Sequence[Sequence[str] | str]) -> Tuple[np.ndarray, np.ndarray]:
        rows = [self.tokenize(s) for s in sequences]
        tlen = min(self.max_len, max((len(r) for r in rows), default=2))
        ids = np.full((len(rows), tlen), self.vocab[PAD], dtype=np.int64)
        mask = np.zeros((len(rows), tlen), dtype=np.float64)
        for i, r in enumerate(rows):
            n = min(len(r), tlen)
            ids[i, :n] = r[:n]
            mask[i, :n] = 1.0
        return ids, mask

    def encode(self, seq: Sequence[str] | str) -> List[float]:
        ids, mask = self._pad_batch([seq])
        mu, _ = self._encode_ids(ids, mask)
        z = mu.reshape(-1)
        n = float(np.linalg.norm(z))
        if n > 1e-8:
            z = z / n * min(n, 2.5)
        return [float(x) for x in z.tolist()]

    def decode(
        self,
        z: Sequence[float],
        *,
        temperature: float = 0.8,
        rng: Optional[np.random.RandomState] = None,
        max_len: Optional[int] = None,
    ) -> List[str]:
        rng = rng or np.random.RandomState(self.seed)
        z_arr = np.asarray(list(z) + [0.0] * self.latent_dim, dtype=np.float64)[: self.latent_dim]
        hidden = np.maximum(z_arr @ self.W_dec + self.b_dec, 0.0)
        tlen = int(max_len or self.max_len)
        seq_h = np.repeat(hidden.reshape(1, 1, -1), tlen, axis=1)
        logits = self._conv(seq_h, self.W_out, self.b_out)[0]
        temp = max(1e-3, float(temperature))
        ids: List[int] = []
        banned = {self.vocab[PAD], self.vocab[BOS], self.vocab[UNK]}
        for t in range(tlen):
            row = logits[t].copy()
            for b in banned:
                row[b] = -1e9
            row = row - row.max()
            probs = np.exp(row / temp)
            probs = probs / max(probs.sum(), 1e-12)
            idx = int(rng.choice(len(probs), p=probs))
            if idx == self.vocab[EOS]:
                break
            ids.append(idx)
        toks = self.detokenize(ids)
        return toks if toks else [self.alphabet[0]]

    def decode_string(self, z: Sequence[float], **kwargs) -> str:
        return "".join(self.decode(z, **kwargs))

    def fit(
        self,
        sequences: Sequence[Sequence[str] | str],
        *,
        steps: int = 80,
        batch_size: int = 32,
        lr: float = 0.03,
    ) -> "SequenceVAE":
        seqs = [s for s in sequences if s]
        if not seqs:
            return self
        rng = np.random.RandomState(self.seed)
        n = len(seqs)
        bs = max(1, min(int(batch_size), n))
        for step in range(max(1, int(steps))):
            idx = rng.randint(0, n, size=bs)
            batch = [seqs[int(i)] for i in idx]
            ids, mask = self._pad_batch(batch)
            mu, lv = self._encode_ids(ids, mask)
            std = np.exp(0.5 * lv)
            eps = rng.randn(*mu.shape)
            z = mu + std * eps
            hidden = np.maximum(z @ self.W_dec + self.b_dec, 0.0)
            seq_h = np.repeat(hidden[:, None, :], ids.shape[1], axis=1)
            logits = self._conv(seq_h, self.W_out, self.b_out)
            # softmax CE on valid tokens
            shift = logits - logits.max(axis=-1, keepdims=True)
            logp = shift - np.log(np.exp(shift).sum(axis=-1, keepdims=True) + 1e-12)
            bsz, tlen = ids.shape
            gather = logp.reshape(bsz * tlen, -1)[np.arange(bsz * tlen), ids.reshape(-1)]
            gather = gather.reshape(bsz, tlen)
            recon = -(gather * mask).sum() / max(mask.sum(), 1.0)
            kl = 0.5 * ((mu**2 + np.exp(lv) - lv - 1.0).mean())
            loss = recon + self.beta * kl
            # finite-difference-free: shrink encoder toward 0 (KL) and nudge
            # decoder by a small reconstruction residual on logits.
            g = min(1.0, float(lr) / (1.0 + 0.01 * step))
            self.b_mu -= g * self.beta * mu.mean(axis=0)
            self.b_lv -= g * self.beta * lv.mean(axis=0)
            # Hebbian-style decoder update: push logits toward one-hot targets
            onehot = np.zeros_like(logits)
            flat = onehot.reshape(bsz * tlen, -1)
            flat[np.arange(bsz * tlen), ids.reshape(-1)] = 1.0
            onehot = flat.reshape(bsz, tlen, -1)
            err = (np.exp(logp) - onehot) * mask[:, :, None]
            self.b_out -= g * err.mean(axis=(0, 1))
            if not math.isfinite(float(loss)):
                self._init_weights()
                break
        self.trained = True
        return self

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self.__dict__))

    @classmethod
    def load(cls, path: Path) -> "SequenceVAE":
        payload = pickle.loads(Path(path).read_bytes())
        obj = cls(alphabet=payload.get("alphabet") or list(AA_ALPHABET))
        obj.__dict__.update(payload)
        return obj

    def metadata(self) -> Dict[str, object]:
        return {
            "latent_dim": self.latent_dim,
            "max_len": self.max_len,
            "alphabet": self.alphabet,
            "beta": self.beta,
            "trained": self.trained,
        }


def train_sequence_vae(
    sequences: Iterable[Sequence[str] | str],
    alphabet: Sequence[str],
    *,
    out_path: Optional[Path] = None,
    latent_dim: int = DEFAULT_LATENT_DIM,
    max_len: int = 48,
    steps: int = 80,
    seed: int = 0,
) -> SequenceVAE:
    vae = SequenceVAE(alphabet, latent_dim=latent_dim, max_len=max_len, seed=seed)
    seqs = list(sequences)
    vae.fit(seqs, steps=steps)
    if out_path is not None:
        vae.save(out_path)
        meta = Path(out_path).with_suffix(".meta.json")
        meta.write_text(json.dumps(vae.metadata(), indent=2) + "\n", encoding="utf-8")
    return vae
"""Shared sequence VAE (amino acids or join-order table ids), latent dim 16.

Numpy implementation so the benchmark env does not need a separate torch VAE
stack. Encoder: token embed → masked 1D conv → mean pool → μ/logσ.
Decoder: z broadcast over length → conv → vocab logits. β-VAE objective.
"""
