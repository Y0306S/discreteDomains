"""Semantic dataset-profile embeddings and a tiny shared-space VAE fallback.

The LLM embedding endpoint is used when reachable. Otherwise a deterministic
hash vector is stored so retrieval still has a cold cosine signal. VAE training
is TUNE-only; LOCK is held out until the encoder is frozen.
"""

from __future__ import annotations

import hashlib
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from mindware.components.experience.profile import cosine, profile_from_dataset_profile, profile_vector

LATENT_DIM = 32
HASH_DIM = 64
SEMANTIC_WEIGHT = 0.10


def profile_text(
    dataset_id: Optional[str],
    dataset_profile: Optional[Dict[str, Any]],
    task_type: Optional[str],
    algorithm_id: Optional[str] = None,
) -> str:
    """Build a short natural-language dataset profile for embedding."""
    try:
        from mindware.third_party.opro.utils import _build_dataset_summary_text

        summary = _build_dataset_summary_text({"dataset_profile": dataset_profile or {}})
    except Exception:
        summary = ""
    if not summary or summary.strip() in ("", "N/A"):
        vec = profile_from_dataset_profile(dataset_profile)
        bits = ", ".join(f"{k}={v:.3f}" for k, v in vec.items())
        summary = f"numeric profile: {bits}"
    algo = f" algorithm={algorithm_id}" if algorithm_id else ""
    return (
        f"dataset={dataset_id or 'unknown'} task_type={task_type or 'unknown'}{algo}. "
        f"{summary}"
    ).strip()


def hash_embed(text: str, dim: int = HASH_DIM) -> List[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big") % (2**32)
    rng = np.random.RandomState(seed)
    vec = rng.normal(size=int(dim)).astype(np.float64)
    n = float(np.linalg.norm(vec))
    if n > 1e-12:
        vec = vec / n
    return [float(x) for x in vec.tolist()]


def llm_embed(text: str, client=None, model: str = "text-embedding-3-small") -> Optional[List[float]]:
    if client is None:
        return None
    try:
        resp = client.embeddings.create(model=model, input=text)
        vec = list(resp.data[0].embedding)
        return [float(x) for x in vec]
    except Exception:
        return None


def embed_profile(
    dataset_id: Optional[str],
    dataset_profile: Optional[Dict[str, Any]],
    task_type: Optional[str],
    *,
    algorithm_id: Optional[str] = None,
    client=None,
    model: str = "text-embedding-3-small",
) -> List[float]:
    text = profile_text(dataset_id, dataset_profile, task_type, algorithm_id)
    vec = llm_embed(text, client=client, model=model)
    if vec:
        return vec
    return hash_embed(text)


class EmbeddingCache:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.data: Dict[str, List[float]] = {}
        if self.path and self.path.is_file():
            try:
                payload = pickle.loads(self.path.read_bytes())
                if isinstance(payload, dict):
                    self.data = {str(k): list(v) for k, v in payload.items()}
            except Exception:
                self.data = {}

    def key(self, task_type: str, dataset_id: str) -> str:
        return f"{task_type}::{dataset_id}".lower()

    def get(self, task_type: str, dataset_id: str) -> Optional[List[float]]:
        return self.data.get(self.key(task_type, dataset_id))

    def put(self, task_type: str, dataset_id: str, vec: Sequence[float]) -> None:
        self.data[self.key(task_type, dataset_id)] = [float(x) for x in vec]

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(pickle.dumps(self.data))


class TinyVAE:
    """One-hidden-layer VAE mapping variable-width vectors into ``LATENT_DIM``.

    Trained with reconstruction + KL. If too few rows, falls back to PCA / SVD.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, hidden: int = 64, seed: int = 0):
        self.latent_dim = int(latent_dim)
        self.hidden = int(hidden)
        self.seed = int(seed)
        self.input_dim: Optional[int] = None
        self.W1 = self.b1 = None
        self.W_mu = self.b_mu = None
        self.W_lv = self.b_lv = None
        self.W2 = self.b2 = None
        self.W3 = self.b3 = None
        self.pca_components: Optional[np.ndarray] = None
        self.pca_mean: Optional[np.ndarray] = None
        self.use_pca = False

    def _init_weights(self, d: int) -> None:
        rng = np.random.RandomState(self.seed)
        self.input_dim = d
        h, z = self.hidden, self.latent_dim
        scale = lambda a, b: rng.randn(a, b) * math.sqrt(2.0 / max(a, 1))
        self.W1, self.b1 = scale(d, h), np.zeros(h)
        self.W_mu, self.b_mu = scale(h, z), np.zeros(z)
        self.W_lv, self.b_lv = scale(h, z), np.zeros(z)
        self.W2, self.b2 = scale(z, h), np.zeros(h)
        self.W3, self.b3 = scale(h, d), np.zeros(d)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    def _encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = self._relu(x @ self.W1 + self.b1)
        mu = h @ self.W_mu + self.b_mu
        lv = np.clip(h @ self.W_lv + self.b_lv, -8.0, 8.0)
        return h, mu, lv

    def fit(self, rows: Sequence[Sequence[float]], steps: int = 200, lr: float = 0.05) -> "TinyVAE":
        X = [np.asarray(r, dtype=np.float64).reshape(-1) for r in rows if r is not None and len(r)]
        if not X:
            return self
        d = max(v.size for v in X)
        mat = np.zeros((len(X), d), dtype=np.float64)
        for i, v in enumerate(X):
            mat[i, : v.size] = v
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        mat = mat / norms
        self.input_dim = d
        if mat.shape[0] < 8:
            self._fit_pca(mat)
            return self
        self._init_weights(d)
        rng = np.random.RandomState(self.seed)
        for _ in range(max(1, int(steps))):
            h, mu, lv = self._encode(mat)
            std = np.exp(0.5 * lv)
            eps = rng.randn(*mu.shape)
            z = mu + std * eps
            h2 = self._relu(z @ self.W2 + self.b2)
            rec = h2 @ self.W3 + self.b3
            err = rec - mat
            recon = err / max(mat.shape[0], 1)
            # KL to N(0,1)
            kl_coef = 0.01
            # SGD on reconstruction only for the linear maps; KL shrinks mu/lv.
            self.W3 -= lr * (h2.T @ recon)
            self.b3 -= lr * recon.mean(axis=0)
            dh2 = recon @ self.W3.T
            dh2 *= (h2 > 0)
            self.W2 -= lr * (z.T @ dh2) / max(mat.shape[0], 1)
            self.b2 -= lr * dh2.mean(axis=0)
            self.W_mu -= lr * kl_coef * mu.mean(axis=0)
            self.W_lv -= lr * kl_coef * lv.mean(axis=0)
        self.use_pca = False
        return self

    def _fit_pca(self, mat: np.ndarray) -> None:
        self.use_pca = True
        self.pca_mean = mat.mean(axis=0)
        centered = mat - self.pca_mean
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            vt = np.eye(mat.shape[1])
        k = min(self.latent_dim, vt.shape[0])
        self.pca_components = vt[:k]
        self.input_dim = mat.shape[1]

    def encode(self, vec: Sequence[float]) -> List[float]:
        x = np.asarray(list(vec), dtype=np.float64).reshape(-1)
        d = int(self.input_dim or x.size or 1)
        padded = np.zeros(d, dtype=np.float64)
        padded[: min(d, x.size)] = x[: min(d, x.size)]
        n = float(np.linalg.norm(padded))
        if n > 1e-12:
            padded = padded / n
        if self.use_pca and self.pca_components is not None:
            z = (padded - (self.pca_mean if self.pca_mean is not None else 0.0)) @ self.pca_components.T
        elif self.W1 is not None:
            _, mu, _ = self._encode(padded.reshape(1, -1))
            z = mu.reshape(-1)
        else:
            z = padded[: self.latent_dim]
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        if z.size < self.latent_dim:
            z = np.pad(z, (0, self.latent_dim - z.size))
        z = z[: self.latent_dim]
        n = float(np.linalg.norm(z))
        if n > 1e-12:
            z = z / n
        return [float(v) for v in z.tolist()]


def pad_views(*views: Optional[Sequence[float]]) -> List[float]:
    parts: List[float] = []
    for v in views:
        if v:
            parts.extend(float(x) for x in v)
    return parts


def vae_cosine(enc: TinyVAE, a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    return float(cosine(enc.encode(a), enc.encode(b)))


def default_cache_path(task_prefix: str, repo: Optional[Path] = None) -> Path:
    root = repo or Path(__file__).resolve().parents[3]
    return root / "history_data" / "meta_resource" / f"semantic_profile_emb_{task_prefix}.pkl"


def default_vae_path(task_prefix: str, repo: Optional[Path] = None) -> Path:
    root = repo or Path(__file__).resolve().parents[3]
    return root / "history_data" / "meta_resource" / f"vae_latent_{task_prefix}.pkl"
"""Semantic dataset-profile embeddings and a tiny shared-space VAE fallback.

The LLM embedding endpoint is used when reachable. Otherwise a deterministic
hash vector is stored so retrieval still has a cold cosine signal. VAE training
is TUNE-only; LOCK is held out until the encoder is frozen.
"""

from __future__ import annotations

import hashlib
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from mindware.components.experience.profile import cosine, profile_from_dataset_profile, profile_vector

LATENT_DIM = 32
HASH_DIM = 64
SEMANTIC_WEIGHT = 0.10


def profile_text(
    dataset_id: Optional[str],
    dataset_profile: Optional[Dict[str, Any]],
    task_type: Optional[str],
    algorithm_id: Optional[str] = None,
) -> str:
    """Build a short natural-language dataset profile for embedding."""
    try:
        from mindware.third_party.opro.utils import _build_dataset_summary_text

        summary = _build_dataset_summary_text({"dataset_profile": dataset_profile or {}})
    except Exception:
        summary = ""
    if not summary or summary.strip() in ("", "N/A"):
        vec = profile_from_dataset_profile(dataset_profile)
        bits = ", ".join(f"{k}={v:.3f}" for k, v in vec.items())
        summary = f"numeric profile: {bits}"
    algo = f" algorithm={algorithm_id}" if algorithm_id else ""
    return (
        f"dataset={dataset_id or 'unknown'} task_type={task_type or 'unknown'}{algo}. "
        f"{summary}"
    ).strip()


def hash_embed(text: str, dim: int = HASH_DIM) -> List[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big") % (2**32)
    rng = np.random.RandomState(seed)
    vec = rng.normal(size=int(dim)).astype(np.float64)
    n = float(np.linalg.norm(vec))
    if n > 1e-12:
        vec = vec / n
    return [float(x) for x in vec.tolist()]


def llm_embed(text: str, client=None, model: str = "text-embedding-3-small") -> Optional[List[float]]:
    if client is None:
        return None
    try:
        resp = client.embeddings.create(model=model, input=text)
        vec = list(resp.data[0].embedding)
        return [float(x) for x in vec]
    except Exception:
        return None


def embed_profile(
    dataset_id: Optional[str],
    dataset_profile: Optional[Dict[str, Any]],
    task_type: Optional[str],
    *,
    algorithm_id: Optional[str] = None,
    client=None,
    model: str = "text-embedding-3-small",
) -> List[float]:
    text = profile_text(dataset_id, dataset_profile, task_type, algorithm_id)
    vec = llm_embed(text, client=client, model=model)
    if vec:
        return vec
    return hash_embed(text)


class EmbeddingCache:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.data: Dict[str, List[float]] = {}
        if self.path and self.path.is_file():
            try:
                payload = pickle.loads(self.path.read_bytes())
                if isinstance(payload, dict):
                    self.data = {str(k): list(v) for k, v in payload.items()}
            except Exception:
                self.data = {}

    def key(self, task_type: str, dataset_id: str) -> str:
        return f"{task_type}::{dataset_id}".lower()

    def get(self, task_type: str, dataset_id: str) -> Optional[List[float]]:
        return self.data.get(self.key(task_type, dataset_id))

    def put(self, task_type: str, dataset_id: str, vec: Sequence[float]) -> None:
        self.data[self.key(task_type, dataset_id)] = [float(x) for x in vec]

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(pickle.dumps(self.data))


class TinyVAE:
    """One-hidden-layer VAE mapping variable-width vectors into ``LATENT_DIM``.

    Trained with reconstruction + KL. If too few rows, falls back to PCA / SVD.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, hidden: int = 64, seed: int = 0):
        self.latent_dim = int(latent_dim)
        self.hidden = int(hidden)
        self.seed = int(seed)
        self.input_dim: Optional[int] = None
        self.W1 = self.b1 = None
        self.W_mu = self.b_mu = None
        self.W_lv = self.b_lv = None
        self.W2 = self.b2 = None
        self.W3 = self.b3 = None
        self.pca_components: Optional[np.ndarray] = None
        self.pca_mean: Optional[np.ndarray] = None
        self.use_pca = False

    def _init_weights(self, d: int) -> None:
        rng = np.random.RandomState(self.seed)
        self.input_dim = d
        h, z = self.hidden, self.latent_dim
        scale = lambda a, b: rng.randn(a, b) * math.sqrt(2.0 / max(a, 1))
        self.W1, self.b1 = scale(d, h), np.zeros(h)
        self.W_mu, self.b_mu = scale(h, z), np.zeros(z)
        self.W_lv, self.b_lv = scale(h, z), np.zeros(z)
        self.W2, self.b2 = scale(z, h), np.zeros(h)
        self.W3, self.b3 = scale(h, d), np.zeros(d)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    def _encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = self._relu(x @ self.W1 + self.b1)
        mu = h @ self.W_mu + self.b_mu
        lv = np.clip(h @ self.W_lv + self.b_lv, -8.0, 8.0)
        return h, mu, lv

    def fit(self, rows: Sequence[Sequence[float]], steps: int = 200, lr: float = 0.05) -> "TinyVAE":
        X = [np.asarray(r, dtype=np.float64).reshape(-1) for r in rows if r is not None and len(r)]
        if not X:
            return self
        d = max(v.size for v in X)
        mat = np.zeros((len(X), d), dtype=np.float64)
        for i, v in enumerate(X):
            mat[i, : v.size] = v
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        mat = mat / norms
        self.input_dim = d
        if mat.shape[0] < 8:
            self._fit_pca(mat)
            return self
        self._init_weights(d)
        rng = np.random.RandomState(self.seed)
        for _ in range(max(1, int(steps))):
            h, mu, lv = self._encode(mat)
            std = np.exp(0.5 * lv)
            eps = rng.randn(*mu.shape)
            z = mu + std * eps
            h2 = self._relu(z @ self.W2 + self.b2)
            rec = h2 @ self.W3 + self.b3
            err = rec - mat
            recon = err / max(mat.shape[0], 1)
            # KL to N(0,1)
            kl_coef = 0.01
            # SGD on reconstruction only for the linear maps; KL shrinks mu/lv.
            self.W3 -= lr * (h2.T @ recon)
            self.b3 -= lr * recon.mean(axis=0)
            dh2 = recon @ self.W3.T
            dh2 *= (h2 > 0)
            self.W2 -= lr * (z.T @ dh2) / max(mat.shape[0], 1)
            self.b2 -= lr * dh2.mean(axis=0)
            self.W_mu -= lr * kl_coef * mu.mean(axis=0)
            self.W_lv -= lr * kl_coef * lv.mean(axis=0)
        self.use_pca = False
        return self

    def _fit_pca(self, mat: np.ndarray) -> None:
        self.use_pca = True
        self.pca_mean = mat.mean(axis=0)
        centered = mat - self.pca_mean
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            vt = np.eye(mat.shape[1])
        k = min(self.latent_dim, vt.shape[0])
        self.pca_components = vt[:k]
        self.input_dim = mat.shape[1]

    def encode(self, vec: Sequence[float]) -> List[float]:
        x = np.asarray(list(vec), dtype=np.float64).reshape(-1)
        d = int(self.input_dim or x.size or 1)
        padded = np.zeros(d, dtype=np.float64)
        padded[: min(d, x.size)] = x[: min(d, x.size)]
        n = float(np.linalg.norm(padded))
        if n > 1e-12:
            padded = padded / n
        if self.use_pca and self.pca_components is not None:
            z = (padded - (self.pca_mean if self.pca_mean is not None else 0.0)) @ self.pca_components.T
        elif self.W1 is not None:
            _, mu, _ = self._encode(padded.reshape(1, -1))
            z = mu.reshape(-1)
        else:
            z = padded[: self.latent_dim]
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        if z.size < self.latent_dim:
            z = np.pad(z, (0, self.latent_dim - z.size))
        z = z[: self.latent_dim]
        n = float(np.linalg.norm(z))
        if n > 1e-12:
            z = z / n
        return [float(v) for v in z.tolist()]


def pad_views(*views: Optional[Sequence[float]]) -> List[float]:
    parts: List[float] = []
    for v in views:
        if v:
            parts.extend(float(x) for x in v)
    return parts


def vae_cosine(enc: TinyVAE, a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    return float(cosine(enc.encode(a), enc.encode(b)))


def default_cache_path(task_prefix: str, repo: Optional[Path] = None) -> Path:
    root = repo or Path(__file__).resolve().parents[3]
    return root / "history_data" / "meta_resource" / f"semantic_profile_emb_{task_prefix}.pkl"


def default_vae_path(task_prefix: str, repo: Optional[Path] = None) -> Path:
    root = repo or Path(__file__).resolve().parents[3]
    return root / "history_data" / "meta_resource" / f"vae_latent_{task_prefix}.pkl"
"""Semantic dataset-profile embeddings and a tiny shared-space VAE fallback.

The LLM embedding endpoint is used when reachable. Otherwise a deterministic
hash vector is stored so retrieval still has a cold cosine signal. VAE training
is TUNE-only; LOCK is held out until the encoder is frozen.
"""

from __future__ import annotations

import hashlib
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from mindware.components.experience.profile import cosine, profile_from_dataset_profile, profile_vector

LATENT_DIM = 32
HASH_DIM = 64
SEMANTIC_WEIGHT = 0.10


def profile_text(
    dataset_id: Optional[str],
    dataset_profile: Optional[Dict[str, Any]],
    task_type: Optional[str],
    algorithm_id: Optional[str] = None,
) -> str:
    """Build a short natural-language dataset profile for embedding."""
    try:
        from mindware.third_party.opro.utils import _build_dataset_summary_text

        summary = _build_dataset_summary_text({"dataset_profile": dataset_profile or {}})
    except Exception:
        summary = ""
    if not summary or summary.strip() in ("", "N/A"):
        vec = profile_from_dataset_profile(dataset_profile)
        bits = ", ".join(f"{k}={v:.3f}" for k, v in vec.items())
        summary = f"numeric profile: {bits}"
    algo = f" algorithm={algorithm_id}" if algorithm_id else ""
    return (
        f"dataset={dataset_id or 'unknown'} task_type={task_type or 'unknown'}{algo}. "
        f"{summary}"
    ).strip()


def hash_embed(text: str, dim: int = HASH_DIM) -> List[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big") % (2**32)
    rng = np.random.RandomState(seed)
    vec = rng.normal(size=int(dim)).astype(np.float64)
    n = float(np.linalg.norm(vec))
    if n > 1e-12:
        vec = vec / n
    return [float(x) for x in vec.tolist()]


def llm_embed(text: str, client=None, model: str = "text-embedding-3-small") -> Optional[List[float]]:
    if client is None:
        return None
    try:
        resp = client.embeddings.create(model=model, input=text)
        vec = list(resp.data[0].embedding)
        return [float(x) for x in vec]
    except Exception:
        return None


def embed_profile(
    dataset_id: Optional[str],
    dataset_profile: Optional[Dict[str, Any]],
    task_type: Optional[str],
    *,
    algorithm_id: Optional[str] = None,
    client=None,
    model: str = "text-embedding-3-small",
) -> List[float]:
    text = profile_text(dataset_id, dataset_profile, task_type, algorithm_id)
    vec = llm_embed(text, client=client, model=model)
    if vec:
        return vec
    return hash_embed(text)


class EmbeddingCache:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.data: Dict[str, List[float]] = {}
        if self.path and self.path.is_file():
            try:
                payload = pickle.loads(self.path.read_bytes())
                if isinstance(payload, dict):
                    self.data = {str(k): list(v) for k, v in payload.items()}
            except Exception:
                self.data = {}

    def key(self, task_type: str, dataset_id: str) -> str:
        return f"{task_type}::{dataset_id}".lower()

    def get(self, task_type: str, dataset_id: str) -> Optional[List[float]]:
        return self.data.get(self.key(task_type, dataset_id))

    def put(self, task_type: str, dataset_id: str, vec: Sequence[float]) -> None:
        self.data[self.key(task_type, dataset_id)] = [float(x) for x in vec]

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(pickle.dumps(self.data))


class TinyVAE:
    """One-hidden-layer VAE mapping variable-width vectors into ``LATENT_DIM``.

    Trained with reconstruction + KL. If too few rows, falls back to PCA / SVD.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, hidden: int = 64, seed: int = 0):
        self.latent_dim = int(latent_dim)
        self.hidden = int(hidden)
        self.seed = int(seed)
        self.input_dim: Optional[int] = None
        self.W1 = self.b1 = None
        self.W_mu = self.b_mu = None
        self.W_lv = self.b_lv = None
        self.W2 = self.b2 = None
        self.W3 = self.b3 = None
        self.pca_components: Optional[np.ndarray] = None
        self.pca_mean: Optional[np.ndarray] = None
        self.use_pca = False

    def _init_weights(self, d: int) -> None:
        rng = np.random.RandomState(self.seed)
        self.input_dim = d
        h, z = self.hidden, self.latent_dim
        scale = lambda a, b: rng.randn(a, b) * math.sqrt(2.0 / max(a, 1))
        self.W1, self.b1 = scale(d, h), np.zeros(h)
        self.W_mu, self.b_mu = scale(h, z), np.zeros(z)
        self.W_lv, self.b_lv = scale(h, z), np.zeros(z)
        self.W2, self.b2 = scale(z, h), np.zeros(h)
        self.W3, self.b3 = scale(h, d), np.zeros(d)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    def _encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = self._relu(x @ self.W1 + self.b1)
        mu = h @ self.W_mu + self.b_mu
        lv = np.clip(h @ self.W_lv + self.b_lv, -8.0, 8.0)
        return h, mu, lv

    def fit(self, rows: Sequence[Sequence[float]], steps: int = 200, lr: float = 0.05) -> "TinyVAE":
        X = [np.asarray(r, dtype=np.float64).reshape(-1) for r in rows if r is not None and len(r)]
        if not X:
            return self
        d = max(v.size for v in X)
        mat = np.zeros((len(X), d), dtype=np.float64)
        for i, v in enumerate(X):
            mat[i, : v.size] = v
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        mat = mat / norms
        self.input_dim = d
        if mat.shape[0] < 8:
            self._fit_pca(mat)
            return self
        self._init_weights(d)
        rng = np.random.RandomState(self.seed)
        for _ in range(max(1, int(steps))):
            h, mu, lv = self._encode(mat)
            std = np.exp(0.5 * lv)
            eps = rng.randn(*mu.shape)
            z = mu + std * eps
            h2 = self._relu(z @ self.W2 + self.b2)
            rec = h2 @ self.W3 + self.b3
            err = rec - mat
            recon = err / max(mat.shape[0], 1)
            # KL to N(0,1)
            kl_coef = 0.01
            # SGD on reconstruction only for the linear maps; KL shrinks mu/lv.
            self.W3 -= lr * (h2.T @ recon)
            self.b3 -= lr * recon.mean(axis=0)
            dh2 = recon @ self.W3.T
            dh2 *= (h2 > 0)
            self.W2 -= lr * (z.T @ dh2) / max(mat.shape[0], 1)
            self.b2 -= lr * dh2.mean(axis=0)
            self.W_mu -= lr * kl_coef * mu.mean(axis=0)
            self.W_lv -= lr * kl_coef * lv.mean(axis=0)
        self.use_pca = False
        return self

    def _fit_pca(self, mat: np.ndarray) -> None:
        self.use_pca = True
        self.pca_mean = mat.mean(axis=0)
        centered = mat - self.pca_mean
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            vt = np.eye(mat.shape[1])
        k = min(self.latent_dim, vt.shape[0])
        self.pca_components = vt[:k]
        self.input_dim = mat.shape[1]

    def encode(self, vec: Sequence[float]) -> List[float]:
        x = np.asarray(list(vec), dtype=np.float64).reshape(-1)
        d = int(self.input_dim or x.size or 1)
        padded = np.zeros(d, dtype=np.float64)
        padded[: min(d, x.size)] = x[: min(d, x.size)]
        n = float(np.linalg.norm(padded))
        if n > 1e-12:
            padded = padded / n
        if self.use_pca and self.pca_components is not None:
            z = (padded - (self.pca_mean if self.pca_mean is not None else 0.0)) @ self.pca_components.T
        elif self.W1 is not None:
            _, mu, _ = self._encode(padded.reshape(1, -1))
            z = mu.reshape(-1)
        else:
            z = padded[: self.latent_dim]
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        if z.size < self.latent_dim:
            z = np.pad(z, (0, self.latent_dim - z.size))
        z = z[: self.latent_dim]
        n = float(np.linalg.norm(z))
        if n > 1e-12:
            z = z / n
        return [float(v) for v in z.tolist()]


def pad_views(*views: Optional[Sequence[float]]) -> List[float]:
    parts: List[float] = []
    for v in views:
        if v:
            parts.extend(float(x) for x in v)
    return parts


def vae_cosine(enc: TinyVAE, a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    return float(cosine(enc.encode(a), enc.encode(b)))


def default_cache_path(task_prefix: str, repo: Optional[Path] = None) -> Path:
    root = repo or Path(__file__).resolve().parents[3]
    return root / "history_data" / "meta_resource" / f"semantic_profile_emb_{task_prefix}.pkl"


def default_vae_path(task_prefix: str, repo: Optional[Path] = None) -> Path:
    root = repo or Path(__file__).resolve().parents[3]
    return root / "history_data" / "meta_resource" / f"vae_latent_{task_prefix}.pkl"
"""Semantic dataset-profile embeddings and a tiny shared-space VAE fallback.

The LLM embedding endpoint is used when reachable. Otherwise a deterministic
hash vector is stored so retrieval still has a cold cosine signal. VAE training
is TUNE-only; LOCK is held out until the encoder is frozen.
"""

from __future__ import annotations

import hashlib
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from mindware.components.experience.profile import cosine, profile_from_dataset_profile, profile_vector

LATENT_DIM = 32
HASH_DIM = 64
SEMANTIC_WEIGHT = 0.10


def profile_text(
    dataset_id: Optional[str],
    dataset_profile: Optional[Dict[str, Any]],
    task_type: Optional[str],
    algorithm_id: Optional[str] = None,
) -> str:
    """Build a short natural-language dataset profile for embedding."""
    try:
        from mindware.third_party.opro.utils import _build_dataset_summary_text

        summary = _build_dataset_summary_text({"dataset_profile": dataset_profile or {}})
    except Exception:
        summary = ""
    if not summary or summary.strip() in ("", "N/A"):
        vec = profile_from_dataset_profile(dataset_profile)
        bits = ", ".join(f"{k}={v:.3f}" for k, v in vec.items())
        summary = f"numeric profile: {bits}"
    algo = f" algorithm={algorithm_id}" if algorithm_id else ""
    return (
        f"dataset={dataset_id or 'unknown'} task_type={task_type or 'unknown'}{algo}. "
        f"{summary}"
    ).strip()


def hash_embed(text: str, dim: int = HASH_DIM) -> List[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big") % (2**32)
    rng = np.random.RandomState(seed)
    vec = rng.normal(size=int(dim)).astype(np.float64)
    n = float(np.linalg.norm(vec))
    if n > 1e-12:
        vec = vec / n
    return [float(x) for x in vec.tolist()]


def llm_embed(text: str, client=None, model: str = "text-embedding-3-small") -> Optional[List[float]]:
    if client is None:
        return None
    try:
        resp = client.embeddings.create(model=model, input=text)
        vec = list(resp.data[0].embedding)
        return [float(x) for x in vec]
    except Exception:
        return None


def embed_profile(
    dataset_id: Optional[str],
    dataset_profile: Optional[Dict[str, Any]],
    task_type: Optional[str],
    *,
    algorithm_id: Optional[str] = None,
    client=None,
    model: str = "text-embedding-3-small",
) -> List[float]:
    text = profile_text(dataset_id, dataset_profile, task_type, algorithm_id)
    vec = llm_embed(text, client=client, model=model)
    if vec:
        return vec
    return hash_embed(text)


class EmbeddingCache:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.data: Dict[str, List[float]] = {}
        if self.path and self.path.is_file():
            try:
                payload = pickle.loads(self.path.read_bytes())
                if isinstance(payload, dict):
                    self.data = {str(k): list(v) for k, v in payload.items()}
            except Exception:
                self.data = {}

    def key(self, task_type: str, dataset_id: str) -> str:
        return f"{task_type}::{dataset_id}".lower()

    def get(self, task_type: str, dataset_id: str) -> Optional[List[float]]:
        return self.data.get(self.key(task_type, dataset_id))

    def put(self, task_type: str, dataset_id: str, vec: Sequence[float]) -> None:
        self.data[self.key(task_type, dataset_id)] = [float(x) for x in vec]

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(pickle.dumps(self.data))


class TinyVAE:
    """One-hidden-layer VAE mapping variable-width vectors into ``LATENT_DIM``.

    Trained with reconstruction + KL. If too few rows, falls back to PCA / SVD.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, hidden: int = 64, seed: int = 0):
        self.latent_dim = int(latent_dim)
        self.hidden = int(hidden)
        self.seed = int(seed)
        self.input_dim: Optional[int] = None
        self.W1 = self.b1 = None
        self.W_mu = self.b_mu = None
        self.W_lv = self.b_lv = None
        self.W2 = self.b2 = None
        self.W3 = self.b3 = None
        self.pca_components: Optional[np.ndarray] = None
        self.pca_mean: Optional[np.ndarray] = None
        self.use_pca = False

    def _init_weights(self, d: int) -> None:
        rng = np.random.RandomState(self.seed)
        self.input_dim = d
        h, z = self.hidden, self.latent_dim
        scale = lambda a, b: rng.randn(a, b) * math.sqrt(2.0 / max(a, 1))
        self.W1, self.b1 = scale(d, h), np.zeros(h)
        self.W_mu, self.b_mu = scale(h, z), np.zeros(z)
        self.W_lv, self.b_lv = scale(h, z), np.zeros(z)
        self.W2, self.b2 = scale(z, h), np.zeros(h)
        self.W3, self.b3 = scale(h, d), np.zeros(d)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    def _encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = self._relu(x @ self.W1 + self.b1)
        mu = h @ self.W_mu + self.b_mu
        lv = np.clip(h @ self.W_lv + self.b_lv, -8.0, 8.0)
        return h, mu, lv

    def fit(self, rows: Sequence[Sequence[float]], steps: int = 200, lr: float = 0.05) -> "TinyVAE":
        X = [np.asarray(r, dtype=np.float64).reshape(-1) for r in rows if r is not None and len(r)]
        if not X:
            return self
        d = max(v.size for v in X)
        mat = np.zeros((len(X), d), dtype=np.float64)
        for i, v in enumerate(X):
            mat[i, : v.size] = v
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        mat = mat / norms
        self.input_dim = d
        if mat.shape[0] < 8:
            self._fit_pca(mat)
            return self
        self._init_weights(d)
        rng = np.random.RandomState(self.seed)
        for _ in range(max(1, int(steps))):
            h, mu, lv = self._encode(mat)
            std = np.exp(0.5 * lv)
            eps = rng.randn(*mu.shape)
            z = mu + std * eps
            h2 = self._relu(z @ self.W2 + self.b2)
            rec = h2 @ self.W3 + self.b3
            err = rec - mat
            recon = err / max(mat.shape[0], 1)
            # KL to N(0,1)
            kl_coef = 0.01
            # SGD on reconstruction only for the linear maps; KL shrinks mu/lv.
            self.W3 -= lr * (h2.T @ recon)
            self.b3 -= lr * recon.mean(axis=0)
            dh2 = recon @ self.W3.T
            dh2 *= (h2 > 0)
            self.W2 -= lr * (z.T @ dh2) / max(mat.shape[0], 1)
            self.b2 -= lr * dh2.mean(axis=0)
            self.W_mu -= lr * kl_coef * mu.mean(axis=0)
            self.W_lv -= lr * kl_coef * lv.mean(axis=0)
        self.use_pca = False
        return self

    def _fit_pca(self, mat: np.ndarray) -> None:
        self.use_pca = True
        self.pca_mean = mat.mean(axis=0)
        centered = mat - self.pca_mean
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            vt = np.eye(mat.shape[1])
        k = min(self.latent_dim, vt.shape[0])
        self.pca_components = vt[:k]
        self.input_dim = mat.shape[1]

    def encode(self, vec: Sequence[float]) -> List[float]:
        x = np.asarray(list(vec), dtype=np.float64).reshape(-1)
        d = int(self.input_dim or x.size or 1)
        padded = np.zeros(d, dtype=np.float64)
        padded[: min(d, x.size)] = x[: min(d, x.size)]
        n = float(np.linalg.norm(padded))
        if n > 1e-12:
            padded = padded / n
        if self.use_pca and self.pca_components is not None:
            z = (padded - (self.pca_mean if self.pca_mean is not None else 0.0)) @ self.pca_components.T
        elif self.W1 is not None:
            _, mu, _ = self._encode(padded.reshape(1, -1))
            z = mu.reshape(-1)
        else:
            z = padded[: self.latent_dim]
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        if z.size < self.latent_dim:
            z = np.pad(z, (0, self.latent_dim - z.size))
        z = z[: self.latent_dim]
        n = float(np.linalg.norm(z))
        if n > 1e-12:
            z = z / n
        return [float(v) for v in z.tolist()]


def pad_views(*views: Optional[Sequence[float]]) -> List[float]:
    parts: List[float] = []
    for v in views:
        if v:
            parts.extend(float(x) for x in v)
    return parts


def vae_cosine(enc: TinyVAE, a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    return float(cosine(enc.encode(a), enc.encode(b)))


def default_cache_path(task_prefix: str, repo: Optional[Path] = None) -> Path:
    root = repo or Path(__file__).resolve().parents[3]
    return root / "history_data" / "meta_resource" / f"semantic_profile_emb_{task_prefix}.pkl"


def default_vae_path(task_prefix: str, repo: Optional[Path] = None) -> Path:
    root = repo or Path(__file__).resolve().parents[3]
    return root / "history_data" / "meta_resource" / f"vae_latent_{task_prefix}.pkl"
