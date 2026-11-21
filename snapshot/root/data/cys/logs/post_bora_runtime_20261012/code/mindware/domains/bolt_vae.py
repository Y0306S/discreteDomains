"""BOLT InfoTransformer VAEs used as latent *state* for protein and query BO.

Protein: UniRef k=1 transformer VAE (latent 256 = bottleneck 2 × d_model 128).
Query: join-plan VAEModule (latent = bn_size × d_neck; LOLBO uses 64).

These wrap the checkpoints in ``/root/data/cys/BOLT`` rather than the numpy
bootstrap SequenceVAE (d=16).
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mindware.domains.paths import BOLT_PEPTIDES, QUERY_ROOT, domain_history
from mindware.domains.queryopt.ceb import IMDB_TABLES

PEPTIDE_VAE_CKPT = (
    BOLT_PEPTIDES
    / "uniref_vae/saved_models/dim128_k1_kl0001_eff256_dff256_pious-sea-2_model_state_epoch_118.pkl"
)
QUERY_VAE_CKPT = QUERY_ROOT / "query_plan_optimization/vae/CEB_64.ckpt"
QUERY_VAE_META = domain_history("query") / "vae/bolt_vae.meta.json"

ALL_AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def job_table_order() -> List[str]:
    """Canonical IMDB table order from JOB schema.sql (sqlglot, sorted)."""
    try:
        qp = str(QUERY_ROOT / "query_plan_optimization")
        if qp not in sys.path:
            sys.path.insert(0, qp)
        from workload.schema import build_table_order  # type: ignore
        from workload.workloads import IMDB_SCHEMA_PATH  # type: ignore

        order = build_table_order(IMDB_SCHEMA_PATH)
        if order:
            return [str(t).lower() for t in order]
    except Exception:
        pass
    return [str(t).lower() for t in IMDB_TABLES]


def ckpt_is_real(path: Path, min_bytes: int = 10_000) -> bool:
    p = Path(path)
    return p.is_file() and p.stat().st_size >= int(min_bytes)


def left_deep_flat(
    table_ids: Sequence[int],
    *,
    alias: int = 1,
    alias_nums: Optional[Sequence[int]] = None,
    op: int = 1,
) -> List[int]:
    ids = [int(x) for x in table_ids]
    if len(ids) < 2:
        ids = (ids + [0, 0])[:2]
    if alias_nums is None:
        anums = [int(alias)] * len(ids)
    else:
        anums = [int(a) for a in alias_nums]
        if len(anums) < len(ids):
            anums = anums + [int(alias)] * (len(ids) - len(anums))
    tokens: List[int] = []
    for i in range(len(ids) - 1):
        tokens.extend([ids[i], anums[i], ids[i + 1], anums[i + 1], int(op)])
    return tokens


def flat_to_table_alias_pairs(
    flat: Sequence[int],
    n_tables: int,
    num_aliases: int,
) -> List[Tuple[int, int]]:
    """Parse 5-grams to unique left-deep (table_id, alias_num); skip OOD ids (no wrap)."""
    seq = [int(x) for x in flat]
    pairs: List[Tuple[int, int]] = []
    seen = set()

    def _ok(t: int, a: int) -> bool:
        return 0 <= int(t) < int(n_tables) and 1 <= int(a) <= int(num_aliases)

    for i in range(0, max(0, len(seq) - 4), 5):
        t1, a1, t2, a2, _op = seq[i : i + 5]
        for t, a in ((t1, a1), (t2, a2)):
            key = (int(t), int(a))
            if not _ok(*key) or key in seen:
                continue
            seen.add(key)
            pairs.append(key)
    return pairs


def flat_to_table_names(flat: Sequence[int], table_order: Sequence[str]) -> List[str]:
    names: List[str] = []
    n = len(table_order) or 1
    n_alias = 8
    for tid, _anum in flat_to_table_alias_pairs(flat, n, n_alias):
        if 0 <= tid < n:
            names.append(table_order[tid])
    return names


def _ensure_one_c(seq: str) -> str:
    while seq.count("C") > 1:
        idxs = [i for i, ch in enumerate(seq) if ch == "C"]
        index = random.choice(idxs)
        replacement = random.choice([aa for aa in ALL_AMINO_ACIDS if aa != "C"])
        seq = seq[:index] + replacement + seq[index + 1 :]
    return seq


class PeptideBoltVAE:
    """UniRef InfoTransformer VAE: encode AA string ↔ z ∈ R^256."""

    def __init__(
        self,
        ckpt: Optional[Path] = None,
        *,
        device: str = "cuda:0",
        max_string_length: int = 50,
        latent_dim: int = 256,
    ):
        ckpt = Path(ckpt) if ckpt is not None else PEPTIDE_VAE_CKPT
        if not ckpt_is_real(ckpt):
            raise FileNotFoundError(f"BOLT peptide VAE checkpoint missing or placeholder: {ckpt}")
        self.ckpt = ckpt
        self.device = str(device)
        self.latent_dim = int(latent_dim)
        self.max_string_length = int(max_string_length)

        peptides = str(BOLT_PEPTIDES)
        if peptides not in sys.path:
            sys.path.insert(0, peptides)
        cwd = os.getcwd()
        try:
            os.chdir(peptides)
            from uniref_vae.load_vae import load_vae, vae_decode, vae_forward  # type: ignore

            self._vae_forward = vae_forward
            self._vae_decode = vae_decode
            self.vae, self.dataobj = load_vae(
                path_to_vae_statedict=str(ckpt),
                dim=self.latent_dim,
                max_string_length=self.max_string_length,
            )
        finally:
            os.chdir(cwd)
        self.vae = self.vae.to(self.device).eval()

    def encode(self, seq: Sequence[str] | str) -> List[float]:
        import torch

        if not isinstance(seq, str):
            seq = "".join(str(t) for t in seq)
        seq = str(seq or "AAA")
        with torch.no_grad():
            z, _loss = self._vae_forward([seq], self.dataobj, self.vae)
        z = z.reshape(-1, self.latent_dim)[0]
        return [float(x) for x in z.detach().cpu().tolist()]

    def decode(self, z: Sequence[float], max_len: Optional[int] = None, **kwargs: Any) -> List[str]:
        return list(self.decode_string(z, max_len=max_len))

    def decode_string(self, z: Sequence[float], max_len: Optional[int] = None, **kwargs: Any) -> str:
        import torch

        arr = [float(v) for v in z]
        if len(arr) < self.latent_dim:
            arr = arr + [0.0] * (self.latent_dim - len(arr))
        t = torch.tensor(
            arr[: self.latent_dim], dtype=torch.float32, device=self.device
        ).view(1, -1)
        with torch.no_grad():
            seqs = self._vae_decode(t, self.vae, self.dataobj)
        seq = str(seqs[0] if seqs else "AAA")
        seq = seq.replace("X", "A").replace("-", "")
        seq = "".join(c for c in seq.upper() if c in ALL_AMINO_ACIDS)
        seq = _ensure_one_c(seq)
        if not seq:
            seq = "AAA"
        cap = int(max_len or self.max_string_length)
        return seq[:cap]


class QueryBoltVAE:
    """BOLT query-plan VAEModule: left-deep join encodings ↔ z ∈ R^{bn×d_neck}."""

    def __init__(
        self,
        ckpt: Optional[Path] = None,
        *,
        device: str = "cuda:0",
        table_order: Optional[Sequence[str]] = None,
        num_aliases: int = 3,
        d_neck: int = 32,
        bn_size: int = 2,
        max_string_length: int = 48,
        meta_path: Optional[Path] = None,
    ):
        ckpt = Path(ckpt) if ckpt is not None else QUERY_VAE_CKPT
        meta_path = Path(meta_path) if meta_path is not None else QUERY_VAE_META
        meta: Dict[str, Any] = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        if not ckpt_is_real(ckpt):
            alt = domain_history("query") / "vae/CEB_64.ckpt"
            if ckpt_is_real(alt):
                ckpt = alt
        if not ckpt_is_real(ckpt):
            raise FileNotFoundError(
                f"BOLT query VAE checkpoint missing or placeholder: {ckpt}. "
                "Train with examples/benchmark/domains/train_bolt_query_vae.py"
            )
        self.ckpt = ckpt
        self.device = str(device)
        self.table_order = [str(t).lower() for t in (table_order or meta.get("table_order") or job_table_order())]
        self.table_to_id = {t: i for i, t in enumerate(self.table_order)}
        self.num_aliases = int(meta.get("num_aliases") or num_aliases)
        self.d_neck = int(meta.get("d_neck") or d_neck)
        self.bn_size = int(meta.get("bn_size") or bn_size)
        self.latent_dim = int(meta.get("latent_dim") or (self.bn_size * self.d_neck))
        self.max_string_length = int(meta.get("max_string_length") or max_string_length)

        qp = str(QUERY_ROOT / "query_plan_optimization")
        if qp not in sys.path:
            sys.path.insert(0, qp)
        from vae.model import VAEModule  # type: ignore
        import torch

        try:
            self.module = VAEModule.load_from_checkpoint(str(ckpt), map_location=self.device)
        except Exception:
            from vae.model import build_vocab  # type: ignore

            vocab, rev_vocab = build_vocab(len(self.table_order), self.num_aliases)
            self.module = VAEModule.load_from_checkpoint(
                str(ckpt),
                vocab=vocab,
                rev_vocab=rev_vocab,
                bn_size=self.bn_size,
                d_enc=128,
                d_dec=128,
                d_neck=self.d_neck,
                map_location=self.device,
            )
        hp = getattr(self.module, "hparams", None)
        if hp is not None:
            self.bn_size = int(getattr(hp, "bn_size", self.bn_size))
            self.d_neck = int(getattr(hp, "d_neck", self.d_neck))
            self.latent_dim = int(self.bn_size * self.d_neck)
        self.module.max_string_length = self.max_string_length
        self.module = self.module.to(self.device).eval()
        self.vocab = self.module.vocab
        self.rev_vocab = self.module.rev_vocab
        self._torch = torch

    def tables_to_flat(self, tables: Sequence[str]) -> List[int]:
        ids = [self.table_to_id.get(str(t).lower(), 0) for t in tables]
        return left_deep_flat(ids)

    def table_alias_pairs_to_flat(self, pairs: Sequence[Tuple[int, int]]) -> List[int]:
        ids = [int(t) for t, _ in pairs]
        anums = [int(a) for _, a in pairs]
        return left_deep_flat(ids, alias_nums=anums)

    def encode(self, seq: Sequence[str] | str) -> List[float]:
        if isinstance(seq, str):
            tables = [tok for tok in seq.replace(",", " ").split() if tok]
        else:
            tables = [str(t) for t in seq]
        flat = self.tables_to_flat(tables)
        return self.encode_flat(flat)

    def encode_table_alias_pairs(self, pairs: Sequence[Tuple[int, int]]) -> List[float]:
        return self.encode_flat(self.table_alias_pairs_to_flat(pairs))

    def encode_flat(self, flat: Sequence[int]) -> List[float]:
        torch = self._torch
        with torch.no_grad():
            out = self.module([list(flat)])
            z = out.get("mu", out["z"])
        z = z.reshape(-1, self.latent_dim)[0]
        return [float(x) for x in z.detach().cpu().tolist()]

    def sample_flat(self, z: Sequence[float], sample_seed: Optional[int] = None) -> List[int]:
        torch = self._torch
        if sample_seed is not None:
            s = int(sample_seed) % (2**31 - 1)
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)
        arr = [float(v) for v in z]
        if len(arr) < self.latent_dim:
            arr = arr + [0.0] * (self.latent_dim - len(arr))
        t = torch.tensor(arr[: self.latent_dim], dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            decoded = self.module.sample(t)
        if not decoded:
            return left_deep_flat([0, 1])
        return [int(x) for x in decoded[0]]

    def sample_constrained(
        self,
        z: Sequence[float],
        allowed: Sequence[Tuple[int, int]],
        *,
        greedy: bool = False,
        sample_seed: Optional[int] = None,
        neighbors_fn=None,
    ) -> List[Tuple[int, int]]:
        """Autoregressive decode with vocab masked to this query's (table, alias) symbols."""
        torch = self._torch
        from vae.model import PAD, START  # type: ignore

        allowed_set = {(int(t), int(a)) for t, a in allowed}
        if not allowed_set:
            return []
        if sample_seed is not None:
            s = int(sample_seed) % (2**31 - 1)
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)
        arr = [float(v) for v in z]
        if len(arr) < self.latent_dim:
            arr = arr + [0.0] * (self.latent_dim - len(arr))
        zt = torch.tensor(arr[: self.latent_dim], dtype=torch.float32, device=self.device).view(1, -1)
        n = 1
        zt = zt.reshape(n, self.latent_dim)
        vocab = self.vocab
        start_id = vocab[START]
        pad_id = vocab[PAD]
        op_ids = [vocab[op] for op in (1, 2, 3) if op in vocab]
        ta_to_id = {ta: vocab[ta] for ta in allowed_set if ta in vocab}

        def _pick(logits_1d, valid_ids: List[int]) -> int:
            if not valid_ids:
                return int(pad_id)
            idx = torch.tensor(valid_ids, device=logits_1d.device, dtype=torch.long)
            masked = logits_1d.index_select(0, idx)
            if greedy:
                j = int(torch.argmax(masked).item())
            else:
                j = int(torch.distributions.Categorical(logits=masked).sample().item())
            return int(valid_ids[j])

        model = self.module.model
        tokens = torch.full((1, 1), fill_value=start_id, device=self.device)
        placed: List[Tuple[int, int]] = []
        remaining = set(allowed_set)
        n_need = len(allowed_set)
        max_steps = max(12, n_need * 3 + 4)
        with torch.no_grad():
            for step in range(max_steps):
                logits = model.decode(zt, tokens)[:, -1, :]
                kind = step % 3
                if kind == 2:
                    valid = op_ids
                elif kind == 0:
                    if not placed:
                        cands = remaining
                    else:
                        cands = {placed[-1]}
                    valid = [ta_to_id[ta] for ta in cands if ta in ta_to_id]
                    if not valid:
                        valid = [ta_to_id[ta] for ta in remaining if ta in ta_to_id] or op_ids
                else:
                    cands = set(remaining)
                    if neighbors_fn is not None and placed:
                        neigh = set(neighbors_fn(list(placed)))
                        hit = cands & neigh
                        if hit:
                            cands = hit
                    valid = [ta_to_id[ta] for ta in cands if ta in ta_to_id]
                    if not valid:
                        valid = [ta_to_id[ta] for ta in remaining if ta in ta_to_id]
                tok = _pick(logits[0], valid)
                tokens = torch.hstack([tokens, torch.tensor([[tok]], device=self.device)])
                if tok == pad_id:
                    break
                if kind in (0, 1):
                    picked = self.rev_vocab.get(tok)
                    if isinstance(picked, tuple) and picked in remaining:
                        placed.append((int(picked[0]), int(picked[1])))
                        remaining.discard(picked)
                if len(placed) >= n_need:
                    break
                if tokens.shape[-1] > getattr(self.module, "max_string_length", 90):
                    break
        return placed

    def decode(
        self,
        z: Sequence[float],
        max_len: Optional[int] = None,
        sample_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        names = flat_to_table_names(
            self.sample_flat(z, sample_seed=sample_seed), self.table_order
        )
        if max_len is not None:
            names = names[: int(max_len)]
        return names if names else [self.table_order[0]]

    def decode_string(self, z: Sequence[float], **kwargs: Any) -> str:
        return " ".join(self.decode(z, **kwargs))


def save_query_vae_meta(
    path: Path,
    *,
    table_order: Sequence[str],
    num_aliases: int,
    d_neck: int,
    bn_size: int,
    max_string_length: int = 48,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "table_order": [str(t).lower() for t in table_order],
        "num_aliases": int(num_aliases),
        "d_neck": int(d_neck),
        "bn_size": int(bn_size),
        "latent_dim": int(bn_size) * int(d_neck),
        "max_string_length": int(max_string_length),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
"""BOLT InfoTransformer VAEs used as latent *state* for protein and query BO.

Protein: UniRef k=1 transformer VAE (latent 256 = bottleneck 2 × d_model 128).
Query: join-plan VAEModule (latent = bn_size × d_neck; LOLBO uses 64).

These wrap the checkpoints in ``/root/data/cys/BOLT`` rather than the numpy
bootstrap SequenceVAE (d=16).
"""
