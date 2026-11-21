"""Protein domain adapter: search in VAE latent z, decode AA sequence, score."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ConfigSpace import ConfigurationSpace

from mindware.domains.base import (
    DomainAdapter,
    EvaluationResult,
    dummy_result,
    latent_config_space,
    z_from_config,
    z_to_config,
)
from mindware.domains.protein.features import PROTEIN_ROLE_HINTS, extra_from_seq
from mindware.domains.protein.oracle import ESMFoldOracle, LookupOracle
from mindware.domains.sequence_vae import AA_ALPHABET, SequenceVAE

DEFAULT_VAE_PATH = Path("history_data/domain_protein/vae/sequence_vae.pkl")


class ProteinAdapter(DomainAdapter):
    domain_id = "protein"
    metric = "fitness"
    higher_is_better = True

    def __init__(
        self,
        task: Optional[Dict[str, Any]] = None,
        *,
        vae: Optional[SequenceVAE] = None,
        vae_path: Optional[Path] = None,
        oracle: Optional[LookupOracle] = None,
        apex: Optional[Any] = None,
        surrogate: Optional[Any] = None,
        dummy: bool = False,
        max_len: int = 48,
        latent_bound: float = 3.0,
        oracle_kind: str = "apex",
        dms_lookup: Optional[Dict[str, float]] = None,
        bolt_reference: Optional[str] = None,
        bolt_anchor: Optional[str] = None,
        bolt_min_similarity: float = 0.75,
        bolt_seed: int = 0,
        bolt_strict_decode: bool = False,
        bolt_reject_dup_sequences: bool = False,
        **kwargs: Any,
    ):
        super().__init__(task=task, **kwargs)
        self.dummy = bool(dummy)
        self.max_len = int(max_len)
        self.vae = vae
        self.apex = apex
        self.surrogate = surrogate
        self.oracle_kind = str(oracle_kind or "apex").lower()
        self.dms_lookup = dict(dms_lookup or {})
        self.bolt_reference = str(bolt_reference) if bolt_reference else None
        self.bolt_anchor = str(bolt_anchor) if bolt_anchor else None
        self.bolt_min_similarity = float(bolt_min_similarity)
        self.bolt_strict_decode = bool(bolt_strict_decode)
        self.bolt_reject_dup_sequences = bool(bolt_reject_dup_sequences)
        self._bolt_seen_sequences: Optional[set[str]] = None
        self._bolt_best_seq: Optional[str] = None
        self._bolt_best_obj = float("inf")
        self._bolt_rng = random.Random(int(bolt_seed))
        if self.vae is None and vae_path and Path(vae_path).is_file():
            # Prefer torch ProteinGym VAE when .pt; else numpy SequenceVAE pickle.
            if str(vae_path).endswith(".pt"):
                from mindware.domains.protein.torch_vae import ProteinGymVAE

                self.vae = ProteinGymVAE.load(
                    Path(vae_path),
                    device=str((self.task or {}).get("device") or "cuda:0"),
                )
            else:
                self.vae = SequenceVAE.load(Path(vae_path))
        if self.vae is None:
            self.vae = SequenceVAE(list(AA_ALPHABET), max_len=self.max_len)
        self.oracle = oracle or LookupOracle({})
        if self.oracle_kind == "esmfold" and not isinstance(self.oracle, ESMFoldOracle):
            cache = Path(
                str((self.task or {}).get("esmfold_cache") or "history_data/domain_protein/proteingym/esmfold_cache.jsonl")
            )
            self.oracle = ESMFoldOracle(cache, device=str((self.task or {}).get("device") or "cuda:0"))
        self.latent_dim = int(getattr(self.vae, "latent_dim", 16))
        self._cs = latent_config_space(self.latent_dim, bound=float(latent_bound))

    def search_space(self, **kwargs: Any) -> ConfigurationSpace:
        return self._cs

    def _decode_bolt_lolbo_style(self, seq: str) -> str:
        """LOLBO-style post-process: no feasibility repair toward reference/anchor."""
        from mindware.domains.protein.bolt_constraints import ensure_one_cysteine

        seq = str(seq or "").replace("X", "A").replace("-", "")
        seq = "".join(c for c in seq.upper() if c in AA_ALPHABET)
        seq = ensure_one_cysteine(seq)
        if not seq:
            seq = "AAA"
        return seq[: self.max_len]

    def decode(self, config: Any) -> str:
        z = z_from_config(config, dim=self.latent_dim)
        seq = self.vae.decode_string(z)
        if self.oracle_kind == "bolt_native" and self.bolt_strict_decode:
            seq = self._decode_bolt_lolbo_style(seq)
        else:
            seq = "".join(c for c in seq.upper() if c in AA_ALPHABET)
            if not seq:
                seq = "G" * 12
            if self.oracle_kind == "bolt_native" and self.bolt_reference:
                from mindware.domains.protein.bolt_constraints import propose_bolt_feasible_seq

                anchor = self.bolt_anchor or seq
                seen = self._bolt_seen_sequences if self.bolt_reject_dup_sequences else None
                seq = propose_bolt_feasible_seq(
                    decoded=seq,
                    reference=self.bolt_reference,
                    anchor=anchor,
                    best=self._bolt_best_seq,
                    min_similarity=self.bolt_min_similarity,
                    rng=self._bolt_rng,
                    seen=seen,
                )
        # Repair toward semantic bins when present.
        extra = extra_from_seq(seq)
        cfg_extra = extra_from_seq(seq)
        # Keep decoded sequence; semantic keys are recorded on evaluate.
        _ = extra, cfg_extra
        return seq[: self.max_len]

    def encode(self, artifact: Any) -> Optional[List[float]]:
        if not artifact:
            return None
        return self.vae.encode(str(artifact))

    def evaluate(self, artifact: Any, task: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        seq = str(artifact or "")
        if self.dummy:
            res = dummy_result({"seq": seq})
            res.artifact = seq
            res.info.update(extra_from_seq(seq))
            res.info["oracle"] = "dummy"
            return res
        feat = extra_from_seq(seq)
        true_dms = self.dms_lookup.get(seq)
        if self.oracle_kind == "surrogate" and self.surrogate is not None:
            obj, hit = self.surrogate(seq)
            info = {
                "oracle_hit": hit,
                "sequence": seq,
                "oracle": "esm_surrogate",
                "pred_fitness": float(-obj) if hit else None,
                "true_dms": true_dms,
                "in_dms_lookup": true_dms is not None,
                **feat,
            }
            return EvaluationResult(
                objective=float(obj),
                artifact=seq,
                feasible=bool(hit),
                info=info,
            )
        if self.oracle_kind == "esmfold":
            obj, hit = self.oracle(seq)
            feasible = hit and abs(float(obj)) < 1e5
            return EvaluationResult(
                objective=float(obj),
                artifact=seq,
                feasible=feasible,
                info={"oracle_hit": hit, "sequence": seq, "oracle": "esmfold_plddt", **feat},
            )
        if self.oracle_kind == "dms":
            # Closed library: unknown seq → infeasible (no hash fallback).
            if true_dms is not None:
                # Maximize fitness → minimize negative
                return EvaluationResult(
                    objective=float(-true_dms),
                    artifact=seq,
                    feasible=True,
                    info={
                        "oracle_hit": True,
                        "sequence": seq,
                        "oracle": "dms_lookup",
                        "true_dms": true_dms,
                        "in_dms_lookup": True,
                        **feat,
                    },
                )
            return EvaluationResult(
                objective=1e10,
                artifact=seq,
                feasible=False,
                info={
                    "oracle_hit": False,
                    "sequence": seq,
                    "oracle": "dms_lookup",
                    "true_dms": None,
                    "in_dms_lookup": False,
                    **feat,
                },
            )
        if self.oracle_kind == "dms_legacy" or (self.apex is None and self.oracle_kind not in {"apex", "surrogate"}):
            obj, hit = self.oracle(seq)
            return EvaluationResult(
                objective=float(obj),
                artifact=seq,
                feasible=True,
                info={"oracle_hit": hit, "sequence": seq, "oracle": "dms_lookup_legacy", **feat},
            )
        if self.oracle_kind == "bolt_native" and self.apex is not None:
            obj, hit = self.apex(seq)
            sim = None
            if hasattr(self.apex, "template"):
                from mindware.domains.protein.bolt_scorer import template_similarity

                sim = template_similarity(seq, getattr(self.apex, "template", ""))
            if hit and float(obj) < self._bolt_best_obj:
                self._bolt_best_obj = float(obj)
                self._bolt_best_seq = str(seq)
            return EvaluationResult(
                objective=float(obj),
                artifact=seq,
                feasible=bool(hit),
                info={
                    "oracle_hit": hit,
                    "sequence": seq,
                    "oracle": "bolt_bacteria_0",
                    "mic_bacteria_0": float(obj) if hit else None,
                    "template_similarity": sim,
                    **feat,
                },
            )
        if self.apex is not None:
            obj, hit = self.apex(seq)
            return EvaluationResult(
                objective=float(obj),
                artifact=seq,
                feasible=True,
                info={"oracle_hit": hit, "sequence": seq, "oracle": "apex_mean_mic", **feat},
            )
        obj, hit = self.oracle(seq)
        return EvaluationResult(
            objective=float(obj),
            artifact=seq,
            feasible=True,
            info={"oracle_hit": hit, "sequence": seq, "oracle": "lookup", **feat},
        )

    def e2e_finetune(
        self,
        sequences: Sequence[str],
        *,
        n_steps: int = 2,
        lr: Optional[float] = None,
    ) -> int:
        """Short BOLT-style VAE finetune on scored peptide sequences (TPE/LOLBO-lite)."""
        if self.oracle_kind != "bolt_native" or self.vae is None:
            return 0
        from mindware.domains.protein.bolt_peptide_e2e import finetune_peptide_bolt_vae

        return finetune_peptide_bolt_vae(
            self.vae,
            list(sequences),
            num_update_epochs=int(n_steps),
            learning_rte=lr,
        )

    def profile(self, task: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        t = task if task is not None else self.task
        n = float(t.get("n_seqs") or len(self.oracle.table) or 0.0)
        mean_len = float(t.get("mean_length") or 20.0)
        return {
            "log_n": float(__import__("math").log1p(n)),
            "mean_length": mean_len,
            "family_hash": float(t.get("family_hash") or 0.0),
            "n_seqs": n,
        }

    def experience_roles(self) -> List[Tuple[str, str]]:
        return list(PROTEIN_ROLE_HINTS)

    def semantic_keys(self) -> List[str]:
        return ["length_bin", "charge_bin", "cys_bin"]

    def semantic_choices(self) -> Dict[str, List[str]]:
        return {
            "length_bin": ["short", "medium", "long"],
            "charge_bin": ["anionic", "neutralish", "cationic"],
            "cys_bin": ["none", "one", "multi"],
        }

    def prompt_schema(self) -> Dict[str, Any]:
        base = super().prompt_schema()
        base["action"] = (
            "Propose JSON with length_bin, charge_bin, cys_bin, and optional motif notes. "
            "Do not emit z_0..z_15; TPE/CMA-ES walks the VAE latent."
        )
        base["artifact"] = "amino_acid_sequence"
        return base


def config_with_features(z: Sequence[float], seq: str) -> Dict[str, Any]:
    return z_to_config(z, extra=extra_from_seq(seq))
