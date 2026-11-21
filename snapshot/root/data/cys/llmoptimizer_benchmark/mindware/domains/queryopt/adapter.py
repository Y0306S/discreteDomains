"""Query-opt adapter: VAE over left-deep join-order tokens, score EXPLAIN cost."""

from __future__ import annotations

import hashlib
import math
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
)
from mindware.domains.queryopt.oracle import dummy_plan_cost, postgres_explain_cost
from mindware.domains.queryopt.sql_parse import (
    QUERY_ROLE_HINTS,
    alias_join_graph,
    alias_symbol_map,
    connected_prefix_repair,
    default_join_order,
    is_connected_order,
    is_legal_order,
    map_tables_to_aliases,
    parse_tables,
    pg_leading_hint,
    random_connected_order,
    repair_order,
)
from mindware.domains.sequence_vae import SequenceVAE

IMDB_TABLES = [
    "title", "kind_type", "info_type", "movie_info", "cast_info", "role_type",
    "name", "aka_name", "aka_title", "char_name", "comp_cast_type", "company_name",
    "company_type", "complete_cast", "keyword", "link_type", "movie_companies",
    "movie_info_idx", "movie_keyword", "movie_link", "person_info",
]


class QueryOptAdapter(DomainAdapter):
    domain_id = "query"
    metric = "cost"
    higher_is_better = False

    def __init__(
        self,
        task: Optional[Dict[str, Any]] = None,
        *,
        vae: Optional[SequenceVAE] = None,
        vae_path: Optional[Path] = None,
        dummy: bool = False,
        dsn: Optional[str] = None,
        latent_bound: float = 3.0,
        query_layers: bool = False,
        log_cost: bool = False,
        n_decode_tries: int = 8,
        **kwargs: Any,
    ):
        super().__init__(task=task, **kwargs)
        self.dummy = bool(dummy)
        self.dsn = dsn or __import__("os").environ.get("QUERYOPT_PG_DSN")
        if self.dsn:
            self.dummy = bool(dummy)
        self.query_layers = bool(query_layers)
        self.log_cost = bool(log_cost)
        self.n_decode_tries = int(n_decode_tries)
        sql = str((self.task or {}).get("sql") or "")
        self.pairs = parse_tables(sql) if sql else [(t, t[:2]) for t in IMDB_TABLES[:6]]
        self.aliases = default_join_order(self.pairs)
        self.join_graph = alias_join_graph(sql, self.aliases)
        alphabet = sorted({t for t, _ in self.pairs} | set(IMDB_TABLES))
        self.vae = vae
        if self.vae is None and vae_path and Path(vae_path).is_file():
            self.vae = SequenceVAE.load(Path(vae_path))
        if self.vae is None:
            self.vae = SequenceVAE(alphabet, max_len=max(8, len(self.aliases) + 2), latent_dim=16)
        self.latent_dim = int(getattr(self.vae, "latent_dim", 16))
        self._cs = latent_config_space(self.latent_dim, bound=float(latent_bound))
        self._warm_start_order: List[str] = list(self.aliases)
        self._rng = random.Random(int((self.task or {}).get("seed") or 0) or 1)
        self.alias_to_ta: Dict[str, Tuple[int, int]] = {}
        self.ta_to_alias: Dict[Tuple[int, int], str] = {}
        table_to_id = getattr(self.vae, "table_to_id", None)
        if isinstance(table_to_id, dict) and table_to_id:
            self.alias_to_ta, self.ta_to_alias = alias_symbol_map(
                self.pairs,
                table_to_id,
                num_aliases=int(getattr(self.vae, "num_aliases", 3) or 3),
            )

    def search_space(self, **kwargs: Any) -> ConfigurationSpace:
        return self._cs

    def _order_from_table_names(self, toks: Sequence[str]) -> List[str]:
        return map_tables_to_aliases(toks, self.pairs)

    def _legal_layer_b(self, order: List[str]) -> bool:
        if not is_legal_order(order, self.aliases):
            return False
        return is_connected_order(order, self.join_graph)

    @staticmethod
    def _z_seed(z: Sequence[float]) -> int:
        payload = ",".join(f"{float(v):.6f}" for v in z)
        return int(hashlib.md5(payload.encode()).hexdigest()[:8], 16)

    def _ta_neighbors(self, placed: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
        placed_aliases = [self.ta_to_alias[p] for p in placed if p in self.ta_to_alias]
        out: List[Tuple[int, int]] = []
        for a in self.aliases:
            if a in self.alias_to_ta and any(a in self.join_graph.get(b, ()) for b in placed_aliases):
                out.append(self.alias_to_ta[a])
        return out

    def _pairs_to_order(self, pairs: Sequence[Tuple[int, int]]) -> List[str]:
        order: List[str] = []
        seen = set()
        for ta in pairs:
            alias = self.ta_to_alias.get((int(ta[0]), int(ta[1])))
            if alias and alias not in seen:
                order.append(alias)
                seen.add(alias)
        return order

    def _constrained_decode(self, z: Sequence[float], *, greedy: bool, seed: int) -> List[str]:
        allowed = [self.alias_to_ta[a] for a in self.aliases if a in self.alias_to_ta]
        sample_fn = getattr(self.vae, "sample_constrained", None)
        if not callable(sample_fn) or not allowed:
            toks = self.vae.decode(z, max_len=max(4, len(self.aliases)), sample_seed=seed)
            return self._order_from_table_names(toks)
        pairs = sample_fn(
            z,
            allowed,
            greedy=greedy,
            sample_seed=seed,
            neighbors_fn=self._ta_neighbors,
        )
        return self._pairs_to_order(pairs)

    def decode(self, config: Any) -> List[str]:
        z = z_from_config(config, dim=self.latent_dim)
        base_z = [float(v) for v in z]
        if not self.query_layers:
            toks = self.vae.decode(base_z, max_len=max(4, len(self.aliases)))
            order = self._order_from_table_names(toks)
            skeleton = config_to_skel(config) if config is not None else "default"
            if skeleton == "fact_first" and self.aliases:
                order = [self.aliases[0]] + [a for a in order if a != self.aliases[0]]
            elif skeleton == "probe_first" and len(self.aliases) > 1:
                order = [self.aliases[-1]] + [a for a in order if a != self.aliases[-1]]
            return repair_order(order, self.aliases)

        seed = self._z_seed(base_z)
        best: List[str] = []
        tries = min(8, max(1, self.n_decode_tries))
        for t in range(tries):
            greedy = t == 0
            order = self._constrained_decode(base_z, greedy=greedy, seed=seed + t)
            order = connected_prefix_repair(order, self.aliases, self.join_graph)
            if self._legal_layer_b(order):
                if t == 0:
                    return order
                best = order
                break
            if len(order) > len(best):
                best = order
        if self._legal_layer_b(best):
            return best
        repaired = connected_prefix_repair(best, self.aliases, self.join_graph)
        if self._legal_layer_b(repaired):
            return repaired
        if self._warm_start_order and self._legal_layer_b(list(self._warm_start_order)):
            return list(self._warm_start_order)
        raise ValueError("decode_no_legal_connected_order")

    def encode(self, artifact: Any) -> Optional[List[float]]:
        if not artifact:
            return None
        aliases = list(artifact)
        if self.alias_to_ta and hasattr(self.vae, "encode_table_alias_pairs"):
            pairs = [self.alias_to_ta[a] for a in aliases if a in self.alias_to_ta]
            if len(pairs) >= 2:
                return self.vae.encode_table_alias_pairs(pairs)
        alias_to_table = {a: t for t, a in self.pairs}
        tables = [alias_to_table.get(a, a) for a in aliases]
        return self.vae.encode(tables)

    def warm_start(self, n_perm: int = 50) -> Tuple[Optional[List[float]], List[str], float]:
        """Layer C: score default + connected perms; store best as decode fallback."""
        rng = self._rng
        candidates = [list(self.aliases)]
        seen = {tuple(self.aliases)}
        for _ in range(max(0, int(n_perm))):
            order = random_connected_order(self.aliases, self.join_graph, rng)
            key = tuple(order)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(order)
        best_order = list(self.aliases)
        best_cost = float("inf")
        min_ok = 50.0 if len(self.aliases) >= 8 else 0.0
        for order in candidates:
            res = self.evaluate(order)
            raw = float((res.info or {}).get("raw_cost", res.objective))
            if not res.feasible:
                continue
            if not (min_ok < raw < 1e9):
                continue
            if raw < best_cost:
                best_cost = raw
                best_order = list(res.artifact or order)
        if not math.isfinite(best_cost) or best_cost >= 1e9:
            for order in candidates:
                res = self.evaluate(order)
                raw = float((res.info or {}).get("raw_cost", res.objective))
                if res.feasible and 0.0 < raw < 1e9 and raw < best_cost:
                    best_cost = raw
                    best_order = list(res.artifact or order)
        self._warm_start_order = best_order
        z = self.encode(best_order)
        return z, best_order, best_cost

    def e2e_finetune(self, plans: Sequence[Sequence[str]], n_steps: int = 4, lr: float = 1e-4) -> None:
        """Short LOLBO-style VAE update on decoded join orders (table-name sequences)."""
        module = getattr(self.vae, "module", None)
        tables_to_flat = getattr(self.vae, "tables_to_flat", None)
        if module is None or tables_to_flat is None or not plans:
            return
        import torch

        alias_to_table = {a: t for t, a in self.pairs}
        flats = []
        enc_pairs = getattr(self.vae, "table_alias_pairs_to_flat", None)
        for plan in plans:
            if callable(enc_pairs) and self.alias_to_ta:
                pairs = [self.alias_to_ta[a] for a in list(plan) if a in self.alias_to_ta]
                if len(pairs) >= 2:
                    flats.append(enc_pairs(pairs))
                    continue
            tables = [alias_to_table.get(a, a) for a in list(plan)]
            flats.append(tables_to_flat(tables))
        if not flats:
            return
        was_training = module.training
        module.train()
        opt = torch.optim.Adam(module.parameters(), lr=float(lr))
        device = next(module.parameters()).device
        for _ in range(max(1, int(n_steps))):
            opt.zero_grad(set_to_none=True)
            out = module(flats)
            loss = out["loss"]
            if hasattr(loss, "mean"):
                loss = loss.mean()
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            opt.step()
        module.eval()
        if was_training:
            module.train()
        _ = device

    def evaluate(self, artifact: Any, task: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        order = list(artifact or self.aliases)
        if self.query_layers:
            if not self._legal_layer_b(order):
                info: Dict[str, Any] = {
                    "join_order": order,
                    "n_tables": float(len(self.aliases)),
                    "query_layers": True,
                    "decode_invalid": True,
                    "oracle": "skipped",
                }
                penalty = math.log1p(1e12) if self.log_cost else 1e10
                return EvaluationResult(
                    objective=float(penalty),
                    artifact=order,
                    feasible=False,
                    info=info,
                )
        else:
            order = repair_order(order, self.aliases)
        t = task if task is not None else self.task
        sql = str((t or {}).get("sql") or "")
        hint = pg_leading_hint(order)
        info: Dict[str, Any] = {
            "join_order": order,
            "n_tables": float(len(self.aliases)),
            "starts_with": order[0] if order else "",
            "hint": hint.strip(),
            "query_layers": bool(self.query_layers),
        }
        if self.dummy or not sql:
            if self.dummy and not sql:
                res = dummy_result({"order": "|".join(order)})
                res.artifact = order
                res.info.update(info)
                return res
            cost = dummy_plan_cost(order, self.aliases)
            info["oracle"] = "dummy_cost"
        else:
            cost = postgres_explain_cost(sql, hint, dsn=self.dsn, join_order=order)
            if cost is None:
                cost = dummy_plan_cost(order, self.aliases)
                info["oracle"] = "dummy_cost_fallback"
            else:
                info["oracle"] = "postgres_explain"
        raw = float(cost)
        info["raw_cost"] = raw
        obj = math.log1p(min(raw, 1e12)) if self.log_cost else raw
        return EvaluationResult(objective=float(obj), artifact=order, feasible=True, info=info)

    def profile(self, task: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        return {
            "n_tables": float(len(self.aliases)),
            "sql_len": float(len(str((task or self.task or {}).get("sql") or ""))),
            "n_preds": float(str((task or self.task or {}).get("sql") or "").lower().count(" and ")),
        }

    def experience_roles(self) -> List[Tuple[str, str]]:
        return list(QUERY_ROLE_HINTS)

    def semantic_keys(self) -> List[str]:
        return ["join_skeleton"]

    def semantic_choices(self) -> Dict[str, List[str]]:
        return {"join_skeleton": ["fact_first", "probe_first", "default"]}

    def prompt_schema(self) -> Dict[str, Any]:
        base = super().prompt_schema()
        base["action"] = (
            "Propose JSON with join_skeleton in {fact_first, probe_first, default}. "
            "TPE/CMA-ES searches VAE latent for the left-deep join permutation."
        )
        base["aliases"] = self.aliases
        return base


def config_to_skel(config: Any) -> str:
    from mindware.domains.base import config_to_dict

    return str(config_to_dict(config).get("join_skeleton") or "default")
