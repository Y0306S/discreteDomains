"""Query adapter: BOLT VAE latent search + real Postgres execution oracle."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from mindware.domains.base import EvaluationResult
from mindware.domains.queryopt.adapter import QueryOptAdapter
from mindware.domains.queryopt.bolt_execution import (
    configure_bolt_db_from_dsn,
    execute_encoded,
    make_database_objective,
    runtime_warm_start_candidates,
)


class QueryBoltExecAdapter(QueryOptAdapter):
    """Arm-C latent search with BOLT aliases codec + execution runtime oracle."""

    domain_id = "query"
    metric = "runtime"
    higher_is_better = False

    def __init__(
        self,
        task: Optional[Dict[str, Any]] = None,
        *,
        execution_timeout: float = 100.0,
        worst_runtime_observed: float = 200.0,
        warm_start_perms: int = 15,
        warm_start_random_z: int = 12,
        **kwargs: Any,
    ):
        kwargs.pop("log_cost", None)
        super().__init__(task=task, log_cost=False, **kwargs)
        self.execution_timeout = float(execution_timeout)
        self.worst_runtime_observed = float(worst_runtime_observed)
        self.warm_start_perms = int(warm_start_perms)
        self.warm_start_random_z = int(warm_start_random_z)
        self._db_obj = None
        tid = str((self.task or {}).get("task_id") or "")
        if self.dsn:
            configure_bolt_db_from_dsn(self.dsn)
        if tid:
            self._db_obj = make_database_objective(
                tid,
                timeout=self.execution_timeout,
                worst_runtime_observed=self.worst_runtime_observed,
            )

    def _penalty_runtime(self) -> float:
        obj = self._db_obj
        if obj is None:
            return min(self.worst_runtime_observed, self.execution_timeout * 2.0)
        return min(float(obj.worst_runtime_observed), float(obj.timeout) * 2.0)

    def _order_to_flat(self, order: Sequence[str]) -> Optional[List[int]]:
        enc_pairs = getattr(self.vae, "table_alias_pairs_to_flat", None)
        if not callable(enc_pairs) or not self.alias_to_ta:
            return None
        pairs = [self.alias_to_ta[a] for a in order if a in self.alias_to_ta]
        if len(pairs) < 2:
            return None
        return list(enc_pairs(pairs))

    def evaluate(self, artifact: Any, task: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        order = list(artifact or self.aliases)
        info: Dict[str, Any] = {
            "join_order": order,
            "n_tables": float(len(self.aliases)),
            "query_layers": bool(self.query_layers),
            "oracle": "bolt_execute",
        }
        if self._db_obj is None:
            pen = self._penalty_runtime()
            info["error"] = "no_db_objective"
            return EvaluationResult(objective=pen, artifact=order, feasible=False, info=info)

        if self.query_layers and not self._legal_layer_b(order):
            pen = self._penalty_runtime()
            info.update({"decode_invalid": True, "raw_runtime": pen, "censored": 1})
            return EvaluationResult(objective=pen, artifact=order, feasible=False, info=info)

        flat = self._order_to_flat(order)
        if not flat:
            pen = self._penalty_runtime()
            info.update({"encode_failed": True, "raw_runtime": pen, "censored": 1})
            return EvaluationResult(objective=pen, artifact=order, feasible=False, info=info)

        info["encoded_plan"] = flat
        runtime, censored = execute_encoded(self._db_obj, flat)
        info["raw_runtime"] = float(runtime)
        info["raw_cost"] = float(runtime)  # arm-C index readers expect raw_cost key
        info["censored"] = int(censored)
        feasible = censored == 0
        return EvaluationResult(
            objective=float(runtime),
            artifact=order,
            feasible=feasible,
            info=info,
        )

    def warm_start(self, n_perm: int = 50) -> Tuple[Optional[List[float]], List[str], float]:
        if self._db_obj is None:
            return super().warm_start(n_perm=n_perm)

        cands = runtime_warm_start_candidates(
            task_id=str((self.task or {}).get("task_id") or ""),
            db_obj=self._db_obj,
            vae=self.vae,
            alias_to_ta=self.alias_to_ta,
            aliases=self.aliases,
            join_graph=self.join_graph,
            rng=self._rng,
            n_perm=min(int(n_perm), self.warm_start_perms),
            n_random_z=self.warm_start_random_z,
            latent_dim=self.latent_dim,
        )
        if not cands:
            return super().warm_start(n_perm=min(5, n_perm))

        best_enc, best_rt, best_cens, best_z = cands[0]
        best_order = list(self.aliases)
        for enc, rt, cens, z in cands:
            if rt < best_rt:
                best_rt, best_enc, best_cens, best_z = rt, enc, cens, z

        pairs_fn = getattr(self.vae, "flat_to_table_alias_pairs", None)
        if hasattr(self.vae, "rev_vocab") and pairs_fn is None:
            from mindware.domains.bolt_vae import flat_to_table_alias_pairs

            pairs_fn = flat_to_table_alias_pairs
        if callable(pairs_fn):
            n_tab = len(getattr(self.vae, "table_order", []) or [])
            n_alias = int(getattr(self.vae, "num_aliases", 3) or 3)
            pairs = pairs_fn(best_enc, n_tab, n_alias)
            order = self._pairs_to_order(pairs)
            if order:
                best_order = order

        if best_z is None:
            best_z = self.encode(best_order)
        if best_z is None and hasattr(self.vae, "encode_flat"):
            try:
                best_z = self.vae.encode_flat(best_enc)
            except Exception:
                best_z = None

        self._warm_start_order = list(best_order)
        return best_z, best_order, float(best_rt)
