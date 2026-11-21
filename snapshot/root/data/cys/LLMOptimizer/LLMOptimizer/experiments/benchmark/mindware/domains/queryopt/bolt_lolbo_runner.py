"""BayesQO-aligned query LOLBO (censored GP + dynamic timeouts) on CEB tasks."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from mindware.domains.history import observation, write_openbox_history
from mindware.domains.paths import QUERY_ROOT, domain_bench

QUERY_QP_ROOT = QUERY_ROOT / "query_plan_optimization"

QUERY_CONSISTENT10 = [
    "CEB_11A114",
    "CEB_11A53",
    "CEB_11A97",
    "CEB_11A13",
    "CEB_11A94",
    "CEB_11A99",
    "CEB_11A133",
    "CEB_11A16",
    "CEB_11B27",
    "CEB_2A75",
]

AUTHOR_CKPT = Path("/root/data/cys/BayesQO/bayes_lqo/vae/CEB_64.ckpt")

# BayesQO query LOLBO defaults (optimize.py / info_transformer_vae_optimization.py)
BAYESQO_LOLBO_KWARGS: Dict[str, Any] = {
    "k": 100,
    "num_update_epochs": 2,
    "init_n_epochs": 80,
    "learning_rte": 0.01,
    "bsz": 1,
    "acq_func": "ts",
    "timeout_strategy": "ours",
    "timeout_percentile": 0.1,
    "gp_stdev_multiplier": 2,
    "censored_obs_is_max": True,
    "update_e2e": False,
    "e2e_freq": 10,
    "thompson_rejection": True,
}


def _ensure_query_lolbo_paths() -> None:
    """Import censored query LOLBO (BayesQO/BOLT query_plan_optimization), not peptide stack."""
    qp = str(QUERY_QP_ROOT)
    if qp in sys.path:
        sys.path.remove(qp)
    sys.path.insert(0, qp)
    for key in list(sys.modules):
        if key == "lolbo" or key.startswith("lolbo."):
            del sys.modules[key]


def _z_config(z_vec: torch.Tensor, dim: int) -> Dict[str, float]:
    arr = z_vec.detach().cpu().float().reshape(-1).tolist()
    return {f"z_{i}": float(arr[i]) for i in range(min(dim, len(arr)))}


def _append_index(path: Path, row: dict) -> None:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, default=str) + "\n"
    with path.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _make_objective(adapter):
    _ensure_query_lolbo_paths()
    from lolbo.latent_space_objective import LatentSpaceObjective

    class QueryExecObjective(LatentSpaceObjective):
        """BayesQO-compatible: encoded plans as x, maximize -runtime, censored oracle."""

        def __init__(self, ad):
            self.adapter = ad
            self.objective_function = ad._db_obj
            super().__init__(init_vae=False)
            self.vae = ad.vae.module
            self.dim = int(ad.latent_dim)

        def vae_decode(self, z):
            if isinstance(z, np.ndarray):
                z = torch.from_numpy(z).float()
            z = z.detach().float()
            device = next(self.vae.parameters()).device
            z = z.to(device)
            if z.dim() == 1:
                z = z.unsqueeze(0)
            self.vae.eval()
            with torch.no_grad():
                samples = self.vae.sample(z=z)
            return [[int(v) for v in s] for s in samples]

        def query_oracle(self, xs, timeouts_list=None):
            if self.objective_function is None:
                return [float("nan")] * len(xs), [1] * len(xs)
            if timeouts_list is None:
                return self.objective_function(xs)
            return self.objective_function(xs, timeouts_list)

        def initialize_vae(self):
            return None

        def vae_forward(self, xs_batch):
            device = next(self.vae.parameters()).device
            self.vae.eval()
            with torch.no_grad():
                out = self.vae(xs_batch)
            z = out.get("mu", out["z"]).reshape(-1, self.dim)
            loss = out["loss"]
            if hasattr(loss, "mean"):
                loss = loss.mean()
            return z, loss

    return QueryExecObjective(adapter)


def run_one(
    *,
    task_id: str,
    n_trials: int,
    seed: int,
    device: str,
    dsn: str,
    ckpt: Path,
    skip_done: bool,
    out_root: Path,
    oracle: str = "bolt_execute",
    execution_timeout: float = 100.0,
) -> Optional[Dict[str, Any]]:
    _ensure_query_lolbo_paths()
    from mindware.domains import get_adapter
    from mindware.domains.bolt_vae import QueryBoltVAE
    from mindware.domains.queryopt.bolt_execution import (
        bao_lolbo_init,
        runtime_to_lolbo_score,
        runtime_warm_start_candidates,
    )
    from mindware.domains.queryopt.ceb import load_sql

    oracle_name = str(oracle or "bolt_execute").lower()
    use_exec = oracle_name in {"bolt_execute", "bolt_exec", "execution", "runtime"}
    opt_name = "bolt_lolbo"
    task_out = out_root / "C" / task_id / opt_name / f"seed{seed}"
    hist_path = task_out / "history.json"
    if skip_done and hist_path.is_file():
        try:
            n = len(json.loads(hist_path.read_text()).get("observations") or [])
        except json.JSONDecodeError:
            n = 0
        if n >= int(n_trials):
            print(json.dumps({"task_id": task_id, "skip": "done", "n_obs": n}), flush=True)
            return None

    sql = load_sql(task_id)
    if not sql:
        print(json.dumps({"task_id": task_id, "skip": "no_sql"}), flush=True)
        return None

    vae = QueryBoltVAE(ckpt, device=device)
    _ensure_query_lolbo_paths()
    from lolbo.lolbo import LOLBOState

    adapter = get_adapter(
        "query",
        oracle="bolt_execute" if use_exec else "explain",
        task={"task_id": task_id, "sql": sql, "seed": seed},
        vae=vae,
        dummy=False,
        dsn=dsn,
        latent_bound=4.0,
        query_layers=True,
        log_cost=False,
        execution_timeout=execution_timeout,
    )
    t0 = time.time()
    init_censoring: Optional[List[int]] = None
    best_cost = 1e9
    bao_init = bao_lolbo_init(adapter, task_id) if use_exec else None
    if bao_init:
        xs = bao_init["xs"]
        ys = bao_init["ys"]
        zs_list = bao_init["zs"]
        init_censoring = bao_init["censoring"]
        worst_rt = float(bao_init["worst_runtime_observed"])
        if adapter._db_obj is not None:
            adapter._db_obj.worst_runtime_observed = worst_rt
            adapter.worst_runtime_observed = worst_rt
        best_i = max(range(len(ys)), key=lambda i: ys[i])
        best_cost = float(-ys[best_i])
        device_t = next(adapter.vae.module.parameters()).device
        zs = [
            torch.tensor(z, dtype=torch.float32, device=device_t).reshape(-1).detach()
            for z in zs_list
        ]
    else:
        init_z, _best_order, best_cost = adapter.warm_start(n_perm=50)
        device_t = next(adapter.vae.module.parameters()).device
        xs = []
        ys = []
        zs = []
        init_censoring = []
        if use_exec and adapter._db_obj is not None:
            cands = runtime_warm_start_candidates(
                task_id=task_id,
                db_obj=adapter._db_obj,
                vae=adapter.vae,
                alias_to_ta=adapter.alias_to_ta,
                aliases=adapter.aliases,
                join_graph=adapter.join_graph,
                rng=adapter._rng,
            )
            for enc, rt, cens, z in cands:
                if z is None:
                    try:
                        z = adapter.vae.encode_flat(enc)
                    except Exception:
                        continue
                xs.append([int(x) for x in enc])
                ys.append(runtime_to_lolbo_score(rt))
                init_censoring.append(int(cens))
                zs.append(
                    torch.tensor(z, dtype=torch.float32, device=device_t).reshape(-1).detach()
                )
        if not xs:
            z0 = torch.tensor(
                list(init_z or [0.0] * adapter.latent_dim),
                dtype=torch.float32,
                device=device_t,
            ).view(1, -1)
            flat0 = adapter.vae.sample_flat(z0.detach().cpu().reshape(-1).tolist())
            xs = [flat0]
            ys = [runtime_to_lolbo_score(float(best_cost))]
            init_censoring = [0]
            zs = [z0.reshape(-1).detach()]
    objective = _make_objective(adapter)

    n_init = len(xs)
    max_calls = max(0, int(n_trials) - n_init)
    lolbo_kw = dict(BAYESQO_LOLBO_KWARGS)
    update_e2e = bool(lolbo_kw.pop("update_e2e", False))
    e2e_freq = int(lolbo_kw.pop("e2e_freq", 10))
    thompson_rejection = bool(lolbo_kw.pop("thompson_rejection", True))
    os.chdir(QUERY_QP_ROOT)
    train_z = torch.stack(zs, dim=0).detach().cpu().float()
    train_y = torch.tensor(ys, dtype=torch.float32).view(-1, 1)
    censor_t = torch.tensor(init_censoring or [0] * n_init, dtype=torch.float32).view(-1, 1)
    train_z = torch.nan_to_num(train_z, nan=0.0, posinf=4.0, neginf=-4.0).clamp(-4.0, 4.0)
    train_y = torch.nan_to_num(train_y, nan=-1e6, posinf=-1e-6, neginf=-1e6)
    state = LOLBOState(
        objective=objective,
        train_x=xs,
        train_y=train_y,
        train_z=train_z,
        censoring=censor_t,
        train_c=None,
        thompson_rejection=thompson_rejection,
        verbose=True,
        **lolbo_kw,
    )
    bo_steps = 0
    max_bo_steps = max(1, max_calls * 15)
    while state.objective.num_calls < max_calls and bo_steps < max_bo_steps:
        try:
            if update_e2e and state.progress_fails_since_last_e2e >= e2e_freq:
                state.update_models_e2e()
                state.recenter()
            else:
                state.update_surrogate_model()
            state.acquisition()
            if state.tr_state.restart_triggered:
                state.initialize_tr_state()
            state.new_best_found = False
            bo_steps += 1
        except Exception as exc:
            print(json.dumps({"task_id": task_id, "warn": str(exc), "n_calls": int(state.objective.num_calls)}), flush=True)
            break
    train_x = list(state.train_x)
    train_y = state.train_y.detach().cpu().float().reshape(-1)
    train_z = state.train_z.detach().cpu().float()
    if train_z.dim() == 3:
        train_z = train_z.reshape(train_z.shape[0], -1)
    dim = int(train_z.shape[-1])
    observations: List[Dict[str, Any]] = []
    censor_hist = (
        state.censoring.detach().cpu().reshape(-1).tolist()
        if state.censoring is not None
        else [0] * len(train_x)
    )
    for i, (plan, yv, zv) in enumerate(zip(train_x, train_y, train_z)):
        score = float(yv.item())
        raw = float(-score) if math.isfinite(score) else 1e9
        cens = int(censor_hist[i]) if i < len(censor_hist) else 0
        oracle_tag = "bolt_execute" if use_exec else "postgres_explain"
        info = {
            "raw_cost": raw,
            "raw_runtime": raw,
            "lolbo_score": score,
            "censored": cens,
            "oracle": oracle_tag,
            "encoded_plan": list(plan) if isinstance(plan, (list, tuple)) else str(plan),
        }
        artifact: Any = list(plan) if isinstance(plan, (list, tuple)) else str(plan).split()
        observations.append(
            observation(
                _z_config(zv, dim),
                raw,
                trial_state=0 if cens == 0 else 1,
                extra={
                    "source": "bolt_lolbo_init" if i < n_init else "bolt_lolbo",
                    "artifact": artifact,
                    "info": info,
                },
            )
        )
    write_openbox_history(
        hist_path,
        task_id=task_id,
        observations=observations,
        meta={"optimizer": opt_name, "oracle": oracle_tag, "n_trials": n_trials},
    )
    raws = [float(o["objectives"][0]) for o in observations if float(o["objectives"][0]) < 1e9]
    row = {
        "arm": "C",
        "task_id": task_id,
        "optimizer": opt_name,
        "oracle": oracle_tag,
        "n_tables": len(adapter.aliases),
        "warm_start_cost": best_cost,
        "warm_start_runtime": best_cost if use_exec else None,
        "execution_timeout": execution_timeout if use_exec else None,
        "n_trials": n_trials,
        "n_obs": len(observations),
        "best_objective": min(raws) if raws else None,
        "best_raw_cost": min(raws) if raws else None,
        "best_raw_runtime": min(raws) if raws else None,
        "seconds": round(time.time() - t0, 2),
        "history": str(hist_path),
        "tr_length_final": float(state.tr_state.length),
        "e2e_updates": int(state.tot_num_e2e_updates),
    }
    _append_index(out_root / "index.jsonl", row)
    print(json.dumps(row), flush=True)
    return row


def run_tasks(
    *,
    n_trials: int = 80,
    seed: int = 1,
    device: str = "cuda:0",
    dsn: str,
    ckpt: Path = AUTHOR_CKPT,
    skip_done: bool = True,
    task_ids: Optional[List[str]] = None,
    out_root: Optional[Path] = None,
    oracle: str = "bolt_execute",
    execution_timeout: float = 100.0,
    n_tasks: Optional[int] = None,
    task_offset: int = 0,
    shard_index: int = 0,
    n_shards: int = 1,
) -> List[Dict[str, Any]]:
    _ensure_query_lolbo_paths()
    from mindware.domains.queryopt.ceb import load_task_ids

    out = out_root or (domain_bench("query") / "arm_c_bo_v2" / "bolt_execute")
    if task_ids:
        tasks = list(task_ids)
    else:
        tasks = load_task_ids(limit=n_tasks, offset=task_offset)
        if int(n_shards) > 1:
            tasks = [tid for i, tid in enumerate(tasks) if i % int(n_shards) == int(shard_index)]
    rows = []
    for tid in tasks:
        try:
            row = run_one(
                task_id=tid,
                n_trials=n_trials,
                seed=seed,
                device=device,
                dsn=dsn,
                ckpt=ckpt,
                skip_done=skip_done,
                out_root=out,
                oracle=oracle,
                execution_timeout=execution_timeout,
            )
            if row:
                rows.append(row)
        except Exception as exc:
            import traceback

            print(json.dumps({"task_id": tid, "error": str(exc), "tb": traceback.format_exc()}), flush=True)
    return rows


def run_consistent10(
    *,
    n_trials: int = 80,
    seed: int = 1,
    device: str = "cuda:0",
    dsn: str,
    ckpt: Path = AUTHOR_CKPT,
    skip_done: bool = True,
    task_ids: Optional[List[str]] = None,
    out_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    return run_tasks(
        n_trials=n_trials,
        seed=seed,
        device=device,
        dsn=dsn,
        ckpt=ckpt,
        skip_done=skip_done,
        task_ids=list(task_ids or QUERY_CONSISTENT10),
        out_root=out_root,
        oracle="bolt_execute",
        execution_timeout=100.0,
    )
