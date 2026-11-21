#!/usr/bin/env python3
"""BOLT execution oracle + TPE on CEB query-plan tasks (no LOLBO).

Uses BOLT DatabaseObjective (real query runtime, aliases codec, timeout/censoring)
with BayesQO VAE latent TPE search. Output under arm_c_bo_v2/bolt_execute/.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains import get_adapter
from mindware.domains.bolt_vae import QueryBoltVAE
from mindware.domains.queryopt.bolt_lolbo_runner import QUERY_CONSISTENT10
from mindware.domains.queryopt.ceb import load_sql, load_task_ids
from mindware.domains.search import run_domain_search

AUTHOR_CKPT = Path("/root/data/cys/BayesQO/bayes_lqo/vae/CEB_64.ckpt")
OUT_ROOT = ROOT / "benchmark_data" / "domain_query" / "arm_c_bo_v2" / "bolt_execute"


def _done(history_path: Path, n_trials: int) -> bool:
    if not history_path.is_file():
        return False
    try:
        payload = json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return len(payload.get("observations") or []) >= int(n_trials)


def _append_index(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, default=str) + "\n"
    with path.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _select_tasks(
    n_tasks: int,
    offset: int,
    shard_index: int,
    n_shards: int,
    task_ids: list[str] | None = None,
) -> list[str]:
    if task_ids:
        return list(task_ids)
    ids = load_task_ids(limit=n_tasks, offset=offset)
    if n_shards <= 1:
        return ids
    return [tid for i, tid in enumerate(ids) if i % int(n_shards) == int(shard_index)]


def run_bolt_tpe(
    *,
    task_ids: list[str] | None,
    n_tasks: int,
    task_offset: int,
    shard_index: int,
    n_shards: int,
    n_trials: int,
    seed: int,
    device: str,
    dsn: str,
    skip_done: bool,
    out_root: Path,
    ckpt: Path,
    execution_timeout: float,
    warm_start_perms: int,
) -> int:
    vae = QueryBoltVAE(ckpt, device=device)
    index_path = out_root / "index.jsonl"
    opt = "tpe"
    tasks = _select_tasks(n_tasks, task_offset, shard_index, n_shards, task_ids)
    print(
        json.dumps(
            {
                "oracle": "bolt_execute",
                "optimizer": opt,
                "n_tasks_selected": len(tasks),
                "shard_index": shard_index,
                "n_shards": n_shards,
                "n_trials": n_trials,
                "dsn": dsn,
                "execution_timeout": execution_timeout,
                "device": device,
                "ckpt": str(ckpt),
            }
        ),
        flush=True,
    )
    n_ok = 0
    for i, tid in enumerate(tasks):
        sql = load_sql(tid)
        if not sql:
            print(json.dumps({"task_id": tid, "skip": "no_sql"}), flush=True)
            continue
        task_out = out_root / "C" / tid / opt / f"seed{seed}"
        hist_path = task_out / "history.json"
        if skip_done and _done(hist_path, n_trials):
            print(json.dumps({"task_id": tid, "skip": "done"}), flush=True)
            n_ok += 1
            continue
        ad = get_adapter(
            "query",
            oracle="bolt_execute",
            task={"task_id": tid, "sql": sql, "seed": seed},
            vae=vae,
            dummy=False,
            dsn=dsn,
            latent_bound=4.0,
            query_layers=True,
            execution_timeout=execution_timeout,
            warm_start_perms=warm_start_perms,
        )
        t0 = time.time()
        init_z, best_order, best_runtime = ad.warm_start(n_perm=warm_start_perms)
        warm = {"order": best_order, "raw_runtime": best_runtime, "init_z": init_z}
        result = run_domain_search(
            ad,
            n_trials=n_trials,
            seed=seed + i,
            optimizer=opt,
            output_dir=task_out,
            init_z=init_z,
            warm_start=warm,
            n_startup_trials=min(10, max(4, n_trials // 8)),
        )
        obs = result.get("observations") or []
        rts = []
        for o in obs:
            info = (o.get("extra_info") or {}).get("info") or {}
            rts.append(float(info.get("raw_runtime", info.get("raw_cost", o.get("objectives", [1e10])[0]))))
        row = {
            "arm": "C",
            "task_id": tid,
            "oracle": "bolt_execute",
            "optimizer": opt,
            "n_tables": len(ad.aliases),
            "ckpt": str(ckpt),
            "query_layers": True,
            "warm_start_runtime": best_runtime,
            "execution_timeout": execution_timeout,
            "n_trials": n_trials,
            "n_obs": len(obs),
            "best_objective": result.get("best_objective"),
            "best_raw_runtime": min(rts) if rts else None,
            "seconds": round(time.time() - t0, 2),
            "history": str(hist_path),
            "dsn_port": dsn.split("port=")[-1].split()[0] if "port=" in dsn else "?",
        }
        _append_index(index_path, row)
        print(json.dumps(row), flush=True)
        n_ok += 1
    return n_ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-trials", type=int, default=80)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--execution-timeout", type=float, default=100.0)
    p.add_argument("--warm-start-perms", type=int, default=15)
    p.add_argument("--n-tasks", type=int, default=3033)
    p.add_argument("--task-offset", type=int, default=0)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument(
        "--dsn",
        default=os.environ.get(
            "QUERYOPT_PG_DSN",
            "host=127.0.0.1 port=5433 dbname=imdb user=imdb password=imdb",
        ),
    )
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    p.add_argument("--ckpt", type=Path, default=AUTHOR_CKPT)
    p.add_argument("--skip-done", action="store_true", default=True)
    p.add_argument("--no-skip-done", action="store_false", dest="skip_done")
    p.add_argument("--task-ids", default="")
    args = p.parse_args()
    os.environ["QUERYOPT_PG_DSN"] = args.dsn
    tasks = [t.strip() for t in args.task_ids.split(",") if t.strip()] or None
    args.out_root.mkdir(parents=True, exist_ok=True)
    n = run_bolt_tpe(
        task_ids=tasks,
        n_tasks=args.n_tasks,
        task_offset=args.task_offset,
        shard_index=args.shard_index,
        n_shards=args.n_shards,
        n_trials=args.n_trials,
        seed=args.seed,
        device=args.device,
        dsn=args.dsn,
        skip_done=args.skip_done,
        out_root=args.out_root,
        ckpt=args.ckpt,
        execution_timeout=args.execution_timeout,
        warm_start_perms=args.warm_start_perms,
    )
    print(json.dumps({"finished": n, "shard_index": args.shard_index}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
