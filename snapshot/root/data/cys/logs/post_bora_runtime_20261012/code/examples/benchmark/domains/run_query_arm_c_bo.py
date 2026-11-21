#!/usr/bin/env python3
"""Arm C BO on CEB tasks: author VAE + query layers + LOLBO + warm-start.

Scale-up from probe_layers_ab (12 tasks). Writes under
benchmark_data/domain_query/arm_c_bo/ and index.jsonl.
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
from mindware.domains.queryopt.ceb import load_sql, load_task_ids
from mindware.domains.search import run_domain_search

AUTHOR_CKPT = Path("/root/data/cys/BayesQO/bayes_lqo/vae/CEB_64.ckpt")
OUT_ROOT = ROOT / "benchmark_data" / "domain_query" / "arm_c_bo_v2"
INDEX_PATH = OUT_ROOT / "index.jsonl"


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


def run_arm_c(
    *,
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
    arm: str,
    optimizer: str = "lolbo",
    task_ids: list[str] | None = None,
) -> int:
    vae = QueryBoltVAE(ckpt, device=device)
    index_path = out_root / "index.jsonl"
    opt = str(optimizer or "lolbo").lower()
    tasks = _select_tasks(n_tasks, task_offset, shard_index, n_shards, task_ids)
    print(
        json.dumps(
            {
                "arm": arm,
                "n_tasks_selected": len(tasks),
                "shard_index": shard_index,
                "n_shards": n_shards,
                "optimizer": opt,
                "n_trials": n_trials,
                "dsn": dsn,
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
        task_out = out_root / arm / tid / opt / f"seed{seed}"
        hist_path = task_out / "history.json"
        if skip_done and _done(hist_path, n_trials):
            print(json.dumps({"task_id": tid, "skip": "done"}), flush=True)
            n_ok += 1
            continue
        ad = get_adapter(
            "query",
            task={"task_id": tid, "sql": sql, "seed": seed},
            vae=vae,
            dummy=False,
            dsn=dsn,
            latent_bound=4.0,
            query_layers=True,
            log_cost=True,
        )
        t0 = time.time()
        init_z, best_order, best_cost = ad.warm_start(n_perm=50)
        warm = {"order": best_order, "raw_cost": best_cost, "init_z": init_z}
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
        raws = []
        for o in obs:
            info = (o.get("extra_info") or {}).get("info") or {}
            raws.append(float(info.get("raw_cost", o.get("objectives", [1e10])[0])))
        row = {
            "arm": arm,
            "task_id": tid,
            "n_tables": len(ad.aliases),
            "ckpt": str(ckpt),
            "optimizer": opt,
            "query_layers": True,
            "warm_start_cost": best_cost,
            "n_trials": n_trials,
            "n_obs": len(obs),
            "best_objective": result.get("best_objective"),
            "best_raw_cost": min(raws) if raws else None,
            "poisoned": bool(raws) and all(r >= 1e9 for r in raws),
            "seconds": round(time.time() - t0, 2),
            "history": str(hist_path),
            "shard_index": shard_index,
            "dsn_port": dsn.split("port=")[-1].split()[0] if "port=" in dsn else "?",
        }
        _append_index(index_path, row)
        print(json.dumps({k: row[k] for k in row if k != "warm_start"}), flush=True)
        n_ok += 1
    return n_ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-tasks", type=int, default=100, help="First N tasks from CEB list (before sharding)")
    p.add_argument("--task-offset", type=int, default=0)
    p.add_argument("--shard-index", type=int, default=0, help="Worker id for parallel Postgres shards")
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--n-trials", type=int, default=80)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--dsn",
        default=os.environ.get(
            "QUERYOPT_PG_DSN",
            "host=127.0.0.1 port=5432 dbname=imdb user=imdb password=imdb",
        ),
    )
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    p.add_argument("--arm", default="C")
    p.add_argument("--ckpt", type=Path, default=AUTHOR_CKPT)
    p.add_argument("--skip-done", action="store_true", default=True)
    p.add_argument("--no-skip-done", action="store_false", dest="skip_done")
    p.add_argument(
        "--optimizer",
        default="lolbo",
        help="lolbo (CMA-ES + e2e) | tr_tpe | tpe | cmaes",
    )
    p.add_argument(
        "--task-ids",
        default="",
        help="Comma-separated CEB task ids. Overrides --n-tasks/--offset/--shards.",
    )
    args = p.parse_args()
    os.environ["QUERYOPT_PG_DSN"] = args.dsn
    args.out_root.mkdir(parents=True, exist_ok=True)
    n = run_arm_c(
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
        arm=args.arm,
        optimizer=args.optimizer,
        task_ids=[t.strip() for t in str(args.task_ids).split(",") if t.strip()] or None,
    )
    print(json.dumps({"finished": n, "shard_index": args.shard_index}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
