#!/usr/bin/env python3
"""Arm A/B/C probe on 12 CEB tasks.

Arm A: author ckpt, no layers, TPE, FROM-list init.
Arm B: fine-tuned ckpt + layers B/C + LOLBO + 50 warm-start EXPLAINs.
Arm C2: author ckpt + layers B/C + LOLBO + fixed decode (no silent fallback).
Arm C3: fine-tuned ckpt + layers B/C + LOLBO + fixed decode.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from mindware.domains import get_adapter
from mindware.domains.bolt_vae import QueryBoltVAE
from mindware.domains.queryopt.ceb import load_sql
from mindware.domains.queryopt.sql_parse import (
    is_connected_order,
    is_legal_order,
    random_connected_order,
)
from mindware.domains.search import run_domain_search

PROBE_TASKS = [
    "CEB_10A35",
    "CEB_10A9",
    "CEB_1A1229",
    "CEB_1A2517",
    "CEB_11A102",
    "CEB_2B118",
    "CEB_2A412",
    "CEB_8A94",
    "CEB_8A121",
    "CEB_6A209",
    "CEB_7A119",
    "CEB_7A10",
]

AUTHOR_CKPT = Path("/root/data/cys/BayesQO/bayes_lqo/vae/CEB_64.ckpt")
FT_CKPT = Path(
    "/root/data/cys/BOLT/optimization/query_plans/query_plan_optimization/vae/CEB_64_ft.ckpt"
)
OUT_ROOT = ROOT / "benchmark_data" / "domain_query" / "probe_layers_ab"


def _safe_decode(ad, z_cfg) -> list | None:
    try:
        return ad.decode(z_cfg)
    except ValueError:
        return None


def _roundtrip(ad, vae, n_rand: int = 20) -> dict:
    alias_to_table = {a: t for t, a in ad.pairs}
    default_tables = [alias_to_table[a] for a in ad.aliases]
    z0 = vae.encode(default_tables)
    dec0 = _safe_decode(ad, {f"z_{i}": z0[i] for i in range(len(z0))}) or []
    rng = np.random.RandomState(0)
    perm = random_connected_order(ad.aliases, ad.join_graph, rng)
    z1 = ad.encode(perm)
    dec1 = (
        _safe_decode(ad, {f"z_{i}": z1[i] for i in range(len(z1))}) if z1 else []
    ) or []
    hits_legal = 0
    hits_connected = 0
    hits_full = 0
    hits_decode = 0
    unique = set()
    for _ in range(n_rand):
        z = {f"z_{j}": float(rng.randn()) for j in range(ad.latent_dim)}
        order = _safe_decode(ad, z)
        if order is None:
            unique.add(None)
            continue
        hits_decode += 1
        unique.add(tuple(order))
        if is_legal_order(order, ad.aliases):
            hits_legal += 1
        if is_connected_order(order, ad.join_graph):
            hits_connected += 1
        if set(order) == set(ad.aliases) and len(order) == len(ad.aliases):
            hits_full += 1
    return {
        "n_tables": len(ad.aliases),
        "default_roundtrip_ok": dec0 == ad.aliases,
        "default_decoded": dec0,
        "connected_roundtrip_ok": dec1 == perm,
        "rand20_decode_ok": hits_decode,
        "rand20_legal": hits_legal,
        "rand20_connected": hits_connected,
        "rand20_full_alias_set": hits_full,
        "rand20_unique": len(unique),
        "query_layers": bool(ad.query_layers),
    }


def _run_arm(arm: str, ckpt: Path, device: str, n_trials: int, dsn: str, out_dir: Path) -> list:
    vae = QueryBoltVAE(ckpt, device=device)
    rows = []
    layers = arm in {"B", "C", "C2", "C3"}
    opt = "lolbo" if layers else "tpe"
    for i, tid in enumerate(PROBE_TASKS):
        sql = load_sql(tid)
        ad = get_adapter(
            "query",
            task={"task_id": tid, "sql": sql, "seed": 1},
            vae=vae,
            dummy=False,
            dsn=dsn,
            latent_bound=4.0,
            query_layers=layers,
            log_cost=layers,
        )
        rt = _roundtrip(ad, vae)
        init_z = None
        warm = None
        t0 = time.time()
        if layers:
            init_z, best_order, best_cost = ad.warm_start(n_perm=50)
            warm = {"order": best_order, "raw_cost": best_cost, "init_z": init_z}
        else:
            try:
                init_z = vae.encode([t for t, _ in ad.pairs])
            except Exception as exc:
                print(f"[{arm}] encode failed {tid}: {exc}")
        task_out = out_dir / arm / tid / opt / "seed1"
        result = run_domain_search(
            ad,
            n_trials=n_trials,
            seed=1 + i,
            optimizer=opt,
            output_dir=task_out,
            init_z=init_z,
            warm_start=warm,
            n_startup_trials=min(6, max(3, n_trials // 4)),
        )
        obs = result.get("observations") or []
        raws = []
        for o in obs:
            info = (o.get("extra_info") or {}).get("info") or {}
            raws.append(float(info.get("raw_cost", o.get("objectives", [1e10])[0])))
        row = {
            "arm": arm,
            "task_id": tid,
            "n_tables": rt["n_tables"],
            "ckpt": str(ckpt),
            "optimizer": opt,
            "query_layers": layers,
            "roundtrip": rt,
            "warm_start": warm,
            "n_trials": n_trials,
            "best_objective": result.get("best_objective"),
            "best_raw_cost": min(raws) if raws else None,
            "seconds": round(time.time() - t0, 2),
            "history": result.get("history_path"),
        }
        rows.append(row)
        print(json.dumps({k: row[k] for k in row if k != "roundtrip"}))
        print(json.dumps({"task_id": tid, "arm": arm, "roundtrip": rt}))
    return rows


def _load_summary(path: Path) -> list:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _merge_summary(existing: list, new_rows: list) -> list:
    by_key = {(r.get("arm"), r.get("task_id")): r for r in existing}
    for row in new_rows:
        by_key[(row.get("arm"), row.get("task_id"))] = row
    return sorted(by_key.values(), key=lambda r: (r.get("arm", ""), PROBE_TASKS.index(r.get("task_id", ""))))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=["A", "B", "C", "C2", "C3", "both"], default="both")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--dsn",
        default=os.environ.get(
            "QUERYOPT_PG_DSN",
            "host=127.0.0.1 port=5432 dbname=imdb user=imdb password=imdb",
        ),
    )
    args = p.parse_args()
    os.environ["QUERYOPT_PG_DSN"] = args.dsn
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    if args.arm in {"A", "both"}:
        print("[probe] Arm A base no layers", AUTHOR_CKPT)
        all_rows.extend(_run_arm("A", AUTHOR_CKPT, args.device, args.n_trials, args.dsn, OUT_ROOT))
    if args.arm in {"B", "both"}:
        if not FT_CKPT.is_file() or FT_CKPT.stat().st_size < 10_000:
            raise SystemExit(f"Arm B needs fine-tuned ckpt at {FT_CKPT}")
        print("[probe] Arm B layers+FT", FT_CKPT)
        all_rows.extend(_run_arm("B", FT_CKPT, args.device, args.n_trials, args.dsn, OUT_ROOT))
    if args.arm == "C":
        print("[probe] Arm C layers+author (no FT)", AUTHOR_CKPT)
        all_rows.extend(_run_arm("C", AUTHOR_CKPT, args.device, args.n_trials, args.dsn, OUT_ROOT))
    if args.arm == "C2":
        print("[probe] Arm C2 layers+author fixed-decode", AUTHOR_CKPT)
        all_rows.extend(_run_arm("C2", AUTHOR_CKPT, args.device, args.n_trials, args.dsn, OUT_ROOT))
    if args.arm == "C3":
        if not FT_CKPT.is_file() or FT_CKPT.stat().st_size < 10_000:
            raise SystemExit(f"Arm C3 needs fine-tuned ckpt at {FT_CKPT}")
        print("[probe] Arm C3 layers+FT fixed-decode", FT_CKPT)
        all_rows.extend(_run_arm("C3", FT_CKPT, args.device, args.n_trials, args.dsn, OUT_ROOT))
    summary = OUT_ROOT / "summary.json"
    merged = _merge_summary(_load_summary(summary), all_rows)
    summary.write_text(json.dumps(merged, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[probe] wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Arm A/B/C probe on 12 CEB tasks.

Arm A: author ckpt, no layers, TPE, FROM-list init.
Arm B: fine-tuned ckpt + layers B/C + LOLBO + 50 warm-start EXPLAINs.
Arm C2: author ckpt + layers B/C + LOLBO + fixed decode (no silent fallback).
Arm C3: fine-tuned ckpt + layers B/C + LOLBO + fixed decode.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from mindware.domains import get_adapter
from mindware.domains.bolt_vae import QueryBoltVAE
from mindware.domains.queryopt.ceb import load_sql
from mindware.domains.queryopt.sql_parse import (
    is_connected_order,
    is_legal_order,
    random_connected_order,
)
from mindware.domains.search import run_domain_search

PROBE_TASKS = [
    "CEB_10A35",
    "CEB_10A9",
    "CEB_1A1229",
    "CEB_1A2517",
    "CEB_11A102",
    "CEB_2B118",
    "CEB_2A412",
    "CEB_8A94",
    "CEB_8A121",
    "CEB_6A209",
    "CEB_7A119",
    "CEB_7A10",
]

AUTHOR_CKPT = Path("/root/data/cys/BayesQO/bayes_lqo/vae/CEB_64.ckpt")
FT_CKPT = Path(
    "/root/data/cys/BOLT/optimization/query_plans/query_plan_optimization/vae/CEB_64_ft.ckpt"
)
OUT_ROOT = ROOT / "benchmark_data" / "domain_query" / "probe_layers_ab"


def _safe_decode(ad, z_cfg) -> list | None:
    try:
        return ad.decode(z_cfg)
    except ValueError:
        return None


def _roundtrip(ad, vae, n_rand: int = 20) -> dict:
    alias_to_table = {a: t for t, a in ad.pairs}
    default_tables = [alias_to_table[a] for a in ad.aliases]
    z0 = vae.encode(default_tables)
    dec0 = _safe_decode(ad, {f"z_{i}": z0[i] for i in range(len(z0))}) or []
    rng = np.random.RandomState(0)
    perm = random_connected_order(ad.aliases, ad.join_graph, rng)
    z1 = ad.encode(perm)
    dec1 = (
        _safe_decode(ad, {f"z_{i}": z1[i] for i in range(len(z1))}) if z1 else []
    ) or []
    hits_legal = 0
    hits_connected = 0
    hits_full = 0
    hits_decode = 0
    unique = set()
    for _ in range(n_rand):
        z = {f"z_{j}": float(rng.randn()) for j in range(ad.latent_dim)}
        order = _safe_decode(ad, z)
        if order is None:
            unique.add(None)
            continue
        hits_decode += 1
        unique.add(tuple(order))
        if is_legal_order(order, ad.aliases):
            hits_legal += 1
        if is_connected_order(order, ad.join_graph):
            hits_connected += 1
        if set(order) == set(ad.aliases) and len(order) == len(ad.aliases):
            hits_full += 1
    return {
        "n_tables": len(ad.aliases),
        "default_roundtrip_ok": dec0 == ad.aliases,
        "default_decoded": dec0,
        "connected_roundtrip_ok": dec1 == perm,
        "rand20_decode_ok": hits_decode,
        "rand20_legal": hits_legal,
        "rand20_connected": hits_connected,
        "rand20_full_alias_set": hits_full,
        "rand20_unique": len(unique),
        "query_layers": bool(ad.query_layers),
    }


def _run_arm(arm: str, ckpt: Path, device: str, n_trials: int, dsn: str, out_dir: Path) -> list:
    vae = QueryBoltVAE(ckpt, device=device)
    rows = []
    layers = arm in {"B", "C", "C2", "C3"}
    opt = "lolbo" if layers else "tpe"
    for i, tid in enumerate(PROBE_TASKS):
        sql = load_sql(tid)
        ad = get_adapter(
            "query",
            task={"task_id": tid, "sql": sql, "seed": 1},
            vae=vae,
            dummy=False,
            dsn=dsn,
            latent_bound=4.0,
            query_layers=layers,
            log_cost=layers,
        )
        rt = _roundtrip(ad, vae)
        init_z = None
        warm = None
        t0 = time.time()
        if layers:
            init_z, best_order, best_cost = ad.warm_start(n_perm=50)
            warm = {"order": best_order, "raw_cost": best_cost, "init_z": init_z}
        else:
            try:
                init_z = vae.encode([t for t, _ in ad.pairs])
            except Exception as exc:
                print(f"[{arm}] encode failed {tid}: {exc}")
        task_out = out_dir / arm / tid / opt / "seed1"
        result = run_domain_search(
            ad,
            n_trials=n_trials,
            seed=1 + i,
            optimizer=opt,
            output_dir=task_out,
            init_z=init_z,
            warm_start=warm,
            n_startup_trials=min(6, max(3, n_trials // 4)),
        )
        obs = result.get("observations") or []
        raws = []
        for o in obs:
            info = (o.get("extra_info") or {}).get("info") or {}
            raws.append(float(info.get("raw_cost", o.get("objectives", [1e10])[0])))
        row = {
            "arm": arm,
            "task_id": tid,
            "n_tables": rt["n_tables"],
            "ckpt": str(ckpt),
            "optimizer": opt,
            "query_layers": layers,
            "roundtrip": rt,
            "warm_start": warm,
            "n_trials": n_trials,
            "best_objective": result.get("best_objective"),
            "best_raw_cost": min(raws) if raws else None,
            "seconds": round(time.time() - t0, 2),
            "history": result.get("history_path"),
        }
        rows.append(row)
        print(json.dumps({k: row[k] for k in row if k != "roundtrip"}))
        print(json.dumps({"task_id": tid, "arm": arm, "roundtrip": rt}))
    return rows


def _load_summary(path: Path) -> list:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _merge_summary(existing: list, new_rows: list) -> list:
    by_key = {(r.get("arm"), r.get("task_id")): r for r in existing}
    for row in new_rows:
        by_key[(row.get("arm"), row.get("task_id"))] = row
    return sorted(by_key.values(), key=lambda r: (r.get("arm", ""), PROBE_TASKS.index(r.get("task_id", ""))))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=["A", "B", "C", "C2", "C3", "both"], default="both")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--dsn",
        default=os.environ.get(
            "QUERYOPT_PG_DSN",
            "host=127.0.0.1 port=5432 dbname=imdb user=imdb password=imdb",
        ),
    )
    args = p.parse_args()
    os.environ["QUERYOPT_PG_DSN"] = args.dsn
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    if args.arm in {"A", "both"}:
        print("[probe] Arm A base no layers", AUTHOR_CKPT)
        all_rows.extend(_run_arm("A", AUTHOR_CKPT, args.device, args.n_trials, args.dsn, OUT_ROOT))
    if args.arm in {"B", "both"}:
        if not FT_CKPT.is_file() or FT_CKPT.stat().st_size < 10_000:
            raise SystemExit(f"Arm B needs fine-tuned ckpt at {FT_CKPT}")
        print("[probe] Arm B layers+FT", FT_CKPT)
        all_rows.extend(_run_arm("B", FT_CKPT, args.device, args.n_trials, args.dsn, OUT_ROOT))
    if args.arm == "C":
        print("[probe] Arm C layers+author (no FT)", AUTHOR_CKPT)
        all_rows.extend(_run_arm("C", AUTHOR_CKPT, args.device, args.n_trials, args.dsn, OUT_ROOT))
    if args.arm == "C2":
        print("[probe] Arm C2 layers+author fixed-decode", AUTHOR_CKPT)
        all_rows.extend(_run_arm("C2", AUTHOR_CKPT, args.device, args.n_trials, args.dsn, OUT_ROOT))
    if args.arm == "C3":
        if not FT_CKPT.is_file() or FT_CKPT.stat().st_size < 10_000:
            raise SystemExit(f"Arm C3 needs fine-tuned ckpt at {FT_CKPT}")
        print("[probe] Arm C3 layers+FT fixed-decode", FT_CKPT)
        all_rows.extend(_run_arm("C3", FT_CKPT, args.device, args.n_trials, args.dsn, OUT_ROOT))
    summary = OUT_ROOT / "summary.json"
    merged = _merge_summary(_load_summary(summary), all_rows)
    summary.write_text(json.dumps(merged, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[probe] wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
