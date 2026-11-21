#!/usr/bin/env python3
"""ProteinGym GFP/AAV latent BO: landscape VAE + ESM surrogate (+ DMS hit metrics).

Arms:
  - tpe / lolbo: latent BO with surrogate oracle
  - random: random z baseline
  - greedy: local AA mutants of start seq scored by surrogate (discrete baseline)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains import get_adapter
from mindware.domains.paths import domain_bench
from mindware.domains.protein.surrogate import ESMFitnessSurrogate
from mindware.domains.protein.torch_vae import ProteinGymVAE
from mindware.domains.search import run_domain_search

PG_ROOT = ROOT / "history_data/domain_protein/proteingym"
VAE_DIR = PG_ROOT / "vae"
OUT_ROOT = domain_bench("protein") / "proteingym_bo"
AA = list("ACDEFGHIKLMNPQRSTVWY")


def _load_lookup(landscape: str) -> Dict[str, float]:
    p = PG_ROOT / landscape / "dms_lookup.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(d, dict) and "scores" in d:
        d = d["scores"]
    return {str(k): float(v) for k, v in d.items()}


def _top_starts(lookup: Dict[str, float], n: int, seed: int) -> List[str]:
    items = sorted(lookup.items(), key=lambda kv: kv[1], reverse=True)
    # take from top 5% with diversity
    top_n = max(n * 20, min(len(items), 2000))
    pool = [s for s, _ in items[:top_n]]
    rng = random.Random(seed)
    rng.shuffle(pool)
    out = []
    seen = set()
    for s in pool:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= n:
            break
    return out


def _greedy_mutants(start: str, surrogate: ESMFitnessSurrogate, n_trials: int, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    best_seq = start
    best_pred = -surrogate(start)[0]  # raw fitness
    hist = [{"trial": 0, "seq": start, "pred_fitness": best_pred, "source": "start"}]
    cur = start
    for t in range(1, n_trials + 1):
        i = rng.randrange(len(cur))
        choices = [a for a in AA if a != cur[i]]
        mut = cur[:i] + rng.choice(choices) + cur[i + 1 :]
        obj, hit = surrogate(mut)
        pred = -obj if hit else -1e6
        hist.append({"trial": t, "seq": mut, "pred_fitness": pred, "source": "greedy"})
        if hit and pred > best_pred:
            best_pred = pred
            best_seq = mut
            cur = mut
    return {"best_seq": best_seq, "best_pred_fitness": best_pred, "traj": hist}


def _summarize_obs(obs: list, lookup: Dict[str, float]) -> Dict[str, Any]:
    best_pred = None
    best_true = None
    hits = 0
    seqs = []
    for o in obs:
        info = (o.get("extra_info") or {}).get("info") or {}
        seq = info.get("sequence") or o.get("artifact")
        if seq:
            seqs.append(seq)
        pred = info.get("pred_fitness")
        if pred is not None:
            best_pred = pred if best_pred is None else max(best_pred, float(pred))
        true = info.get("true_dms")
        if true is not None:
            hits += 1
            best_true = true if best_true is None else max(best_true, float(true))
        elif seq and seq in lookup:
            hits += 1
            best_true = lookup[seq] if best_true is None else max(best_true, lookup[seq])
    return {
        "best_pred_fitness": best_pred,
        "best_true_dms": best_true,
        "dms_hit_rate": hits / max(len(obs), 1),
        "n_unique_seq": len(set(seqs)),
    }


def run_landscape(
    landscape: str,
    *,
    n_starts: int,
    n_trials: int,
    seed: int,
    device: str,
    optimizer: str,
    include_baselines: bool,
) -> list:
    lookup = _load_lookup(landscape)
    vae_path = VAE_DIR / f"{landscape}_vae.pt"
    if not vae_path.is_file():
        raise SystemExit(f"missing VAE {vae_path}")
    head = PG_ROOT / landscape / "esm_fitness_head.pt"
    cache = PG_ROOT / landscape / "esm_embed_cache.pt"
    if not head.is_file():
        raise SystemExit(f"missing surrogate head {head}")

    vae = ProteinGymVAE.load(vae_path, device=device)
    sur = ESMFitnessSurrogate(
        device=device,
        cache_path=cache if cache.is_file() else None,
        head_path=head,
        max_len=vae.max_len + 2,
    )
    starts = _top_starts(lookup, n_starts, seed)
    rows = []
    out_dir = OUT_ROOT / landscape
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, start in enumerate(starts):
        task_id = f"{landscape}_s{i:02d}"
        t0 = time.time()
        init_z = vae.encode(start)
        adapter = get_adapter(
            "protein",
            task={"task_id": task_id, "start_seq": start, "landscape": landscape, "device": device},
            vae=vae,
            surrogate=sur,
            oracle_kind="surrogate",
            dms_lookup=lookup,
            dummy=False,
            max_len=vae.max_len,
            latent_bound=4.0,
        )
        task_out = out_dir / task_id / optimizer / f"seed{seed}"
        result = run_domain_search(
            adapter,
            n_trials=n_trials,
            seed=seed + i,
            optimizer=optimizer,
            output_dir=task_out,
            init_z=init_z,
            n_startup_trials=min(8, max(3, n_trials // 4)),
        )
        stats = _summarize_obs(result.get("observations") or [], lookup)
        row = {
            "landscape": landscape,
            "task_id": task_id,
            "arm": optimizer,
            "start_seq": start,
            "start_true_dms": lookup.get(start),
            "n_trials": n_trials,
            "best_objective": result.get("best_objective"),
            "seconds": round(time.time() - t0, 2),
            **stats,
        }
        rows.append(row)
        print(json.dumps(row))

        if include_baselines:
            # random z
            t1 = time.time()
            adapter_r = get_adapter(
                "protein",
                task={"task_id": task_id + "_rand", "start_seq": start, "landscape": landscape, "device": device},
                vae=vae,
                surrogate=sur,
                oracle_kind="surrogate",
                dms_lookup=lookup,
                max_len=vae.max_len,
                latent_bound=4.0,
            )
            res_r = run_domain_search(
                adapter_r,
                n_trials=n_trials,
                seed=1000 + seed + i,
                optimizer="random",
                output_dir=out_dir / task_id / "random" / f"seed{seed}",
            )
            st_r = _summarize_obs(res_r.get("observations") or [], lookup)
            row_r = {
                "landscape": landscape,
                "task_id": task_id,
                "arm": "random",
                "start_true_dms": lookup.get(start),
                "n_trials": n_trials,
                "seconds": round(time.time() - t1, 2),
                **st_r,
            }
            rows.append(row_r)
            print(json.dumps(row_r))

            # greedy
            t2 = time.time()
            g = _greedy_mutants(start, sur, n_trials, seed=2000 + seed + i)
            true = lookup.get(g["best_seq"])
            row_g = {
                "landscape": landscape,
                "task_id": task_id,
                "arm": "greedy",
                "start_true_dms": lookup.get(start),
                "n_trials": n_trials,
                "best_pred_fitness": g["best_pred_fitness"],
                "best_true_dms": true,
                "dms_hit_rate": 1.0 if true is not None else 0.0,
                "n_unique_seq": len({h["seq"] for h in g["traj"]}),
                "seconds": round(time.time() - t2, 2),
            }
            (out_dir / task_id / "greedy" / f"seed{seed}").mkdir(parents=True, exist_ok=True)
            (out_dir / task_id / "greedy" / f"seed{seed}" / "traj.json").write_text(
                json.dumps(g, indent=2) + "\n", encoding="utf-8"
            )
            rows.append(row_g)
            print(json.dumps(row_g))

    summary_path = out_dir / f"summary_{optimizer}_n{n_starts}_t{n_trials}.json"
    summary_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(summary_path), "n_rows": len(rows)}))
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--landscape", choices=["gfp", "aav", "both"], default="both")
    p.add_argument("--n-starts", type=int, default=4)
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--optimizer", default="tpe", choices=["tpe", "lolbo", "cmaes", "random"])
    p.add_argument("--baselines", action="store_true", help="Also run random-z and greedy arms")
    args = p.parse_args()
    landscapes = ["gfp", "aav"] if args.landscape == "both" else [args.landscape]
    all_rows = []
    for ls in landscapes:
        all_rows.extend(
            run_landscape(
                ls,
                n_starts=args.n_starts,
                n_trials=args.n_trials,
                seed=args.seed,
                device=args.device,
                optimizer=args.optimizer,
                include_baselines=args.baselines,
            )
        )
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "summary_all.json").write_text(json.dumps(all_rows, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
