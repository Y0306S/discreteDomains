#!/usr/bin/env python3
"""Phase B: full per-algo GP landscape cache.

Gated by Phase A winner JSON. Use --force to build anyway (e.g. after a smoke run).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.components.experience.retrieve import ExperienceBank
from mindware.components.experience.surrogate_landscape import (
    LandscapeCache,
    default_cache_path,
    get_algo_cs,
    predict_landscape,
    sample_query_configs,
    spearman_sim,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=ROOT / "history_data" / "experience_bank_v2.jsonl")
    p.add_argument(
        "--winner",
        type=Path,
        default=ROOT
        / "benchmark_data"
        / "real_hpo_v2_mltest20_algo40"
        / "analysis"
        / "lodo_similarity"
        / "surrogate_landscape_winner.json",
    )
    p.add_argument("--n-query-configs", type=int, default=None)
    p.add_argument("--min-prefix-trials", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--cache-tag", default="", help="Suffix after surrogate_landscape, e.g. holdout")
    p.add_argument("--max-datasets", type=int, default=None, help="Smoke: cap datasets per algo")
    args = p.parse_args(argv)

    n_query = args.n_query_configs
    min_prefix = args.min_prefix_trials
    hyper: dict = {}
    if args.winner.is_file():
        payload = json.loads(args.winner.read_text(encoding="utf-8"))
        if not args.force and payload.get("pass_gate") is False:
            print("Phase A LOCK gate failed; skip production cache (pass --force to override).")
            return 2
        winner = payload.get("winner") or {}
        n_query = n_query or int(winner.get("n_query_configs") or 50)
        min_prefix = min_prefix or int(winner.get("min_prefix_trials") or 24)
        hyper = dict(winner)
    n_query = int(n_query or 50)
    min_prefix = int(min_prefix or 24)

    bank = ExperienceBank.load(args.bank)
    by: dict = {}
    for exp in bank.experiences:
        prob = exp.get("problem") or {}
        algo = str(prob.get("algorithm_id") or "").lower()
        task = str(prob.get("task_type") or "").lower()
        ds = str(prob.get("dataset_id") or "").lower()
        trials = (exp.get("retrieval") or {}).get("trials") or []
        if not algo or not ds or len(trials) < min_prefix:
            continue
        by.setdefault((task, algo), []).append((ds, trials))

    for (task, algo), rows in by.items():
        prefix = "cls" if "class" in task else "rgs"
        tag = f"_{args.cache_tag}" if args.cache_tag else ""
        cache = LandscapeCache(
            ROOT / "history_data" / "meta_resource" / f"surrogate_landscape{tag}_{prefix}.pkl"
        )
        cache.payload.setdefault("hyperparams", {}).update(
            {"n_query_configs": n_query, "min_prefix_trials": min_prefix, **hyper}
        )
        try:
            cs = get_algo_cs(algo, task)
        except Exception as exc:
            print(f"skip {task}/{algo}: {exc}")
            continue
        qcfgs = sample_query_configs(cs, n_query)
        cache.payload.setdefault("sampled_configurations", {})[algo] = n_query
        if args.max_datasets:
            rows = rows[: args.max_datasets]
        vecs = {}
        for ds, trials in rows:
            pred = predict_landscape(trials, cs, algo, qcfgs, min_trials=min_prefix)
            if pred is None:
                continue
            cache.set_pred(algo, ds, pred)
            vecs[ds] = pred
        sim = {}
        keys = list(vecs)
        for i, d1 in enumerate(keys):
            sim.setdefault(d1, {})
            sim[d1][d1] = 1.0
            for d2 in keys[i + 1 :]:
                s = spearman_sim(vecs[d1], vecs[d2])
                sim.setdefault(d1, {})[d2] = s
                sim.setdefault(d2, {})[d1] = s
        cache.payload.setdefault("similarity", {})[algo] = sim
        cache.save()
        print(f"wrote {cache.path} algo={algo} n={len(vecs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Phase B: full per-algo GP landscape cache.

Gated by Phase A winner JSON. Use --force to build anyway (e.g. after a smoke run).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.components.experience.retrieve import ExperienceBank
from mindware.components.experience.surrogate_landscape import (
    LandscapeCache,
    default_cache_path,
    get_algo_cs,
    predict_landscape,
    sample_query_configs,
    spearman_sim,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=ROOT / "history_data" / "experience_bank_v2.jsonl")
    p.add_argument(
        "--winner",
        type=Path,
        default=ROOT
        / "benchmark_data"
        / "real_hpo_v2_mltest20_algo40"
        / "analysis"
        / "lodo_similarity"
        / "surrogate_landscape_winner.json",
    )
    p.add_argument("--n-query-configs", type=int, default=None)
    p.add_argument("--min-prefix-trials", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--cache-tag", default="", help="Suffix after surrogate_landscape, e.g. holdout")
    p.add_argument("--max-datasets", type=int, default=None, help="Smoke: cap datasets per algo")
    args = p.parse_args(argv)

    n_query = args.n_query_configs
    min_prefix = args.min_prefix_trials
    hyper: dict = {}
    if args.winner.is_file():
        payload = json.loads(args.winner.read_text(encoding="utf-8"))
        if not args.force and payload.get("pass_gate") is False:
            print("Phase A LOCK gate failed; skip production cache (pass --force to override).")
            return 2
        winner = payload.get("winner") or {}
        n_query = n_query or int(winner.get("n_query_configs") or 50)
        min_prefix = min_prefix or int(winner.get("min_prefix_trials") or 24)
        hyper = dict(winner)
    n_query = int(n_query or 50)
    min_prefix = int(min_prefix or 24)

    bank = ExperienceBank.load(args.bank)
    by: dict = {}
    for exp in bank.experiences:
        prob = exp.get("problem") or {}
        algo = str(prob.get("algorithm_id") or "").lower()
        task = str(prob.get("task_type") or "").lower()
        ds = str(prob.get("dataset_id") or "").lower()
        trials = (exp.get("retrieval") or {}).get("trials") or []
        if not algo or not ds or len(trials) < min_prefix:
            continue
        by.setdefault((task, algo), []).append((ds, trials))

    for (task, algo), rows in by.items():
        prefix = "cls" if "class" in task else "rgs"
        tag = f"_{args.cache_tag}" if args.cache_tag else ""
        cache = LandscapeCache(
            ROOT / "history_data" / "meta_resource" / f"surrogate_landscape{tag}_{prefix}.pkl"
        )
        cache.payload.setdefault("hyperparams", {}).update(
            {"n_query_configs": n_query, "min_prefix_trials": min_prefix, **hyper}
        )
        try:
            cs = get_algo_cs(algo, task)
        except Exception as exc:
            print(f"skip {task}/{algo}: {exc}")
            continue
        qcfgs = sample_query_configs(cs, n_query)
        cache.payload.setdefault("sampled_configurations", {})[algo] = n_query
        if args.max_datasets:
            rows = rows[: args.max_datasets]
        vecs = {}
        for ds, trials in rows:
            pred = predict_landscape(trials, cs, algo, qcfgs, min_trials=min_prefix)
            if pred is None:
                continue
            cache.set_pred(algo, ds, pred)
            vecs[ds] = pred
        sim = {}
        keys = list(vecs)
        for i, d1 in enumerate(keys):
            sim.setdefault(d1, {})
            sim[d1][d1] = 1.0
            for d2 in keys[i + 1 :]:
                s = spearman_sim(vecs[d1], vecs[d2])
                sim.setdefault(d1, {})[d2] = s
                sim.setdefault(d2, {})[d1] = s
        cache.payload.setdefault("similarity", {})[algo] = sim
        cache.save()
        print(f"wrote {cache.path} algo={algo} n={len(vecs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
