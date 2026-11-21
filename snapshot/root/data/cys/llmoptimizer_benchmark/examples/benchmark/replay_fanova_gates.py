#!/usr/bin/env python3
"""Offline replay: compare 1D-proxy vs sklearn-RF vs full fANOVA on finished H2 trajs.

No new LLM calls. Reconstructs each slot's scored configs and reports which HPs
each engine would keep at selected prefixes.
"""

from __future__ import annotations

import ast
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_BASE = ROOT / "benchmark_data" / "real_hpo_v2_mltest20_algo40"
DEFAULT_ARM = "ptllm_tpe__P5plusA5T50X50__H2"
PREFIXES = (24, 35, 50, 80)
THRESH = 0.10


def _parse_config(raw: str) -> dict:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        try:
            val = ast.literal_eval(text)
            return val if isinstance(val, dict) else {}
        except Exception:
            return {}


def load_successes(traj: Path) -> list[dict]:
    rows = []
    with traj.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, rec in enumerate(reader):
            try:
                score = float(rec.get("perf") or "")
            except Exception:
                continue
            if score != score:
                continue
            cfg = _parse_config(rec.get("config_for_prompt") or "")
            if not cfg:
                continue
            rows.append({"trial": i + 1, "score": score, "config": cfg})
    return rows


def find_trajs(base: Path, arm: str) -> list[tuple[str, str, Path]]:
    out = []
    for task in ("cls", "rgs"):
        root = base / task
        if not root.is_dir():
            continue
        for ds_dir in sorted(root.iterdir()):
            run = ds_dir / arm
            if not run.is_dir():
                continue
            trajs = sorted(run.glob("CASH-*/*_traj.csv"))
            if trajs:
                out.append((task, ds_dir.name, trajs[0]))
    return out


def keep_set(imps: dict[str, float], thresh: float) -> set[str]:
    return {k for k, v in imps.items() if float(v) >= thresh}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--arm", default=DEFAULT_ARM)
    parser.add_argument("--thresh", type=float, default=THRESH)
    parser.add_argument("--max-slots", type=int, default=0)
    parser.add_argument(
        "--out",
        default=str(DEFAULT_BASE / "analysis" / "gate_replay" / "fanova_engine_compare.csv"),
    )
    args = parser.parse_args()

    from importlib.util import module_from_spec, spec_from_file_location

    gate_path = ROOT / "mindware" / "components" / "optimizers" / "experience_gate.py"
    spec = spec_from_file_location("experience_gate", gate_path)
    gate = module_from_spec(spec)
    sys.modules["experience_gate"] = gate
    assert spec.loader is not None
    spec.loader.exec_module(gate)
    default_fanova_python = gate.default_fanova_python
    hp_importances_full = gate.hp_importances_full
    hp_importances_proxy = gate.hp_importances_proxy
    hp_importances_rf = gate.hp_importances_rf

    py = default_fanova_python()
    print(f"FANOVA_PYTHON={py or '(missing)'}")

    trajs = find_trajs(Path(args.base), args.arm)
    if args.max_slots:
        trajs = trajs[: int(args.max_slots)]
    print(f"slots={len(trajs)} arm={args.arm}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "dataset",
        "n",
        "n_hps",
        "proxy_keep",
        "rf_keep",
        "full_keep",
        "full_engine",
        "jaccard_proxy_rf",
        "jaccard_proxy_full",
        "jaccard_rf_full",
        "proxy_top",
        "rf_top",
        "full_top",
        "proxy_json",
        "rf_json",
        "full_json",
    ]
    rows = []
    agree_pr = Counter()
    agree_pf = Counter()
    agree_rf = Counter()
    engine_used = Counter()

    for task, dataset, traj in trajs:
        successes = load_successes(traj)
        if len(successes) < 24:
            print(f"skip {task}/{dataset}: n={len(successes)}")
            continue
        for n in PREFIXES:
            if len(successes) < n:
                continue
            prefix = successes[:n]
            proxy = hp_importances_proxy(prefix)
            rf = hp_importances_rf(prefix)
            full, used = hp_importances_full(prefix)
            engine_used[used] += 1
            kp, kr, kf = keep_set(proxy, args.thresh), keep_set(rf, args.thresh), keep_set(full, args.thresh)
            jp, jf, jr = jaccard(kp, kr), jaccard(kp, kf), jaccard(kr, kf)
            agree_pr[round(jp, 2)]
            rows.append(
                {
                    "task": task,
                    "dataset": dataset,
                    "n": n,
                    "n_hps": max(len(proxy), len(rf), len(full)),
                    "proxy_keep": ";".join(sorted(kp)) or "-",
                    "rf_keep": ";".join(sorted(kr)) or "-",
                    "full_keep": ";".join(sorted(kf)) or "-",
                    "full_engine": used,
                    "jaccard_proxy_rf": f"{jp:.3f}",
                    "jaccard_proxy_full": f"{jf:.3f}",
                    "jaccard_rf_full": f"{jr:.3f}",
                    "proxy_top": _top(proxy),
                    "rf_top": _top(rf),
                    "full_top": _top(full),
                    "proxy_json": json.dumps({k: round(v, 3) for k, v in sorted(proxy.items())}),
                    "rf_json": json.dumps({k: round(v, 3) for k, v in sorted(rf.items())}),
                    "full_json": json.dumps({k: round(v, 3) for k, v in sorted(full.items())}),
                }
            )
            print(
                f"{task}/{dataset} n={n:2d} engine={used:14s} "
                f"keep proxy={len(kp)} rf={len(kr)} full={len(kf)} "
                f"J(p,f)={jf:.2f} J(r,f)={jr:.2f}"
            )

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    if rows:
        def mean_j(key: str) -> float:
            return sum(float(r[key]) for r in rows) / len(rows)

        print()
        print(f"wrote {out_path} ({len(rows)} rows)")
        print(f"mean Jaccard proxy-rf   {mean_j('jaccard_proxy_rf'):.3f}")
        print(f"mean Jaccard proxy-full {mean_j('jaccard_proxy_full'):.3f}")
        print(f"mean Jaccard rf-full    {mean_j('jaccard_rf_full'):.3f}")
        print("full engine counts:", dict(engine_used))
    return 0


def _top(imps: dict[str, float], k: int = 3) -> str:
    ranked = sorted(imps.items(), key=lambda kv: kv[1], reverse=True)[:k]
    if not ranked:
        return "-"
    return ";".join(f"{n}={v:.2f}" for n, v in ranked)


if __name__ == "__main__":
    raise SystemExit(main())
