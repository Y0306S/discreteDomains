"""Dump train-80 classical BO trajectories into FSBO/RGPE group files.

Primary source is the OpenBox archive
``history_data/history_data/<tt>/<metric>/<dataset>/<algorithm>.json``
(one ordered study per pair, no optimizer-variant subfolder). Live Wave 1
writes under ``history_data/<tt>/.../<variant>/`` are not used, to avoid
leaking test-20 eval trajs and colliding with running jobs.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from mindware.components.optimizers.transfer_hpo.schema import (
    SOURCE_VARIANT_PRIORITY,
    all_group_ids,
    build_hpo_cs,
    encode_config,
    group_id,
    schema_spec,
    configspace_spec,
    schema_fingerprint,
    vector_dim,
)
from mindware.components.optimizers.transfer_hpo.protocol import (
    HISTORY_ROOT, TRAIN_MANIFEST, allowed_task_ids, checked_source_path, sha256_file,
)

DEFAULT_TRAIN_JSONL = Path(
    "/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/ml_train_80_fixed.jsonl"
)
DEFAULT_TEST_JSONL = Path(
    "/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/ml_test_20_fixed.jsonl"
)
DEFAULT_HISTORY = Path(
    "/root/blockdata/llmoptimizer_benchmark/history_data/history_data"
)


def load_pairs(jsonl: Path) -> Set[Tuple[str, str, str]]:
    out: Set[Tuple[str, str, str]] = set()
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            m = rec.get("metadata") or rec
            out.add((str(m["task_type"]), str(m["dataset"]), str(m["algorithm"])))
    return out


def _metric(tt: str) -> str:
    return "bal_acc" if tt == "cls" else "mse"


def _obs_y(obs: Dict[str, Any], task_type: str) -> Optional[Tuple[float, float]]:
    """Return (y_maximize, y_minimize) or None if the trial failed."""
    try:
        obj = obs.get("objectives")
        if isinstance(obj, (list, tuple)):
            y_min = float(obj[0])
        else:
            y_min = float(obj)
    except Exception:
        return None
    if y_min != y_min or y_min <= -1e18 or y_min >= 1e18:
        return None
    # OpenBox stores minimize: cls is -bal_acc, rgs is mse.
    return (-y_min, y_min)


def _load_history_json(path: Path) -> List[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, dict):
        return list(payload.get("observations") or [])
    return []


def _n_obs(path: Path) -> int:
    return len(_load_history_json(path))


def pick_source_file(hist_root: Path, tt: str, dataset: str, algorithm: str) -> Optional[Tuple[str, Path, int]]:
    """Prefer the archive ``<dataset>/<algorithm>.json``; then variant folders."""
    base = hist_root / tt / _metric(tt) / dataset
    archive = base / f"{algorithm}.json"
    if archive.is_file():
        checked_source_path(archive)
        n = _n_obs(archive)
        if n > 0:
            return "archive", archive, n
    if not base.is_dir():
        return None
    ranked: List[Tuple[int, int, str, Path]] = []
    for idx, variant in enumerate(SOURCE_VARIANT_PRIORITY):
        path = base / variant / f"{algorithm}.json"
        if not path.is_file():
            continue
        checked_source_path(path)
        n = _n_obs(path)
        if n <= 0:
            continue
        ranked.append((idx, -n, variant, path))
    if not ranked:
        return None
    ranked.sort()
    _prio, neg_n, variant, path = ranked[0]
    return variant, path, -neg_n


def export_group(
    tt: str,
    algorithm: str,
    train_pairs: Iterable[Tuple[str, str, str]],
    hist_root: Path,
    min_obs: int = 10,
    schema_kwargs=None,
) -> Dict[str, Any]:
    cs = build_hpo_cs(tt, algorithm, **(schema_kwargs or {}))
    gid = group_id(tt, algorithm)
    tasks: Dict[str, Any] = {}
    skipped_encode = 0
    skipped_short = 0
    missing = 0
    n_train = 0
    exclusions = []
    for ptt, dataset, algo in sorted(train_pairs):
        if ptt != tt or algo != algorithm:
            continue
        n_train += 1
        picked = pick_source_file(hist_root, tt, dataset, algorithm)
        if picked is None:
            missing += 1
            exclusions.append({"task_id": f"{tt}/{dataset}/{algo}", "reason": "missing_history"})
            continue
        variant, path, _nfile = picked
        rows_x: List[List[float]] = []
        y_max: List[float] = []
        y_min: List[float] = []
        cfgs: List[Dict[str, Any]] = []
        times: List[str] = []
        raw_obs = _load_history_json(path)
        raw_obs.sort(key=lambda o: str(o.get("create_time") or ""))
        for obs in raw_obs:
            state = obs.get("trial_state")
            if state not in (None, 0, "0"):
                continue
            ys = _obs_y(obs, tt)
            if ys is None:
                continue
            cfg = obs.get("config") or {}
            if not isinstance(cfg, dict):
                skipped_encode += 1
                continue
            vec = encode_config(cs, cfg, algorithm)
            if vec is None:
                skipped_encode += 1
                continue
            rows_x.append(vec.tolist())
            y_max.append(float(ys[0]))
            y_min.append(float(ys[1]))
            cfgs.append(dict(cfg))
            times.append(str(obs.get("create_time") or ""))
        if len(rows_x) < min_obs:
            skipped_short += 1
            exclusions.append({
                "task_id": f"{tt}/{dataset}/{algo}", "reason": "too_few_valid_observations",
                "n_valid": len(rows_x), "n_raw": len(raw_obs), "source_path": str(path),
            })
            continue
        tid = f"{tt}/{dataset}/{algorithm}"
        tasks[tid] = {
            "dataset": dataset,
            "variant": variant,
            "source_path": str(path),
            "source_sha256": sha256_file(path),
            "n": len(rows_x),
            "X": rows_x,
            "y_maximize": y_max,
            "y_minimize": y_min,
            "configs": cfgs,
            "times": times,
        }
    n_full = sum(1 for t in tasks.values() if t["n"] >= 80)
    return {
        "format_version": 2,
        "group": gid,
        "task_type": tt,
        "algorithm": algorithm,
        "dim": vector_dim(cs),
        "schema": schema_spec(cs),
        "configspace": configspace_spec(cs),
        "schema_fingerprint": schema_fingerprint(cs),
        "defaults": dict(cs.get_default_configuration()),
        "train_manifest_sha256": sha256_file(TRAIN_MANIFEST),
        "n_train_slots": n_train,
        "n_source_tasks": len(tasks),
        "n_full_80": n_full,
        "n_missing_history": missing,
        "n_skipped_short": skipped_short,
        "n_skipped_encode": skipped_encode,
        "min_obs": min_obs,
        "ready_for_meta_train": len(tasks) >= 2,
        "exclusions": exclusions,
        "tasks": tasks,
    }


def export_all(
    out_dir: Path,
    train_jsonl: Path = DEFAULT_TRAIN_JSONL,
    test_jsonl: Path = DEFAULT_TEST_JSONL,
    hist_root: Path = DEFAULT_HISTORY,
    min_obs: int = 10,
) -> Dict[str, Any]:
    train_pairs = load_pairs(train_jsonl)
    test_pairs = load_pairs(test_jsonl)
    permitted = allowed_task_ids()
    if any("%s/%s/%s" % pair not in permitted for pair in train_pairs):
        raise ValueError("Training sources must be a subset of ml_train_80_fixed")
    if train_pairs & test_pairs:
        raise ValueError("Training and test tasks overlap")
    if hist_root.resolve() != HISTORY_ROOT.resolve():
        raise ValueError("Only the specified history_data/history_data root is permitted")
    out_dir.mkdir(parents=True, exist_ok=True)
    opt_dir = out_dir / "optformer"
    opt_dir.mkdir(parents=True, exist_ok=True)
    seq_path = opt_dir / "sequences.jsonl"
    coverage: List[Dict[str, Any]] = []
    n_seq = 0
    with seq_path.open("w", encoding="utf-8") as seq_out:
        groups = [(gid, "", {}) for gid in all_group_ids()]
        groups += [("rgs:" + algo, "_no_poisson", {"y_neg_mask": False})
                   for algo in ("extra_trees", "random_forest")]
        for gid, suffix, schema_kwargs in groups:
            tt, algo = gid.split(":", 1)
            payload = export_group(tt, algo, train_pairs, hist_root, min_obs=min_obs, schema_kwargs=schema_kwargs)
            if suffix:
                payload["schema_variant"] = "no_poisson"
                payload["note"] = "Allowed source observations filtered to the target-compatible non-Poisson schema."
            dump_path = out_dir / f"{tt}_{algo}{suffix}.json"
            dump_path.write_text(json.dumps(payload), encoding="utf-8")
            variants = Counter(t["variant"] for t in payload["tasks"].values())
            coverage.append(
                {
                    "group": gid + suffix,
                    "n_train_slots": payload["n_train_slots"],
                    "n_source_tasks": payload["n_source_tasks"],
                    "n_full_80": payload["n_full_80"],
                    "n_missing_history": payload["n_missing_history"],
                    "n_obs_total": int(sum(t["n"] for t in payload["tasks"].values())),
                    "dim": payload["dim"],
                    "ready_for_meta_train": payload["ready_for_meta_train"],
                    "source_variants": dict(variants),
                    "dump": str(dump_path),
                }
            )
            for tid, rec in (payload.get("tasks") or {}).items():
                if suffix:
                    continue
                seq_out.write(
                    json.dumps(
                        {
                            "task_id": tid,
                            "task_type": payload["task_type"],
                            "dataset": rec["dataset"],
                            "algorithm": payload["algorithm"],
                            "algorithm_condition": (
                                "unknown_mixed" if rec["variant"] == "archive" else rec["variant"]
                            ),
                            "source_path": rec["source_path"],
                            "source_sha256": rec["source_sha256"],
                            "configspace": payload["configspace"],
                            "schema_fingerprint": payload["schema_fingerprint"],
                            "n": rec["n"],
                            "X": rec["X"],
                            "y_maximize": rec["y_maximize"],
                            "configs": rec["configs"],
                            "times": rec.get("times") or [],
                            "ordered": True,
                            "shuffled": False,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                n_seq += 1
    summary = {
        "format_version": 2,
        "train_manifest_sha256": sha256_file(TRAIN_MANIFEST),
        "n_train_pairs": len(train_pairs),
        "n_test_pairs": len(test_pairs),
        "pair_overlap": len(train_pairs & test_pairs),
        "n_groups": len(coverage),
        "n_groups_ready": sum(1 for r in coverage if r["ready_for_meta_train"]),
        "hist_root": str(hist_root),
        "n_optformer_sequences": n_seq,
        "optformer_sequences": str(seq_path),
        "source_variant_priority": ["archive"] + list(SOURCE_VARIANT_PRIORITY),
        "note": (
            "Source tasks are ml_train_80_fixed (dataset, algorithm) pairs only, "
            "read from history_data/history_data archive studies (ordered "
            "OpenBox observations). Test-20 pairs are never used as sources."
        ),
        "groups": coverage,
    }
    (out_dir / "coverage.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    tasks_train = [
        {
            "task_type": tt,
            "dataset": ds,
            "algorithm": algo,
            "metric": "bal_acc" if tt == "cls" else "mse",
        }
        for tt, ds, algo in sorted(train_pairs)
    ]
    (out_dir / "tasks_train80.json").write_text(json.dumps(tasks_train, indent=2), encoding="utf-8")
    (opt_dir / "meta.json").write_text(
        json.dumps(
            {
                "n_sequences": n_seq,
                "dump_dir": str(out_dir),
                "sequences": str(seq_path),
                "note": (
                    "Each line is one ml_train_80_fixed pair, trials in create_time order. "
                    "OptFormer T5X train is a separate gated script."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary
