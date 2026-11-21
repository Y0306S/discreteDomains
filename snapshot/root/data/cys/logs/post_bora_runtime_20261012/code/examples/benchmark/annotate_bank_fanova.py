#!/usr/bin/env python3
"""Build experience_bank_v3.jsonl with official fANOVA + minimal prompt cards.

Does not overwrite v2 banks. Stores full importance_1d / importance_total so
gate configs can be replayed without re-running fANOVA.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from build_experience_bank import _iter_history_files
from mindware.components.experience.bank_gates import (
    DEFAULT_GATE_CONFIG,
    apply_gates,
    merge_gate_config,
)
from mindware.components.experience.contrastive import fit_bank_weights, save_weights
from mindware.components.experience.profile import load_dataset_info_table
from mindware.components.experience.retrieve import ExperienceBank


def eval_holdout_datasets(tasks_json: Path) -> set[str]:
    tasks = json.loads(Path(tasks_json).read_text(encoding="utf-8"))
    return {str(t.get("dataset") or "").lower() for t in tasks}


def _write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for exp in rows:
            f.write(json.dumps(exp, ensure_ascii=False) + "\n")


def _fit_or_copy_weights(rows: list, dest: Path, copy_from: Path | None, max_pairs: int) -> None:
    try:
        if len(rows) < 16:
            raise RuntimeError("too few rows")
        bank = ExperienceBank(experiences=rows)
        weights = fit_bank_weights(bank, max_pairs=max_pairs)
        save_weights(weights, dest)
        print(f"fitted weights → {dest}: {weights}")
        return
    except Exception as exc:
        print(f"weight fit failed ({exc})")
    if copy_from and copy_from.is_file():
        shutil.copy2(copy_from, dest)
        print(f"copied weights {copy_from} → {dest}")


def replay_gates_on_jsonl(src: Path, dest: Path, cfg: dict) -> list:
    rows = []
    with src.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            exp = json.loads(line)
            apply_gates(exp, cfg)
            rows.append(exp)
    _write_jsonl(dest, rows)
    return rows


def _build_one(job: dict):
    """Worker for ProcessPoolExecutor. Returns (status, payload)."""
    from mindware.components.experience.builder import build_experience_from_bo_json, load_trials_from_bo_json
    from mindware.components.experience.retrieve import attach_retrieval_index

    path = Path(job["path"])
    algo = job["algo"]
    higher = job["higher"]
    try:
        trials = load_trials_from_bo_json(
            path,
            algorithm=algo,
            higher_is_better=True if higher else False,
            openbox_minimize=True,
        )
        if len(trials) < int(job["min_trials"]):
            return ("skip", str(path))
        exp = build_experience_from_bo_json(
            path,
            optimizer="bo_history",
            dataset_id=job["dataset"],
            algorithm_id=algo,
            task_type=job["task_type"],
            metric=job["metric"],
            higher_is_better=higher,
            profile=job["profile"],
            fanova=True,
            gate_config=job["gate_cfg"],
            fanova_pair_topk=int(job["pair_topk"]),
        )
        attach_retrieval_index(exp, trials, profile_vec=job["profile"], max_trials=0)
        if "embed_text" in (exp.get("delivery") or {}):
            exp["delivery"].pop("embed_text", None)
        return ("ok", exp)
    except Exception as exc:
        return ("err", f"{path}: {type(exc).__name__}: {exc}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history-root", type=Path, default=ROOT / "history_data")
    p.add_argument("--output", type=Path, default=ROOT / "history_data" / "experience_bank_v3.jsonl")
    p.add_argument(
        "--holdout-output",
        type=Path,
        default=ROOT / "history_data" / "experience_bank_v3_holdout40.jsonl",
    )
    p.add_argument(
        "--tasks-json",
        type=Path,
        default=ROOT / "benchmark_data" / "real_hpo_v2_mltest20_algo40" / "tasks_40.json",
    )
    p.add_argument("--min-trials", type=int, default=8)
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--pair-topk", type=int, default=5)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_false", dest="resume")
    p.add_argument("--fit-weights", action="store_true", default=True)
    p.add_argument("--no-fit-weights", action="store_false", dest="fit_weights")
    p.add_argument("--max-pairs", type=int, default=400)
    p.add_argument("--gate-config", type=Path, default=None, help="JSON overrides; default v3 gates")
    p.add_argument(
        "--replay-only",
        action="store_true",
        help="Re-apply gates to an existing v3 jsonl (no fANOVA).",
    )
    args = p.parse_args(argv)

    import multiprocessing as mp

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    gate_cfg = dict(DEFAULT_GATE_CONFIG)
    if args.gate_config and args.gate_config.is_file():
        gate_cfg = merge_gate_config(json.loads(args.gate_config.read_text(encoding="utf-8")))

    if args.replay_only:
        if not args.output.is_file():
            raise SystemExit(f"missing {args.output}")
        rows = replay_gates_on_jsonl(args.output, args.output, gate_cfg)
        eval_ds = eval_holdout_datasets(args.tasks_json)
        hold = [
            e
            for e in rows
            if str((e.get("problem") or {}).get("dataset_id") or "").lower() not in eval_ds
        ]
        _write_jsonl(args.holdout_output, hold)
        print(f"replayed gates n={len(rows)} holdout={len(hold)}")
        return 0

    files = _iter_history_files(args.history_root)
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        raise SystemExit(f"No history JSON under {args.history_root}")

    done_by_uri: dict[str, dict] = {}
    if args.resume and args.output.is_file():
        with args.output.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                exp = json.loads(line)
                uri = str((exp.get("meta") or {}).get("run_uri") or "")
                fan = (exp.get("knowledge") or {}).get("fanova") or {}
                if uri and fan.get("importance_1d"):
                    done_by_uri[uri] = exp
        print(f"resume: {len(done_by_uri)} existing records in {args.output}")

    info = load_dataset_info_table(ROOT)
    jobs = []
    rows: list[dict] = []
    skipped_resume = 0
    for path in files:
        algo = path.stem
        dataset = path.parent.name
        metric = path.parent.parent.name
        tt_short = path.parent.parent.parent.name
        task_type = "classification" if tt_short == "cls" else "regression"
        higher = metric not in {"mse", "mae", "rmse"}
        uri = str(path)
        if uri in done_by_uri:
            rows.append(done_by_uri[uri])
            skipped_resume += 1
            continue
        profile = info.get((task_type, dataset)) or info.get((tt_short, dataset)) or {}
        jobs.append(
            {
                "path": str(path),
                "algo": algo,
                "dataset": dataset,
                "metric": metric,
                "task_type": task_type,
                "higher": higher,
                "profile": profile,
                "gate_cfg": gate_cfg,
                "pair_topk": int(args.pair_topk),
                "min_trials": int(args.min_trials),
            }
        )
    print(
        f"Building v3 bank from {len(files)} histories "
        f"({len(jobs)} to run, {skipped_resume} resumed) → {args.output} jobs={args.jobs}"
    )

    skipped = 0
    errors = 0
    n_done = 0
    workers = max(1, int(args.jobs))
    if workers == 1:
        futures_iter = ((_build_one(job), job) for job in jobs)
        for status_payload, _job in futures_iter:
            status, payload = status_payload
            n_done += 1
            if status == "ok":
                rows.append(payload)
            elif status == "skip":
                skipped += 1
            else:
                errors += 1
                print(f"[skip] {payload}")
            if n_done % 20 == 0:
                print(f"  {n_done}/{len(jobs)} new, {len(rows)} kept, {skipped} short, {errors} errors")
                _write_jsonl(args.output, rows)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            fmap = {pool.submit(_build_one, job): job for job in jobs}
            for fut in as_completed(fmap):
                status, payload = fut.result()
                n_done += 1
                if status == "ok":
                    rows.append(payload)
                elif status == "skip":
                    skipped += 1
                else:
                    errors += 1
                    print(f"[skip] {payload}")
                    if errors <= 5:
                        traceback.print_exc()
                if n_done % 20 == 0:
                    print(f"  {n_done}/{len(jobs)} new, {len(rows)} kept, {skipped} short, {errors} errors")
                    _write_jsonl(args.output, rows)

    _write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} experiences to {args.output}")

    eval_ds = eval_holdout_datasets(args.tasks_json)
    hold = [
        e
        for e in rows
        if str((e.get("problem") or {}).get("dataset_id") or "").lower() not in eval_ds
    ]
    _write_jsonl(args.holdout_output, hold)
    print(f"Wrote {len(hold)} holdout40 experiences to {args.holdout_output} (dropped {len(eval_ds)} eval datasets)")

    n_usable = sum(1 for e in rows if (e.get("delivery") or {}).get("usable_for_prompt"))
    engines: dict[str, int] = {}
    for e in rows:
        eng = str(((e.get("knowledge") or {}).get("fanova") or {}).get("engine") or "missing")
        engines[eng] = engines.get(eng, 0) + 1
    print(f"usable_for_prompt={n_usable}/{len(rows)} engines={engines}")

    if args.fit_weights:
        v2w = ROOT / "history_data" / "experience_bank_v2.weights.json"
        v2wh = ROOT / "history_data" / "experience_bank_v2_holdout40.weights.json"
        _fit_or_copy_weights(rows, args.output.with_suffix(".weights.json"), v2w if v2w.is_file() else None, args.max_pairs)
        _fit_or_copy_weights(
            hold,
            args.holdout_output.with_suffix(".weights.json"),
            v2wh if v2wh.is_file() else None,
            args.max_pairs,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
