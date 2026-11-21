#!/usr/bin/env python3
"""TPE (default) in BOLT VAE latent space: 80 trials × ~100 datasets.

Protein: UniRef z∈R^256, APEX mean-MIC oracle, one start sequence per task.
Query: CEB_64 z∈R^64, dummy EXPLAIN-cost until Postgres is up, one SQL per task.

Writes OpenBox JSON under history_data/domain_{protein,query}/ and traj CSVs
under benchmark_data/domain_{protein,query}/bo80_bolt/. Never touches the
tabular algo40 tree.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains.bolt_vae import PEPTIDE_VAE_CKPT, PeptideBoltVAE, QueryBoltVAE, ckpt_is_real
from mindware.domains.paths import BOLT_PEPTIDES, domain_bench, domain_history
from mindware.domains.queryopt.ceb import load_sql, load_task_ids
from mindware.domains.search import run_domain_search


def _done(hist: Path, n_trials: int) -> bool:
    if not hist.is_file():
        return False
    try:
        payload = json.loads(hist.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    obs = payload.get("observations") or []
    return len(obs) >= int(n_trials)


def _protein_seqs(limit: int, offset: int) -> List[str]:
    path = BOLT_PEPTIDES / "apex_oracle/init_data/seed_0_init.txt"
    seqs = [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith(">")
    ]
    return seqs[int(offset) : int(offset) + int(limit)]


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


def _cap_cpu_threads(n: int) -> None:
    n = max(1, int(n))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = str(n)
    try:
        import torch

        torch.set_num_threads(n)
        torch.set_num_interop_threads(min(n, 4))
    except Exception:
        pass


def _load_obs(path: Path) -> List[Dict[str, Any]]:
    from mindware.domains.history import load_openbox_history

    try:
        payload = load_openbox_history(path)
    except FileNotFoundError:
        return []
    return list(payload.get("observations") or [])


def _best_hist(*paths: Path) -> Path:
    best: Optional[Path] = None
    best_n = -1
    for p in paths:
        n = len(_load_obs(p))
        if n > best_n:
            best, best_n = p, n
    return best if best is not None else paths[0]


def _parse_pep_idx(task_id: str) -> int:
    return int(str(task_id).strip().split("_")[-1])


def run_protein(
    *,
    n_datasets: int,
    n_trials: int,
    offset: int,
    seed: int,
    device: str,
    skip_done: bool,
    optimizer: str = "tpe",
    task_ids: Optional[List[str]] = None,
    protein_oracle: str = "mean_mic",
    min_similarity: float = 0.75,
    constraint_index: int = 0,
    bench_subdir: str = "bo80_bolt",
    bolt_init_points: int = 1000,
    max_string_length: int = 30,
    bolt_strict_decode: bool = False,
    bolt_reject_dup_sequences: bool = False,
    e2e_freq: Optional[int] = None,
    allow_vanilla_tpe: bool = False,
    bolt_history_suffix: Optional[str] = None,
) -> int:
    from mindware.domains import get_adapter
    from mindware.domains.protein.apex_scorer import ApexMeanMIC
    from mindware.domains.protein.bolt_constraints import bolt_reference_sequence
    from mindware.domains.protein.bolt_scorer import BoltBacteria0Scorer
    from mindware.domains.protein.bolt_seed0_init import (
        BOLT_SEED0_HPARAMS,
        build_seed0_observations,
        total_trial_budget,
    )
    from mindware.domains.protein.bolt_peptide_e2e import (
        load_peptide_vae_checkpoint,
        peptide_vae_checkpoint_path,
    )
    from mindware.domains.protein.checkpoint_rotate import copy_two_deep

    vae_max_len = int(max_string_length) if protein_oracle == "bolt_native" else 50
    oracle_mode = str(protein_oracle or "mean_mic").lower()
    oracle_tag = "bolt_bacteria_0" if oracle_mode == "bolt_native" else "apex_mean_mic"
    bolt_ref = None
    if task_ids:
        all_seqs = _protein_seqs(10**6, 0)
        jobs = []
        for tid in task_ids:
            idx = _parse_pep_idx(tid)
            if idx < 0 or idx >= len(all_seqs):
                print(f"[protein] skip {tid} (idx out of range n={len(all_seqs)})")
                continue
            jobs.append((f"pep_{idx:03d}", idx, all_seqs[idx]))
    else:
        seqs = _protein_seqs(n_datasets, offset)
        jobs = [(f"pep_{offset + i:03d}", offset + i, seq) for i, seq in enumerate(seqs)]
    opt = str(optimizer or "tpe").lower()
    if oracle_mode == "bolt_native" and opt == "tpe" and not allow_vanilla_tpe:
        opt = "tr_tpe"
    if bolt_history_suffix:
        hist_suffix = str(bolt_history_suffix)
        if not hist_suffix.endswith(".json"):
            hist_suffix = f"{hist_suffix}.json"
    else:
        hist_suffix = f"{opt}_bolt_native.json" if oracle_mode == "bolt_native" else f"{opt}_bolt.json"
    index = domain_bench("protein") / bench_subdir / "index.jsonl"
    n_init = int(bolt_init_points) if oracle_mode == "bolt_native" else 0
    total_budget = (
        total_trial_budget(n_init_points=n_init, n_post_init_trials=int(n_trials))
        if oracle_mode == "bolt_native"
        else int(n_trials)
    )
    seed0_obs_cache: Optional[List[Dict[str, Any]]] = None
    n_ok = 0
    for task_id, idx, seq in jobs:
        task_constraint_index = int(idx)
        bolt_ref = (
            bolt_reference_sequence(constraint_index=task_constraint_index)
            if oracle_mode == "bolt_native"
            else None
        )
        apex = (
            BoltBacteria0Scorer(
                constraint_index=task_constraint_index,
                min_similarity=min_similarity,
            )
            if oracle_mode == "bolt_native"
            else ApexMeanMIC()
        )
        hist = domain_history("protein") / "fitness" / task_id / hist_suffix
        out_dir = domain_bench("protein") / bench_subdir / task_id / opt / f"seed{seed}"
        bench_hist = out_dir / "history.json"
        src_hist = _best_hist(hist, bench_hist)
        resume_obs = _load_obs(src_hist)
        if skip_done and len(resume_obs) >= int(total_budget):
            print(f"[protein] skip {task_id} (done n={len(resume_obs)}/{total_budget})")
            n_ok += 1
            continue
        t0 = time.time()
        vae = PeptideBoltVAE(PEPTIDE_VAE_CKPT, device=device, max_string_length=vae_max_len)
        vae_ckpt = peptide_vae_checkpoint_path(out_dir)
        if not vae_ckpt.is_file() and hist.parent.is_dir():
            alt = peptide_vae_checkpoint_path(hist.parent)
            if alt.is_file():
                vae_ckpt = alt
        if vae_ckpt.is_file():
            meta = load_peptide_vae_checkpoint(vae, vae_ckpt)
            if meta.get("vae_restored"):
                print(json.dumps({"task_id": task_id, "vae_resume": True, **meta}))
        try:
            init_z = vae.encode(seq)
        except Exception as exc:
            print(f"[protein] encode failed {task_id}: {exc}")
            continue
        adapter = get_adapter(
            "protein",
            task={"task_id": task_id, "start_seq": seq, "n_seqs": 1, "mean_length": float(len(seq))},
            vae=vae,
            apex=apex,
            dummy=False,
            max_len=vae_max_len,
            latent_bound=4.0,
            oracle_kind="bolt_native" if oracle_mode == "bolt_native" else "apex",
            bolt_reference=bolt_ref,
            bolt_anchor=seq,
            bolt_min_similarity=min_similarity,
            bolt_seed=int(seed) + int(idx),
            bolt_strict_decode=bool(bolt_strict_decode),
            bolt_reject_dup_sequences=bool(bolt_reject_dup_sequences),
        )
        merged_obs = list(resume_obs)
        if oracle_mode == "bolt_native" and not resume_obs:
            if seed0_obs_cache is None:
                seed0_obs_cache = build_seed0_observations(
                    vae,
                    limit=n_init,
                )
            seen = {((o.get("extra_info") or {}).get("artifact")) for o in merged_obs}
            for o in seed0_obs_cache:
                art = (o.get("extra_info") or {}).get("artifact")
                if art and art not in seen:
                    merged_obs.append(o)
                    seen.add(art)
        n_start = 1 if oracle_mode == "bolt_native" else min(10, max(4, 80 // 8))
        bolt_e2e_freq = (
            int(e2e_freq)
            if e2e_freq is not None
            else int(BOLT_SEED0_HPARAMS.get("e2e_freq") or 10)
        )
        result = run_domain_search(
            adapter,
            n_trials=total_budget,
            # Original 100-task batches used --seed 1, so pep_k is seed 1+(k%100).
            seed=int(seed) + (int(idx) % 100),
            optimizer=opt,
            output_dir=out_dir,
            init_z=init_z,
            n_startup_trials=n_start,
            resume_observations=merged_obs or None,
            e2e_freq=bolt_e2e_freq if oracle_mode == "bolt_native" else None,
            e2e_update_epochs=int(BOLT_SEED0_HPARAMS.get("num_update_epochs") or 2)
            if oracle_mode == "bolt_native"
            else None,
        )
        hist.parent.mkdir(parents=True, exist_ok=True)
        src = Path(result["history_path"])
        if src.resolve() != hist.resolve():
            hist.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        copy_two_deep(peptide_vae_checkpoint_path(out_dir), peptide_vae_checkpoint_path(hist.parent))
        n_obs = len(_load_obs(hist)) if hist.is_file() else len(_load_obs(src))
        row = {
            "domain": "protein",
            "task_id": task_id,
            "start_seq": seq,
            "n_init_points": n_init if oracle_mode == "bolt_native" else 0,
            "n_post_init_trials": int(n_trials) if oracle_mode == "bolt_native" else None,
            "n_trials": total_budget if oracle_mode == "bolt_native" else int(n_trials),
            "n_obs": n_obs,
            "resumed_from": result.get("resumed_from") or len(resume_obs),
            "best_objective": result.get("best_objective"),
            "seconds": round(time.time() - t0, 2),
            "history": str(hist),
            "vae": "bolt_uniref_256",
            "oracle": oracle_tag,
            "optimizer": opt,
            "min_similarity": min_similarity if oracle_mode == "bolt_native" else None,
            "constraint_index": task_constraint_index if oracle_mode == "bolt_native" else None,
            "bolt_reference": bolt_ref,
            "max_string_length": vae_max_len if oracle_mode == "bolt_native" else 50,
            "bolt_strict_decode": bool(bolt_strict_decode) if oracle_mode == "bolt_native" else None,
            "bolt_hparams": dict(BOLT_SEED0_HPARAMS) if oracle_mode == "bolt_native" else None,
        }
        _append_index(index, row)
        print(json.dumps(row))
        n_ok += 1
    return n_ok


def run_query(
    *,
    n_datasets: int,
    n_trials: int,
    offset: int,
    seed: int,
    device: str,
    skip_done: bool,
    dummy: bool,
    dsn: Optional[str],
    optimizer: str = "tpe",
) -> int:
    from mindware.domains import get_adapter

    vae = QueryBoltVAE(device=device)
    opt = str(optimizer or "tpe").lower()
    index = domain_bench("query") / "bo80_bolt" / "index.jsonl"
    n_ok = 0
    for i, tid in enumerate(load_task_ids(limit=n_datasets, offset=offset)):
        sql = load_sql(tid)
        if not sql:
            print(f"[query] skip {tid} (no SQL)")
            continue
        hist = domain_history("query") / "cost" / tid / f"{opt}_bolt.json"
        if skip_done and _done(hist, n_trials):
            print(f"[query] skip {tid} (done)")
            n_ok += 1
            continue
        t0 = time.time()
        adapter = get_adapter(
            "query",
            task={"task_id": tid, "sql": sql},
            vae=vae,
            dummy=dummy,
            dsn=dsn,
            latent_bound=4.0,
        )
        try:
            init_z = vae.encode([t for t, _ in adapter.pairs])
        except Exception as exc:
            print(f"[query] encode failed {tid}: {exc}")
            init_z = None
        out_dir = domain_bench("query") / "bo80_bolt" / tid / opt / f"seed{seed}"
        result = run_domain_search(
            adapter,
            n_trials=n_trials,
            seed=seed + i,
            optimizer=opt,
            output_dir=out_dir,
            init_z=init_z,
            n_startup_trials=min(10, max(4, n_trials // 8)),
        )
        hist.parent.mkdir(parents=True, exist_ok=True)
        src = Path(result["history_path"])
        if src.resolve() != hist.resolve():
            hist.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        row = {
            "domain": "query",
            "task_id": tid,
            "n_tables": len(adapter.aliases),
            "n_trials": n_trials,
            "best_objective": result.get("best_objective"),
            "seconds": round(time.time() - t0, 2),
            "history": str(hist),
            "vae": f"bolt_ceb_{vae.latent_dim}",
            "oracle": "dummy_cost" if dummy else "postgres_explain",
            "optimizer": opt,
        }
        _append_index(index, row)
        print(json.dumps(row))
        n_ok += 1
    return n_ok


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=["protein", "query", "both"], default="both")
    p.add_argument("--n-datasets", type=int, default=100)
    p.add_argument("--n-trials", type=int, default=80)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--skip-done", action="store_true", default=True)
    p.add_argument("--no-skip-done", action="store_false", dest="skip_done")
    p.add_argument("--query-dummy", action="store_true", default=None,
                    help="Force dummy cost. Default: dummy unless --query-dsn or QUERYOPT_PG_DSN is set.")
    p.add_argument("--query-dsn", default=None)
    p.add_argument("--optimizer", default="tpe", help="tpe | tr_tpe | cmaes | hebo | random")
    p.add_argument(
        "--task-ids",
        default="",
        help="Comma-separated protein task ids (e.g. pep_009,pep_791). Overrides --n-datasets/--offset.",
    )
    p.add_argument(
        "--cpu-threads",
        type=int,
        default=int(os.environ.get("PEPTIDE_BO_THREADS", "8")),
        help="Cap OpenMP/MKL/torch threads per process (for multi-GPU sharding).",
    )
    p.add_argument(
        "--protein-oracle",
        choices=["mean_mic", "bolt_native"],
        default="mean_mic",
        help="Protein oracle: mean_mic (11-pathogen) or bolt_native (bacteria_0 + 75%% sim).",
    )
    p.add_argument("--min-similarity", type=float, default=0.75)
    p.add_argument(
        "--constraint-index",
        type=int,
        default=0,
        help="BOLT REFERENCE_SEQUENCE index for similarity constraint (seed_0 uses 0).",
    )
    p.add_argument(
        "--bench-subdir",
        default=None,
        help="Benchmark output subdir under domain_protein (default: bo80_bolt or bo80_bolt_native).",
    )
    p.add_argument(
        "--bolt-init-points",
        type=int,
        default=1000,
        help="Pre-scored seed_0 init pool size for bolt_native (no oracle calls).",
    )
    p.add_argument(
        "--max-string-length",
        type=int,
        default=30,
        help="BOLT VAE max_string_length (seed_0 uses 30).",
    )
    p.add_argument(
        "--bolt-strict-decode",
        action="store_true",
        help="LOLBO-style decode: no propose_bolt_feasible_seq repair; penalize infeasible/duplicate seqs.",
    )
    p.add_argument(
        "--bolt-reject-dup-sequences",
        action="store_true",
        help="Repair decode then reject/resample sequences already in history (bolt_native TPE).",
    )
    p.add_argument(
        "--e2e-freq",
        type=int,
        default=None,
        help="VAE finetune every N stalls without incumbent improvement (bolt_native; default seed0=10).",
    )
    p.add_argument(
        "--allow-vanilla-tpe",
        action="store_true",
        help="Keep --optimizer tpe for bolt_native (do not auto-upgrade to tr_tpe).",
    )
    p.add_argument(
        "--bolt-history-suffix",
        default=None,
        help="Override history filename under fitness/<task_id>/ (e.g. tpe_vanilla_dedup_bolt_native.json).",
    )
    args = p.parse_args(argv)
    _cap_cpu_threads(args.cpu_threads)

    n = 0
    if args.domain in {"protein", "both"}:
        if not ckpt_is_real(PEPTIDE_VAE_CKPT):
            raise SystemExit(f"peptide VAE missing: {PEPTIDE_VAE_CKPT}")
        bench_subdir = args.bench_subdir or (
            "bo80_bolt_tr25k_strict"
            if args.protein_oracle == "bolt_native" and args.bolt_strict_decode
            else (
                "bo80_bolt_native_tr"
                if args.protein_oracle == "bolt_native"
                else "bo80_bolt"
            )
        )
        opt = args.optimizer
        if (
            args.protein_oracle == "bolt_native"
            and str(opt).lower() == "tpe"
            and not args.allow_vanilla_tpe
        ):
            opt = "tr_tpe"
        n += run_protein(
            n_datasets=args.n_datasets,
            n_trials=args.n_trials,
            offset=args.offset,
            seed=args.seed,
            device=args.device,
            skip_done=args.skip_done,
            optimizer=opt,
            task_ids=[t.strip() for t in str(args.task_ids).split(",") if t.strip()] or None,
            protein_oracle=args.protein_oracle,
            min_similarity=args.min_similarity,
            constraint_index=args.constraint_index,
            bench_subdir=bench_subdir,
            bolt_init_points=args.bolt_init_points,
            max_string_length=args.max_string_length,
            bolt_strict_decode=bool(args.bolt_strict_decode),
            bolt_reject_dup_sequences=bool(args.bolt_reject_dup_sequences),
            e2e_freq=args.e2e_freq,
            allow_vanilla_tpe=bool(args.allow_vanilla_tpe),
            bolt_history_suffix=args.bolt_history_suffix,
        )
    if args.domain in {"query", "both"}:
        n += run_query(
            n_datasets=args.n_datasets,
            n_trials=args.n_trials,
            offset=args.offset,
            seed=args.seed,
            device=args.device,
            skip_done=args.skip_done,
            dummy=(
                True
                if args.query_dummy is True
                else False
                if args.query_dummy is False
                else not bool(args.query_dsn or __import__("os").environ.get("QUERYOPT_PG_DSN"))
            ),
            dsn=args.query_dsn,
            optimizer=args.optimizer,
        )
    print(f"finished tasks={n} domain={args.domain} optimizer={args.optimizer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
