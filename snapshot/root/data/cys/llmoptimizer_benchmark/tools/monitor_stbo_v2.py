#!/usr/bin/env python3
"""Anomaly monitor for the instrumented STBO rerun. Read-only; safe to poll."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

RUN = Path("/root/data/cys/blockdata_archive/domain_protein/stbo_init100_r100_bsz50_e2e10_v2_instrumented")
PREV = Path("/root/data/cys/blockdata_archive/domain_protein/stbo_init100_r100_bsz50_e2e10/index.jsonl")
PIDS = Path("/root/data/logs/stbo_init100_fleet/fleet.pids")
LOGS = Path("/root/data/logs/stbo_init100_fleet")
TARGET = 1000


def main() -> int:
    alerts: list[str] = []
    idx = RUN / "index.jsonl"
    rows = {}
    if idx.is_file():
        for line in idx.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                rows[r["task_id"]] = r
    n = len(rows)

    alive = 0
    if PIDS.is_file():
        pids = [p for p in PIDS.read_text().split() if p]
        for p in pids:
            if Path(f"/proc/{p}").exists():
                alive += 1
        if alive == 0 and n < TARGET:
            alerts.append(f"ALL WORKERS DEAD at {n}/{TARGET}")
        elif alive < len(pids) and n < TARGET:
            alerts.append(f"only {alive}/{len(pids)} workers alive")

    # previous run must be untouched
    prev_n = sum(1 for _ in PREV.open(encoding="utf-8")) if PREV.is_file() else 0
    if prev_n != 1001:
        alerts.append(f"PREVIOUS RUN CHANGED: {prev_n} rows (expected 1001)")

    # crashes
    bad = subprocess.run(
        ["grep", "-lE", "Traceback|CUDA out of memory|MemoryError"],
        input="", capture_output=True, text=True,
    )
    hits = [str(p) for p in LOGS.glob("shard*_gpu*.log")
            if subprocess.run(["grep", "-qE", "Traceback|CUDA out of memory|MemoryError", str(p)]).returncode == 0]
    if hits:
        alerts.append(f"tracebacks in {len(hits)} worker log(s): {[Path(h).name for h in hits][:4]}")

    # per-row integrity on completed tasks
    viol = trunc = nofeas = badbudget = 0
    for r in rows.values():
        if r.get("best_feasible_similarity") is not None and r["best_feasible_similarity"] < 0.75 - 1e-9:
            viol += 1
        if r.get("truncated"):
            trunc += 1
        if not r.get("n_init_feasible"):
            nofeas += 1
        if r.get("budget_respected") is False:
            badbudget += 1
    if viol:
        alerts.append(f"{viol} tasks violate similarity>=0.75")
    if nofeas:
        alerts.append(f"{nofeas} tasks with n_init_feasible falsy")
    if badbudget:
        alerts.append(f"{badbudget} tasks with budget_respected=false")

    # provenance integrity on the newest few histories
    checked = unmapped = 0
    batch_sizes: Counter = Counter()
    hs = sorted(RUN.glob("pep_*/bolt_native/seed1/history.json"), key=os.path.getmtime)[-3:]
    for h in hs:
        try:
            obs = json.loads(h.read_text(encoding="utf-8"))["observations"]
        except Exception as e:  # noqa: BLE001
            alerts.append(f"unreadable history {h.parent.parent.parent.name}: {type(e).__name__}")
            continue
        checked += 1
        unmapped += sum(1 for o in obs if o["extra_info"].get("phase") is None)
        b: Counter = Counter()
        for o in obs:
            x = o["extra_info"]
            if x.get("phase") == "acquisition":
                b[x["batch_id"]] += 1
        batch_sizes.update(b.values())
    if unmapped:
        alerts.append(f"{unmapped} observations with phase=None (provenance gap)")
    if checked and not batch_sizes:
        alerts.append("no acquisition batches found in recent histories")

    free_gb = os.statvfs(RUN).f_bavail * os.statvfs(RUN).f_frsize / 1024**3
    if free_gb < 80:
        alerts.append(f"disk low: {free_gb:.0f} GB free")

    done = n >= TARGET
    print(f"[{subprocess.run(['date','+%H:%M:%S'],capture_output=True,text=True).stdout.strip()}] "
          f"{n}/{TARGET} tasks | workers {alive} | trunc {trunc} | free {free_gb:.0f}GB")
    if checked:
        top = ", ".join(f"{k}x{v}" for k, v in batch_sizes.most_common(4))
        print(f"  provenance: {checked} histories, unmapped={unmapped}, acq batch sizes [{top}]")
    if alerts:
        print("  ANOMALIES:")
        for a in alerts:
            print(f"    - {a}")
    else:
        print("  no anomalies")
    print(f"  RUN_COMPLETE={done}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
