#!/usr/bin/env python3
"""Train-split BO interval analysis + Pareto selection of 32 RL tasks.

Pipeline:
  1. Load train JSONL (default: ml_train_80_fixed.jsonl)
  2. Filter to algo40-style algorithms with >= min successful BO trials
  3. Compute interval stats (0-ALL, 0-40, 40-80, 80-100) from frozen history_data
  4. Pareto front on (late_n_new, late_pct) where:
       late_n_new = n_new_bests(40-80) + n_new_bests(80-100)
       late_pct   = pct_improve(40-80) + pct_improve(80-100)
  5. Pick 16 cls + 16 rgs (fill to target with ranked non-front tasks if needed)
  6. Write CSV/JSON + HTML report
"""

from __future__ import annotations

import argparse
import csv
import html as htmlmod
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from examples.benchmark.analyze_bo_interval_improvements import (  # noqa: E402
    HIST_ROOT_DEFAULT,
    IntervalSpec,
    best_at_index,
    compute_pct_improvement,
    load_bo_perf_series,
    new_best_events,
    running_best,
)

DEFAULT_TRAIN = Path(
    "/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/ml_train_80_fixed.jsonl"
)
DEFAULT_OUT = REPO / "benchmark_data" / "rl_task_selection" / "bo_interval_improve_train80_n100"
ALGO40_ALGS = frozenset(
    {"catboost", "extra_trees", "gradient_boosting", "lightgbm", "xgboost"}
)
INTERVALS = [
    IntervalSpec("0-ALL", 0, 99),
    IntervalSpec("0-40", 0, 40),
    IntervalSpec("40-80", 40, 80),
    IntervalSpec("80-100", 80, 99),
]


@dataclass
class TaskRow:
    task_type: str
    dataset: str
    algorithm: str
    n_trials: int
  # interval -> stats
    by_interval: Dict[str, Dict[str, float]]

    @property
    def task_key(self) -> str:
        return f"{self.task_type}/{self.dataset}/{self.algorithm}"

    @property
    def late_n_new(self) -> int:
        return int(
            self.by_interval["40-80"]["n_new_bests"]
            + self.by_interval["80-100"]["n_new_bests"]
        )

    @property
    def late_pct(self) -> float:
        return float(
            self.by_interval["40-80"]["pct_improve"]
            + self.by_interval["80-100"]["pct_improve"]
        )

    @property
    def whole_pct(self) -> float:
        return float(self.by_interval["0-ALL"]["pct_improve"])

    @property
    def whole_n_new(self) -> int:
        return int(self.by_interval["0-ALL"]["n_new_bests"])


def load_train_tasks(path: Path) -> List[Dict[str, str]]:
    tasks: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            md = rec.get("metadata") or {}
            tt = md.get("task_type")
            ds = md.get("dataset")
            algo = md.get("algorithm")
            if not tt or not ds or not algo:
                continue
            tasks.append({"task_type": str(tt), "dataset": str(ds), "algorithm": str(algo)})
    return tasks


def filter_pool(
    tasks: Sequence[Dict[str, str]],
    *,
    hist_root: Path,
    n_trials: int,
    min_trials: int,
    algos: frozenset[str],
    min_abs_first_perf: float,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for t in tasks:
        if t["algorithm"] not in algos:
            continue
        tt, ds, algo = t["task_type"], t["dataset"], t["algorithm"]
        metric = "bal_acc" if tt == "cls" else "mse"
        hp = hist_root / tt / metric / ds / f"{algo}.json"
        if not hp.is_file():
            continue
        perfs = load_bo_perf_series(hp, n_trials=n_trials)
        if len(perfs) < min_trials:
            continue
        if abs(perfs[0]) < min_abs_first_perf:
            continue
        out.append(t)
    return out


def analyze_task(
    t: Dict[str, str],
    *,
    hist_root: Path,
    n_trials: int,
) -> Optional[TaskRow]:
    tt, ds, algo = t["task_type"], t["dataset"], t["algorithm"]
    metric = "bal_acc" if tt == "cls" else "mse"
    hp = hist_root / tt / metric / ds / f"{algo}.json"
    perfs = load_bo_perf_series(hp, n_trials=n_trials)
    if not perfs:
        return None
    bests = running_best(perfs)
    iters = list(range(len(bests)))
    events = new_best_events(iters, bests)

    seg_events: Dict[str, List[Tuple[int, float, float, float]]] = {
        seg.name: [] for seg in INTERVALS
    }
    for e_iter, before, after, delta in events:
        for seg in INTERVALS:
            if seg.start_idx <= e_iter <= seg.end_idx:
                seg_events[seg.name].append((e_iter, before, after, delta))

    by_interval: Dict[str, Dict[str, float]] = {}
    for seg in INTERVALS:
        b0 = best_at_index(bests, seg.start_idx)
        b1 = best_at_index(bests, seg.end_idx)
        if b0 is None or b1 is None:
            continue
        deltas = [float(e[3]) for e in seg_events[seg.name]]
        n_new = len(deltas)
        by_interval[seg.name] = {
            "start_best": float(b0),
            "end_best": float(b1),
            "pct_improve": compute_pct_improvement(float(b0), float(b1)),
            "n_new_bests": float(n_new),
            "total_delta": float(sum(deltas)) if deltas else 0.0,
            "mean_delta_new_best": float(sum(deltas) / n_new) if n_new else float("nan"),
            "median_delta_new_best": float(median(deltas)) if n_new else float("nan"),
        }

    if len(by_interval) != len(INTERVALS):
        return None

    return TaskRow(
        task_type=tt,
        dataset=ds,
        algorithm=algo,
        n_trials=len(perfs),
        by_interval=by_interval,
    )


def pareto_front(rows: Sequence[TaskRow], x_key: str, y_key: str) -> List[TaskRow]:
    """Non-dominated set for maximize (x, y)."""

    def xy(r: TaskRow) -> Tuple[float, float]:
        if x_key == "late_n_new":
            xv = float(r.late_n_new)
        elif x_key == "late_pct":
            xv = r.late_pct
        elif x_key == "whole_n_new":
            xv = float(r.whole_n_new)
        else:
            xv = r.whole_pct

        if y_key == "late_n_new":
            yv = float(r.late_n_new)
        elif y_key == "late_pct":
            yv = r.late_pct
        elif y_key == "whole_n_new":
            yv = float(r.whole_n_new)
        else:
            yv = r.whole_pct
        return xv, yv

    front: List[TaskRow] = []
    for i, p in enumerate(rows):
        px, py = xy(p)
        dominated = False
        for j, q in enumerate(rows):
            if i == j:
                continue
            qx, qy = xy(q)
            if qx >= px and qy >= py and (qx > px or qy > py):
                dominated = True
                break
        if not dominated:
            front.append(p)
    return front


def rank_score(rows: Sequence[TaskRow]) -> Dict[str, float]:
    """Higher is better: average percentile rank of late_n_new and late_pct."""
    if not rows:
        return {}
    n_new_vals = sorted((r.late_n_new for r in rows))
    pct_vals = sorted((r.late_pct for r in rows))

    def pct_rank(v: float, sorted_vals: List[float]) -> float:
        if not sorted_vals:
            return 0.0
        # fraction of values <= v
        le = sum(1 for x in sorted_vals if x <= v)
        return le / len(sorted_vals)

    out: Dict[str, float] = {}
    for r in rows:
        s = 0.5 * pct_rank(float(r.late_n_new), n_new_vals) + 0.5 * pct_rank(
            r.late_pct, pct_vals
        )
        out[r.task_key] = s
    return out


def select_balanced(
    pool: Sequence[TaskRow],
    *,
    target_per_type: int,
    pareto_x: str,
    pareto_y: str,
    unique_datasets: bool,
) -> Tuple[List[TaskRow], Dict[str, Any]]:
    meta: Dict[str, Any] = {"pareto_x": pareto_x, "pareto_y": pareto_y}
    selected: List[TaskRow] = []
    per_type: Dict[str, List[TaskRow]] = {"cls": [], "rgs": []}

    for tt in ("cls", "rgs"):
        sub = [r for r in pool if r.task_type == tt]
        front = pareto_front(sub, pareto_x, pareto_y)
        scores = rank_score(sub)
        front_set = {r.task_key for r in front}

        meta[f"{tt}_pool"] = len(sub)
        meta[f"{tt}_pareto_size"] = len(front)

        # Prefer pareto tasks, ranked by composite score; then fill from rest.
        ranked = sorted(sub, key=lambda r: (-scores[r.task_key], -r.late_n_new, -r.late_pct))
        pareto_ranked = [r for r in ranked if r.task_key in front_set]
        other_ranked = [r for r in ranked if r.task_key not in front_set]
        order = pareto_ranked + other_ranked

        picked: List[TaskRow] = []
        used_ds: set[str] = set()
        for r in order:
            if len(picked) >= target_per_type:
                break
            if unique_datasets and r.dataset in used_ds:
                continue
            picked.append(r)
            used_ds.add(r.dataset)

        # Relax unique-dataset constraint if needed.
        if len(picked) < target_per_type:
            for r in order:
                if len(picked) >= target_per_type:
                    break
                if r in picked:
                    continue
                picked.append(r)

        per_type[tt] = picked
        meta[f"{tt}_from_pareto"] = sum(1 for r in picked if r.task_key in front_set)

    selected = per_type["cls"] + per_type["rgs"]
    meta["selected_n"] = len(selected)
    meta["selected_cls"] = len(per_type["cls"])
    meta["selected_rgs"] = len(per_type["rgs"])
    return selected, meta


def write_summary_csv(rows: Sequence[TaskRow], path: Path, selected_keys: set[str]) -> None:
    fields = [
        "task_type",
        "dataset",
        "algorithm",
        "task_key",
        "interval",
        "pct_improve",
        "n_new_bests",
        "total_delta",
        "mean_delta_new_best",
        "median_delta_new_best",
        "late_pct_sum_40_80_80_100",
        "late_n_new_sum_40_80_80_100",
        "selected",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            for seg in INTERVALS:
                st = r.by_interval[seg.name]
                w.writerow(
                    {
                        "task_type": r.task_type,
                        "dataset": r.dataset,
                        "algorithm": r.algorithm,
                        "task_key": r.task_key,
                        "interval": seg.name,
                        "pct_improve": st["pct_improve"],
                        "n_new_bests": int(st["n_new_bests"]),
                        "total_delta": st["total_delta"],
                        "mean_delta_new_best": st["mean_delta_new_best"],
                        "median_delta_new_best": st["median_delta_new_best"],
                        "late_pct_sum_40_80_80_100": r.late_pct,
                        "late_n_new_sum_40_80_80_100": r.late_n_new,
                        "selected": int(r.task_key in selected_keys),
                    }
                )


def write_events_csv(rows: Sequence[TaskRow], hist_root: Path, n_trials: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "task_type",
                "dataset",
                "algorithm",
                "task_key",
                "interval",
                "new_best_iter",
                "best_before",
                "best_after",
                "delta",
            ]
        )
        for r in rows:
            tt, ds, algo = r.task_type, r.dataset, r.algorithm
            metric = "bal_acc" if tt == "cls" else "mse"
            hp = hist_root / tt / metric / ds / f"{algo}.json"
            perfs = load_bo_perf_series(hp, n_trials=n_trials)
            bests = running_best(perfs)
            events = new_best_events(list(range(len(bests))), bests)
            for e_iter, before, after, delta in events:
                for seg in INTERVALS:
                    if seg.start_idx <= e_iter <= seg.end_idx:
                        w.writerow([tt, ds, algo, r.task_key, seg.name, e_iter, before, after, delta])


def quantiles(vals: Sequence[float]) -> Dict[str, float]:
    if not vals:
        return {}
    s = sorted(vals)
    n = len(s)

    def q(p: float) -> float:
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return float(s[idx])

    return {"p25": q(0.25), "p50": q(0.50), "p75": q(0.75), "p90": q(0.90)}


TABLE_HEAD = """<thead><tr>
<th>task_type</th><th>dataset</th><th>algorithm</th><th>interval</th>
<th>pct_improve</th><th># new-bests</th><th>total_delta</th>
<th>mean delta</th><th>median delta</th>
<th>late pct (40-80+80-100)</th><th>late #new (40-80+80-100)</th>
<th>delta list</th><th>passes criteria</th>
</tr></thead>"""


def render_table_rows(
    df,
    ev_map: Dict[Tuple[str, str], List[float]],
    *,
    criteria_keys: set[str],
    row_attr: str,
    highlight: bool = False,
) -> str:
    import pandas as pd

    parts: List[str] = []
    for _, r in df.iterrows():
        interval = str(r["interval"])
        task_key = str(r["task_key"])
        passes = task_key in criteria_keys
        deltas = ev_map.get((task_key, interval), [])
        deltas_cell = ""
        if deltas:
            deltas_fmt = ", ".join(f"{float(x):.6g}" for x in deltas)
            deltas_cell = (
                f"<details><summary class='tag'>{len(deltas)} deltas</summary>"
                f"<code>{htmlmod.escape(deltas_fmt)}</code></details>"
            )
        mean_delta = (
            float(r["mean_delta_new_best"]) if pd.notna(r["mean_delta_new_best"]) else None
        )
        med_delta = (
            float(r["median_delta_new_best"]) if pd.notna(r["median_delta_new_best"]) else None
        )
        row_cls = "sel" if highlight else ""
        parts.append(
            f"<tr class='{row_cls}' data-interval='{htmlmod.escape(interval)}' "
            f"data-{row_attr}='1'>"
            f"<td>{htmlmod.escape(str(r['task_type']))}</td>"
            f"<td>{htmlmod.escape(str(r['dataset']))}</td>"
            f"<td>{htmlmod.escape(str(r['algorithm']))}</td>"
            f"<td>{htmlmod.escape(interval)}</td>"
            f"<td>{float(r['pct_improve']):.6g}</td>"
            f"<td>{int(r['n_new_bests'])}</td>"
            f"<td>{float(r['total_delta']):.6g}</td>"
            f"<td>{(f'{mean_delta:.6g}' if mean_delta is not None else '')}</td>"
            f"<td>{(f'{med_delta:.6g}' if med_delta is not None else '')}</td>"
            f"<td>{float(r['late_pct_sum_40_80_80_100']):.6g}</td>"
            f"<td>{int(r['late_n_new_sum_40_80_80_100'])}</td>"
            f"<td>{deltas_cell}</td>"
            f"<td>{'yes' if passes else ''}</td>"
            "</tr>"
        )
    return "".join(parts)


def generate_html(
    *,
    summary_csv: Path,
    events_csv: Path,
    out_html: Path,
    selected_keys: set[str],
    selected_order: List[str],
    criteria_keys: set[str],
    selection_meta: Dict[str, Any],
    metric_proposal: Dict[str, Any],
    criteria_desc: str,
) -> None:
    import pandas as pd

    sum_df = pd.read_csv(summary_csv)
    ev_df = pd.read_csv(events_csv)

    keep_intervals = [seg.name for seg in INTERVALS]
    sum_df = sum_df[sum_df["interval"].isin(keep_intervals)].copy()

    ev_map: Dict[Tuple[str, str], List[float]] = {}
    for (task_key, interval), g in ev_df.groupby(["task_key", "interval"], sort=False):
        ev_map[(str(task_key), str(interval))] = list(
            g.sort_values("new_best_iter")["delta"].astype(float)
        )

    def kpi(interval: str, frame) -> Tuple[float, float, float]:
        sub = frame[frame["interval"] == interval]
        if sub.empty:
            return float("nan"), float("nan"), float("nan")
        return (
            float(sub["pct_improve"].median()),
            float(sub["pct_improve"].mean()),
            float(sub["n_new_bests"].median()),
        )

    dropdown = ["ALL"] + keep_intervals
    options_html = "".join(
        f'<option value="{htmlmod.escape(x)}">{htmlmod.escape(x)}</option>' for x in dropdown
    )

    def kpi_card(label: str, frame) -> str:
        med, mean, med_n = kpi(label, frame)
        return (
            f"<div class='kpi'><b>{med:.4f}</b>"
            f"<span>{htmlmod.escape(label)} · pct median</span>"
            f"<div class='small'>mean pct={mean:.4g} · n_new median={med_n:.1f}</div></div>"
        )

    kpi_html = "".join(kpi_card(x, sum_df) for x in keep_intervals)

    sel32_df = sum_df[sum_df["task_key"].isin(selected_keys)].copy()
    order_map = {k: i for i, k in enumerate(selected_order)}
    sel32_df["_ord"] = sel32_df["task_key"].map(order_map)
    interval_ord = {name: i for i, name in enumerate(keep_intervals)}
    sel32_df["_iord"] = sel32_df["interval"].map(interval_ord)
    sel32_df = sel32_df.sort_values(["_ord", "_iord"]).drop(columns=["_ord", "_iord"])

    kpi32_html = "".join(kpi_card(x, sel32_df) for x in keep_intervals)

    proposal_lines = []
    for k, v in metric_proposal.items():
        proposal_lines.append(f"<li><b>{htmlmod.escape(str(k))}</b>: {htmlmod.escape(str(v))}</li>")
    proposal_html = "<ul>" + "".join(proposal_lines) + "</ul>"

    sel_lines = [
        f"<li>Criteria: <code>{htmlmod.escape(criteria_desc)}</code></li>",
        f"<li>Passed criteria: <b>{selection_meta.get('criteria_pass_n', 0)}</b> / "
        f"{selection_meta.get('pool_n', 0)} pool tasks</li>",
        f"<li>Selected best 32: <b>{selection_meta.get('selected_n', 0)}</b> "
        f"({selection_meta.get('selected_cls', 0)} cls + {selection_meta.get('selected_rgs', 0)} rgs)</li>",
        f"<li>Pareto axes: <code>{selection_meta.get('pareto_x')}</code> vs "
        f"<code>{selection_meta.get('pareto_y')}</code></li>",
        f"<li>From Pareto front: cls {selection_meta.get('cls_from_pareto', 0)}/16, "
        f"rgs {selection_meta.get('rgs_from_pareto', 0)}/16</li>",
        f"<li>Pareto pool sizes (among criteria-pass): cls {selection_meta.get('cls_pareto_size')}, "
        f"rgs {selection_meta.get('rgs_pareto_size')}</li>",
    ]
    sel_html = "<ul>" + "".join(sel_lines) + "</ul>"

    rows_html = render_table_rows(
        sum_df, ev_map, criteria_keys=criteria_keys, row_attr="pool", highlight=False
    )
    rows32_html = render_table_rows(
        sel32_df, ev_map, criteria_keys=criteria_keys, row_attr="pick32", highlight=True
    )

    html_txt = f"""<!doctype html>
<html><head><meta charset='utf-8'/>
<meta name='viewport' content='width=device-width, initial-scale=1'/>
<title>Train BO interval improvements + Pareto pick-32</title>
<style>
:root{{--bg:#f7f5f0;--ink:#1c1917;--muted:#78716c;--line:#d6d3d1;--card:#fffefb;--sel:#ecfdf5;}}
body{{margin:0;font-family:ui-serif,Georgia,serif;background:var(--bg);color:var(--ink);line-height:1.45;}}
.wrap{{max-width:1200px;margin:0 auto;padding:28px 20px 80px;}}
h1{{margin:0 0 8px;font-size:1.7rem;}}
.sub{{color:var(--muted);font-size:0.95rem;}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:14px 0;}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;}}
.kpi b{{display:block;font-size:1.2rem;}}
.kpi span{{color:var(--muted);font-size:0.9rem;}}
table{{width:100%;border-collapse:collapse;font-size:0.82rem;background:var(--card);}}
th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top;}}
th{{position:sticky;top:0;background:#f5f5f4;z-index:1;font-size:0.75rem;text-transform:uppercase;}}
tr.sel{{background:var(--sel);}}
code{{background:#f5f5f4;padding:1px 6px;border-radius:6px;white-space:pre;}}
.small{{color:var(--muted);font-size:0.9rem;}}
.tag{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:0.78rem;}}
select{{padding:6px 10px;border:1px solid var(--line);border-radius:10px;background:#fff;}}
.selectRow{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px;}}
.hidden{{display:none;}}
</style></head><body><div class='wrap'>
<h1>Train BO interval improvements (100 trials) + Pareto pick-32</h1>
<div class='sub'>Source: <code>{htmlmod.escape(str(summary_csv))}</code></div>

<div class='card'><h2>Metric proposal (for choosing 32)</h2>
{proposal_html}
</div>

<div class='card'><h2>Selection result</h2>
{sel_html}
</div>

<div class='card'>
<h2>Best 32 (pass criteria) — same rows as main table</h2>
<p class='small'>128 rows = 32 tasks × 4 intervals. Sorted in selection order (16 cls, then 16 rgs).</p>
<div class='selectRow'>
<label class='small' for='intervalSel32'>Filter interval:</label>
<select id='intervalSel32'>{options_html}</select>
<span class='small' id='rowCount32'></span>
</div>
<div class='grid'>{kpi32_html}</div>
<table id='tbl32'>
{TABLE_HEAD}
<tbody>
{rows32_html}
</tbody></table>
</div>

<div class='card'><h2>Full train pool</h2>
<div class='grid'>{kpi_html}</div></div>

<div class='card'>
<div class='selectRow'>
<label class='small' for='intervalSel'>Filter interval:</label>
<select id='intervalSel'>{options_html}</select>
<label class='small'><input type='checkbox' id='onlyCriteria'/> criteria-pass only</label>
<span class='small' id='rowCount'></span>
</div>
<table id='tblPool'>
{TABLE_HEAD}
<tbody>
{rows_html}
</tbody></table>
</div>

<script>
function bindIntervalFilter(selId, rowSelector, countId, extraFn) {{
  const sel=document.getElementById(selId);
  const rows=document.querySelectorAll(rowSelector);
  const rowCount=document.getElementById(countId);
  function apply(){{
    const v=sel.value;
    let n=0;
    rows.forEach(r=>{{
      const it=r.getAttribute('data-interval');
      const okExtra=extraFn?extraFn(r):true;
      const show=(v==='ALL'||it===v)&&okExtra;
      r.classList.toggle('hidden',!show);
      if(show) n++;
    }});
    rowCount.textContent='Showing '+n+' rows.';
  }}
  sel.addEventListener('change',apply);
  return apply;
}}
const apply32=bindIntervalFilter('intervalSel32','#tbl32 tr[data-pick32]','rowCount32',null);
const onlyCriteria=document.getElementById('onlyCriteria');
const applyPool=bindIntervalFilter('intervalSel','#tblPool tr[data-pool]','rowCount',function(r){{
  if(!onlyCriteria.checked) return true;
  const cells=r.querySelectorAll('td');
  return cells.length&&cells[cells.length-1].textContent==='yes';
}});
onlyCriteria.addEventListener('change',applyPool);
</script>
</div></body></html>"""

    out_html.write_text(html_txt, encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--hist-root", type=Path, default=HIST_ROOT_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--min-trials", type=int, default=100)
    p.add_argument("--target-n", type=int, default=32)
    p.add_argument("--target-per-type", type=int, default=16)
    p.add_argument("--pareto-x", default="late_n_new", choices=["late_n_new", "late_pct", "whole_n_new", "whole_pct"])
    p.add_argument("--pareto-y", default="late_pct", choices=["late_n_new", "late_pct", "whole_n_new", "whole_pct"])
    p.add_argument("--min-late-n-new", type=int, default=1, help="Criteria: min late #new-bests (40-80+80-100)")
    p.add_argument("--min-late-pct", type=float, default=0.0, help="Criteria: min late pct sum (40-80+80-100)")
    p.add_argument(
        "--max-late-pct",
        type=float,
        default=None,
        help="Criteria: max late pct sum. Excludes scale artifacts where a near-zero "
        "starting best inflates pct_improve (mse tasks). Default None = no cap "
        "(original behaviour).",
    )
    p.add_argument("--no-unique-datasets", action="store_true")
    args = p.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tasks = load_train_tasks(args.train_jsonl)
    pool_tasks = filter_pool(
        all_tasks,
        hist_root=args.hist_root,
        n_trials=args.n_trials,
        min_trials=args.min_trials,
        algos=ALGO40_ALGS,
        min_abs_first_perf=1e-6,
    )

    rows: List[TaskRow] = []
    for t in pool_tasks:
        r = analyze_task(t, hist_root=args.hist_root, n_trials=args.n_trials)
        if r is not None:
            rows.append(r)

    criteria_keys = {
        r.task_key
        for r in rows
        if r.late_n_new >= args.min_late_n_new
        and r.late_pct >= args.min_late_pct
        and (args.max_late_pct is None or r.late_pct <= args.max_late_pct)
    }
    criteria_rows = [r for r in rows if r.task_key in criteria_keys]

    criteria_desc = (
        f"late_n_new >= {args.min_late_n_new}"
        + (f" AND late_pct >= {args.min_late_pct:g}" if args.min_late_pct > 0 else "")
        + (f" AND late_pct <= {args.max_late_pct:g}" if args.max_late_pct is not None else "")
    )

    selected, sel_meta = select_balanced(
        criteria_rows,
        target_per_type=args.target_per_type,
        pareto_x=args.pareto_x,
        pareto_y=args.pareto_y,
        unique_datasets=not args.no_unique_datasets,
    )
    selected_keys = {r.task_key for r in selected}
    selected_order = [r.task_key for r in selected]
    sel_meta["pool_n"] = len(rows)
    sel_meta["criteria_pass_n"] = len(criteria_rows)
    sel_meta["criteria_desc"] = criteria_desc

    # Metric proposal based on pool quantiles.
    late_n = [float(r.late_n_new) for r in rows]
    late_pct = [r.late_pct for r in rows]
    whole_pct = [r.whole_pct for r in rows]
    q_n = quantiles(late_n)
    q_lp = quantiles(late_pct)
    q_wp = quantiles(whole_pct)

    metric_proposal = {
        "Selection criteria (applied)": criteria_desc,
        "Criteria-pass pool size": (
            f"{len(criteria_rows)} / {len(rows)} tasks pass before Pareto pick"
        ),
        "Primary Pareto axes (recommended)": (
            "late_n_new = n_new(40-80)+n_new(80-100) vs "
            "late_pct = pct(40-80)+pct(80-100); maximize both"
        ),
        "Why #new-bests matters": (
            f"median late #new-bests={q_n.get('p50', 0):.1f}; "
            "counts are stable when pct is tiny (common in cls bal_acc)"
        ),
        "Why late pct still matters": (
            f"median late pct sum={q_lp.get('p50', 0):.3f}%; "
            f"p75={q_lp.get('p75', 0):.3f}%; captures non-convergence after trial 40"
        ),
        "Suggested minimum gates (optional)": (
            f"late_n_new >= 1 (≈{(sum(1 for x in late_n if x>=1)/len(late_n)*100):.0f}% of pool); "
            f"or late_pct >= {q_lp.get('p75', 0):.2f}% for 'clearly moving late'"
        ),
        "Whole-run context": (
            f"median whole pct={q_wp.get('p50', 0):.2f}%; "
            "use whole-run only as tie-breaker, not primary hardness"
        ),
        "Fill rule to 32": (
            "16 cls + 16 rgs; take Pareto front first, then rank-fill by "
            "0.5·percentile(late_n_new)+0.5·percentile(late_pct); prefer unique datasets"
        ),
    }

    summary_csv = out_dir / "bo_interval_improve_summary.csv"
    events_csv = out_dir / "bo_interval_improve_events.csv"
    write_summary_csv(rows, summary_csv, selected_keys)
    write_events_csv(rows, args.hist_root, args.n_trials, events_csv)

    picked_json = out_dir / "picked_32_pareto.json"
    picked_csv = out_dir / "picked_32_pareto.csv"
    picked_payload = {
        "source": str(args.train_jsonl),
        "pool_n": len(rows),
        "selection": sel_meta,
        "metric_proposal": metric_proposal,
        "quantiles": {
            "late_n_new": q_n,
            "late_pct_sum": q_lp,
            "whole_pct": q_wp,
        },
        "tasks": [
            {
                "task_type": r.task_type,
                "dataset": r.dataset,
                "algorithm": r.algorithm,
                "metric": "bal_acc" if r.task_type == "cls" else "mse",
                "n_trials": r.n_trials,
                "late_n_new": r.late_n_new,
                "late_pct_sum": r.late_pct,
                "whole_pct": r.whole_pct,
                "whole_n_new": r.whole_n_new,
                "pct_40_80": r.by_interval["40-80"]["pct_improve"],
                "pct_80_100": r.by_interval["80-100"]["pct_improve"],
                "n_new_40_80": int(r.by_interval["40-80"]["n_new_bests"]),
                "n_new_80_100": int(r.by_interval["80-100"]["n_new_bests"]),
            }
            for r in selected
        ],
    }
    picked_json.write_text(json.dumps(picked_payload, indent=2) + "\n", encoding="utf-8")

    with picked_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task_type",
                "dataset",
                "algorithm",
                "late_n_new",
                "late_pct_sum",
                "pct_40_80",
                "pct_80_100",
                "n_new_40_80",
                "n_new_80_100",
                "whole_pct",
                "whole_n_new",
            ],
        )
        w.writeheader()
        for r in selected:
            w.writerow(
                {
                    "task_type": r.task_type,
                    "dataset": r.dataset,
                    "algorithm": r.algorithm,
                    "late_n_new": r.late_n_new,
                    "late_pct_sum": r.late_pct,
                    "pct_40_80": r.by_interval["40-80"]["pct_improve"],
                    "pct_80_100": r.by_interval["80-100"]["pct_improve"],
                    "n_new_40_80": int(r.by_interval["40-80"]["n_new_bests"]),
                    "n_new_80_100": int(r.by_interval["80-100"]["n_new_bests"]),
                    "whole_pct": r.whole_pct,
                    "whole_n_new": r.whole_n_new,
                }
            )

    html_path = out_dir / "bo_interval_improve_summary.html"
    generate_html(
        summary_csv=summary_csv,
        events_csv=events_csv,
        out_html=html_path,
        selected_keys=selected_keys,
        selected_order=selected_order,
        criteria_keys=criteria_keys,
        selection_meta=sel_meta,
        metric_proposal=metric_proposal,
        criteria_desc=criteria_desc,
    )

    print(f"pool tasks analyzed: {len(rows)}")
    print(f"criteria pass ({criteria_desc}): {len(criteria_rows)}")
    print(f"selected: {len(selected)} (cls={sel_meta['selected_cls']}, rgs={sel_meta['selected_rgs']})")
    print(f"pareto front sizes: cls={sel_meta['cls_pareto_size']}, rgs={sel_meta['rgs_pareto_size']}")
    print(f"from pareto: cls={sel_meta['cls_from_pareto']}, rgs={sel_meta['rgs_from_pareto']}")
    print(f"wrote {summary_csv}")
    print(f"wrote {events_csv}")
    print(f"wrote {picked_json}")
    print(f"wrote {picked_csv}")
    print(f"wrote {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
