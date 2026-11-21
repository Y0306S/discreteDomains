#!/usr/bin/env python3
"""Patch p5plus_dynamic_gates_senior_report.html with Qwen s1b_attr_t2 results."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
P5 = REPO / "benchmark_data/real_hpo_v2_mltest20_algo40/analysis/p5plus"


def qrow_by_method(report: dict, part: str, method: str) -> dict | None:
    for row in report[f"part_{part.lower()}"]:
        if row["method"] == method:
            return row
    return None


def badge_html(w, l, t) -> str:
    if w == "—":
        return "—"
    w, l, t = int(w), int(l), int(t)
    if w > l:
        return '<span class="badge badge-yes">Beat</span>'
    if w == l:
        return '<span class="badge badge-tie">Tie</span>'
    return '<span class="badge badge-no">No</span>'


def fbal(x) -> str:
    return f"{float(x):.3f}" if x not in ("", None, "—") else "—"


def qbal(q: dict | None) -> str:
    return f"{q['mean_balanced_norm']:.3f}" if q else "—"


def main() -> None:
    report = json.loads((P5 / "qwen_dynamic_gates_report_data.json").read_text(encoding="utf-8"))
    html_path = P5 / "p5plus_dynamic_gates_senior_report.html"
    html = html_path.read_text(encoding="utf-8")

    gpt: dict[str, dict] = {}
    for row in csv.DictReader((P5 / "gpt_p5_dynamic_gates_vs_baseline.csv").open(encoding="utf-8")):
        gpt[row["method"]] = row

    base_bal = next(
        r["mean_balanced_norm"] for r in report["top15"] if r["variant"] == report["base_variant"]
    )

    part_a_methods = [
        ("H0", "50% / 50%"),
        ("HDUCB", "Adaptive (discounted UCB)"),
        ("HPUCB_arm", "Adaptive PUCB (arm-best)"),
        ("HPUCB_breakthrough", "Adaptive PUCB (breakthrough Q)"),
        ("HPUCB", "Adaptive PUCB"),
        ("HDTS", "Discounted Thompson"),
        ("HAP", "Adaptive pursuit"),
        ("HDMAB", "DMAB restart"),
    ]
    part_a_rows: list[str] = []
    for method, mix in part_a_methods:
        q = qrow_by_method(report, "a", method)
        g = gpt.get(method, {})
        hl = ' class="highlight"' if q and int(q["wins_vs_H2"]) > int(q["losses_vs_H2"]) else ""
        part_a_rows.append(
            f"        <tr{hl}><td><strong>{method}</strong></td><td>{mix}</td>"
            f'<td class="num">{g.get("wins_vs_H2", "—")}</td><td class="num">{g.get("losses_vs_H2", "—")}</td><td class="num">{g.get("ties", "—")}</td>'
            f'<td class="num">{q["wins_vs_H2"] if q else "—"}</td><td class="num">{q["losses_vs_H2"] if q else "—"}</td><td class="num">{q["ties_vs_H2"] if q else "—"}</td>'
            f'<td>{badge_html(q["wins_vs_H2"], q["losses_vs_H2"], q["ties_vs_H2"]) if q else "—"}</td>'
            f'<td class="num">{fbal(g.get("mean_balanced_norm_39arm"))}</td><td class="num">{qbal(q)}</td></tr>'
        )
    part_a_rows.append(
        '        <tr class="baseline"><td><strong>H2 (baseline)</strong></td><td>70% / 30%</td>'
        '<td class="num">—</td><td class="num">—</td><td class="num">—</td><td class="num">—</td><td class="num">—</td><td class="num">—</td>'
        '<td><span class="badge badge-part">Reference</span></td><td class="num">0.892</td>'
        f'<td class="num">{base_bal:.3f}</td></tr>'
    )

    part_b_methods = [
        ("TRdyStrength", "Readiness latch + confidence-weighted language"),
        ("TRdyFanova", "Readiness + fANOVA HP filter"),
        ("TRdy", "Readiness latch (replaces trial-50 clock)"),
        ("TRace", "Readiness + statistical racing on bins"),
        ("TRdyTurbo", "Readiness + TuRBO trust region"),
        ("TContrast", "Contrast-based insertion trigger"),
        ("TStag", "Insert on stagnation"),
    ]
    part_b_rows: list[str] = []
    for method, desc in part_b_methods:
        q = qrow_by_method(report, "b", method)
        g = gpt.get(method, {})
        hl = ' class="highlight"' if q and int(q["wins_vs_H2"]) > int(q["losses_vs_H2"]) else ""
        part_b_rows.append(
            f"        <tr{hl}><td>{method}</td><td>{desc}</td>"
            f'<td class="num">{g.get("wins_vs_H2", "—")}</td><td class="num">{g.get("losses_vs_H2", "—")}</td>'
            f'<td class="num">{q["wins_vs_H2"] if q else "—"}</td><td class="num">{q["losses_vs_H2"] if q else "—"}</td>'
            f'<td>{badge_html(q["wins_vs_H2"], q["losses_vs_H2"], q["ties_vs_H2"]) if q else "—"}</td>'
            f'<td class="num">{fbal(g.get("mean_balanced_norm_39arm"))}</td><td class="num">{qbal(q)}</td></tr>'
        )

    part_c_methods = [
        ("XStag", "p_explore rises when stagnating"),
        ("XCov", "p_explore rises when coverage low"),
        ("XRefine", "Hard switch by refine rules"),
        ("XHAP", "Adaptive pursuit on exploit vs explore arms"),
        ("XInt", "Exploit 3 LLM turns after improvement"),
    ]
    part_c_rows: list[str] = []
    for method, desc in part_c_methods:
        q = qrow_by_method(report, "c", method)
        g = gpt.get(method, {})
        part_c_rows.append(
            f"        <tr><td>{method}</td><td>{desc}</td>"
            f'<td class="num">{g.get("wins_vs_H2", "—")}</td><td class="num">{g.get("losses_vs_H2", "—")}</td>'
            f'<td class="num">{q["wins_vs_H2"] if q else "—"}</td><td class="num">{q["losses_vs_H2"] if q else "—"}</td>'
            f'<td>{badge_html(q["wins_vs_H2"], q["losses_vs_H2"], q["ties_vs_H2"]) if q else "—"}</td>'
            f'<td class="num">{fbal(g.get("mean_balanced_norm_39arm"))}</td><td class="num">{qbal(q)}</td></tr>'
        )

    lb: list[str] = []
    for r in report["top15"]:
        hl = (
            ' class="highlight"'
            if r["rank"] == 1
            else (' class="baseline"' if r["variant"] == report["base_variant"] else "")
        )
        display = (
            r["display"]
            .replace("P5plusA5T50X50", "P5+ A5 T50 X50")
            .replace("P5plusA5TRdyFanovaX50", "P5+ A5 TRdyFanova X50")
            .replace("P5plusA5TRdyStrengthX50", "P5+ A5 TRdyStrength X50")
            .replace("__", " / ")
        )
        lb.append(
            f'        <tr{hl}><td class="num">{r["rank"]}</td><td>{display}</td><td>{r["part_label"]}</td>'
            f'<td class="num">{r["avg_rank"]:.2f}</td><td class="num">{r["mean_balanced_norm"]:.3f}</td>'
            f'<td class="num">{r["avg_rank_cls"]:.2f}</td><td class="num">{r["avg_rank_rgs"]:.2f}</td></tr>'
        )

    q_a_beats = sum(1 for r in report["part_a"] if int(r["wins_vs_H2"]) > int(r["losses_vs_H2"]))
    q_b_beats = sum(1 for r in report["part_b"] if int(r["wins_vs_H2"]) > int(r["losses_vs_H2"]))
    arms = str(report["complete_arms"])

    html = html.replace(
        '<div class="meta">Generated: 18 Aug 2026 &nbsp;|&nbsp; Benchmark: <code>real_hpo_v2_mltest20_algo40</code> &nbsp;|&nbsp; 40 dataset×algorithm slots × 80 trials</div>',
        '<div class="meta">Generated: 22 Aug 2026 &nbsp;|&nbsp; Benchmark: <code>real_hpo_v2_mltest20_algo40</code> &nbsp;|&nbsp; 40 dataset×algorithm slots × 80 trials &nbsp;|&nbsp; GPT-4o-mini + Qwen s1b_attr_t2 LoRA</div>',
    )

    part_a_block = f"""  <h3>Part A — LLM call frequency <span class="badge badge-part">same X50 prompt</span></h3>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th rowspan="2">Method</th><th rowspan="2">LLM/TPE mix</th>
          <th colspan="3" class="num">GPT W/L/T</th><th colspan="3" class="num">Qwen W/L/T</th>
          <th rowspan="2">Qwen vs H2</th><th colspan="2" class="num">Bal. norm</th>
        </tr>
        <tr><th class="num">W</th><th class="num">L</th><th class="num">T</th><th class="num">W</th><th class="num">L</th><th class="num">T</th><th class="num">GPT</th><th class="num">Qwen</th></tr>
      </thead>
      <tbody>
{chr(10).join(part_a_rows)}
      </tbody>
    </table>
    <p><strong>Part A verdict:</strong> GPT: 2 beat, 1 tie, 9 lose. <strong>Qwen ({arms} arms, algo-correct 40/40):</strong> {q_a_beats} Part-A hybrids beat H2 on head-to-head wins. Best Qwen Part-A arm: <strong>HPUCB_arm</strong> (23–13–4). Fixed 50/50 (H0) is competitive (21–13–6) but not the top Qwen arm.</p>
  </div>"""

    part_b_block = f"""  <h3>Part B — Region insertion timing <span class="badge badge-part">H2 mix fixed</span></h3>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th rowspan="2">Method</th><th rowspan="2">What changes</th>
          <th colspan="2" class="num">GPT W/L</th><th colspan="2" class="num">Qwen W/L</th>
          <th rowspan="2">Qwen vs H2</th><th colspan="2" class="num">Bal. norm</th>
        </tr>
        <tr><th class="num">W</th><th class="num">L</th><th class="num">W</th><th class="num">L</th><th class="num">GPT</th><th class="num">Qwen</th></tr>
      </thead>
      <tbody>
{chr(10).join(part_b_rows)}
      </tbody>
    </table>
    <p><strong>Part B verdict:</strong> GPT: 0/7 beat baseline. <strong>Qwen:</strong> {q_b_beats}/7 beat H2 on head-to-head wins — notably <strong>TRdyFanova</strong> (22–16–2) and <strong>TRdyTurbo</strong> (21–18–1). Fanova is the strongest Part-B gate on Qwen despite failing on GPT (15–24–1).</p>
  </div>"""

    part_c_block = f"""  <h3>Part C — Explore / exploit <span class="badge badge-part">H2 mix fixed</span></h3>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th rowspan="2">Method</th><th rowspan="2">What changes</th>
          <th colspan="2" class="num">GPT W/L</th><th colspan="2" class="num">Qwen W/L</th>
          <th rowspan="2">Qwen vs H2</th><th colspan="2" class="num">Bal. norm</th>
        </tr>
        <tr><th class="num">W</th><th class="num">L</th><th class="num">W</th><th class="num">L</th><th class="num">GPT</th><th class="num">Qwen</th></tr>
      </thead>
      <tbody>
{chr(10).join(part_c_rows)}
      </tbody>
    </table>
    <p><strong>Part C verdict:</strong> GPT: 0/5 beat baseline. <strong>Qwen:</strong> 0/5 beat H2; adaptive X policies remain close but do not clearly beat fixed X50/H2.</p>
  </div>"""

    html = re.sub(
        r"  <h3>Part A — LLM call frequency.*?</div>\n\n  <h3>Part B",
        part_a_block + "\n\n  <h3>Part B",
        html,
        count=1,
        flags=re.S,
    )
    html = re.sub(
        r"  <h3>Part B — Region insertion timing.*?</div>\n\n  <h3>Part C",
        part_b_block + "\n\n  <h3>Part C",
        html,
        count=1,
        flags=re.S,
    )
    html = re.sub(
        r"  <h3>Part C — Explore / exploit.*?</div>\n\n  <h2>6\. Full leaderboard",
        part_c_block + "\n\n  <h2>6. Full leaderboard",
        html,
        count=1,
        flags=re.S,
    )

    qwen_section = f"""
  <h2>6b. Qwen s1b_attr_t2 leaderboard (top 15 of {arms} arms)</h2>
  <div class="card">
    <p>Model: <code>llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2</code> on <code>real_hpo_v2_mltest20_algo40_p5_s1b_attr_t2_p5plus</code>.
    Metrics: pooled trial normalization within the {arms}-arm Qwen P5+ peer pool (cls+rgs, drop worst 20%).
    All arms below: <strong>algo-correct 40/40</strong> task slots after the Aug 2026 repair pass.</p>
    <table>
      <thead>
        <tr>
          <th class="num">#</th><th>Method</th><th>Part</th>
          <th class="num">Avg rank</th><th class="num">Bal. norm</th><th class="num">Cls rank</th><th class="num">Rgs rank</th>
        </tr>
      </thead>
      <tbody>
{chr(10).join(lb)}
      </tbody>
    </table>
    <p>Full table: <code>analysis/p5plus/qwen_p5_p5plus_summary.csv</code> ({arms} rows, all 40/40).</p>
    <div class="callout callout-ok">
      <strong>Qwen vs GPT on dynamic gates:</strong> On Qwen, <strong>HPUCB_arm</strong> and <strong>TRdyFanova</strong> both beat the H2 baseline head-to-head — unlike GPT where only H0/HDUCB beat H2 and Fanova did not.
      The repaired Qwen sweep changes the Part B story: readiness+fANOVA is worth a confirmatory follow-up on Qwen even though it failed on GPT.
    </div>
  </div>
"""
    if "<h2>6b. Qwen s1b_attr_t2 leaderboard" not in html:
        html = html.replace(
            "<h2>7. Conclusions &amp; recommended next steps</h2>",
            qwen_section + "\n  <h2>7. Conclusions &amp; recommended next steps</h2>",
        )

    html = html.replace(
        '<div class="stat"><div class="num">2</div><div class="lbl">Arms beat baseline (H2)</div></div>',
        f'<div class="stat"><div class="num">2</div><div class="lbl">GPT arms beat H2</div></div>\n      <div class="stat"><div class="num">{q_b_beats}</div><div class="lbl">Qwen Part-B beat H2</div></div>',
    )

    if "qwen_p5_p5plus_summary.csv" not in html:
        html = html.replace(
            '<tr><td>39-arm leaderboard CSV</td><td><code>analysis/p5plus/gpt_p5_p5plus_summary.csv</code></td></tr>',
            '<tr><td>39-arm leaderboard CSV</td><td><code>analysis/p5plus/gpt_p5_p5plus_summary.csv</code></td></tr>\n        <tr><td>Qwen 37-arm leaderboard CSV</td><td><code>analysis/p5plus/qwen_p5_p5plus_summary.csv</code></td></tr>\n        <tr><td>Qwen head-to-head vs H2</td><td><code>analysis/p5plus/qwen_p5_dynamic_gates_vs_baseline.csv</code></td></tr>',
        )

    html_path.write_text(html, encoding="utf-8")
    print(f"Patched {html_path}")
    print(f"q_a_beats={q_a_beats} q_b_beats={q_b_beats}")


if __name__ == "__main__":
    main()
