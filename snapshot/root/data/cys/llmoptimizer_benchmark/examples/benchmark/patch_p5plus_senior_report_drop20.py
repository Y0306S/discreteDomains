#!/usr/bin/env python3
"""Refresh p5plus_dynamic_gates_senior_report.html bal-norm columns from drop-20% pooled table."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
P5 = REPO / "benchmark_data/real_hpo_v2_mltest20_algo40/analysis/p5plus"
DROP20 = (
    REPO
    / "benchmark_data/real_hpo_v2_mltest20_algo40/analysis/avg_rank/mean_norm_table_pooled_drop20pct_seed1.csv"
)
METRIC_NOTE = (
    "pooled trial normalization (cls+rgs, <strong>drop worst 20%</strong>), seed 1, "
    "from <code>mean_norm_table_pooled_drop20pct_seed1.csv</code>"
)
QWEN_DISK_PREFIX = "ptllm_tpe_qwen_s1b_attr_t2__"


def load_drop20() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in csv.DictReader(DROP20.open(encoding="utf-8")):
        out[row["variant"]] = {
            "bal": float(row["mean_balanced_norm"]),
            "avg_rank": float(row["avg_rank"]),
            "avg_rank_cls": float(row["avg_rank_cls"]),
            "avg_rank_rgs": float(row["avg_rank_rgs"]),
            "datasets": int(row["datasets"]),
        }
    return out


def fbal(x: float | None) -> str:
    return f"{x:.3f}" if x is not None else "—"


def gpt_variant(prompt: str, hybrid: str = "H2") -> str:
    return f"ptllm_tpe__{prompt}__{hybrid}" if "__" not in prompt else f"ptllm_tpe__{prompt}"


def qwen_disk_variant(report_variant: str) -> str:
    return report_variant.replace("ptllm_tpe_qwen__", QWEN_DISK_PREFIX, 1)


def badge_html(w, l, t) -> str:
    if w == "—":
        return "—"
    w, l, t = int(w), int(l), int(t)
    if w > l:
        return '<span class="badge badge-yes">Beat</span>'
    if w == l:
        return '<span class="badge badge-tie">Tie</span>'
    return '<span class="badge badge-no">No</span>'


def qrow_by_method(report: dict, part: str, method: str) -> dict | None:
    for row in report[f"part_{part.lower()}"]:
        if row["method"] == method:
            return row
    return None


def main() -> None:
    drop = load_drop20()
    report = json.loads((P5 / "qwen_dynamic_gates_report_data.json").read_text(encoding="utf-8"))
    html_path = P5 / "p5plus_dynamic_gates_senior_report.html"
    html = html_path.read_text(encoding="utf-8")

    gpt_h2h: dict[str, dict] = {}
    for row in csv.DictReader((P5 / "gpt_p5_dynamic_gates_vs_baseline.csv").open(encoding="utf-8")):
        gpt_h2h[row["method"]] = row

    def gpt_bal(method: str, prompt: str = "P5plusA5T50X50") -> str:
        v = gpt_variant(prompt, method) if method != prompt else gpt_variant(method)
        if method in drop:
            return fbal(drop[method]["bal"])
        row = drop.get(gpt_variant(prompt, method) if "__" not in method else gpt_variant(method))
        return fbal(row["bal"]) if row else "—"

    def qwen_bal(report_variant: str) -> str:
        row = drop.get(qwen_disk_variant(report_variant))
        return fbal(row["bal"]) if row else "—"

    # --- Part A ---
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
        g = gpt_h2h.get(method, {})
        hl = ' class="highlight"' if q and int(q["wins_vs_H2"]) > int(q["losses_vs_H2"]) else ""
        gbal = fbal(drop.get(gpt_variant("P5plusA5T50X50", method), {}).get("bal"))
        part_a_rows.append(
            f"        <tr{hl}><td><strong>{method}</strong></td><td>{mix}</td>"
            f'<td class="num">{g.get("wins_vs_H2", "—")}</td><td class="num">{g.get("losses_vs_H2", "—")}</td><td class="num">{g.get("ties", "—")}</td>'
            f'<td class="num">{q["wins_vs_H2"] if q else "—"}</td><td class="num">{q["losses_vs_H2"] if q else "—"}</td><td class="num">{q["ties_vs_H2"] if q else "—"}</td>'
            f'<td>{badge_html(q["wins_vs_H2"], q["losses_vs_H2"], q["ties_vs_H2"]) if q else "—"}</td>'
            f'<td class="num">{gbal}</td><td class="num">{qwen_bal(q["variant"]) if q else "—"}</td></tr>'
        )
    h2_gpt = drop.get(gpt_variant("P5plusA5T50X50", "H2"))
    h2_qwen = drop.get(qwen_disk_variant(report["base_variant"]))
    part_a_rows.append(
        '        <tr class="baseline"><td><strong>H2 (baseline)</strong></td><td>70% / 30%</td>'
        '<td class="num">—</td><td class="num">—</td><td class="num">—</td><td class="num">—</td><td class="num">—</td><td class="num">—</td>'
        '<td><span class="badge badge-part">Reference</span></td>'
        f'<td class="num">{fbal(h2_gpt["bal"] if h2_gpt else None)}</td>'
        f'<td class="num">{fbal(h2_qwen["bal"] if h2_qwen else None)}</td></tr>'
    )

    part_b_methods = [
        ("TRdyStrength", "Readiness latch + confidence-weighted language", "P5plusA5TRdyStrengthX50"),
        ("TRdyFanova", "Readiness + fANOVA HP filter", "P5plusA5TRdyFanovaX50"),
        ("TRdy", "Readiness latch (replaces trial-50 clock)", "P5plusA5TRdyX50"),
        ("TRace", "Readiness + statistical racing on bins", "P5plusA5TRaceX50"),
        ("TRdyTurbo", "Readiness + TuRBO trust region", "P5plusA5TRdyTurboX50"),
        ("TContrast", "Contrast-based insertion trigger", "P5plusA5TContrastX50"),
        ("TStag", "Insert on stagnation", "P5plusA5TStagX50"),
    ]
    part_b_rows: list[str] = []
    for method, desc, prompt in part_b_methods:
        q = qrow_by_method(report, "b", method)
        g = gpt_h2h.get(method, {})
        hl = ' class="highlight"' if q and int(q["wins_vs_H2"]) > int(q["losses_vs_H2"]) else ""
        gbal = fbal(drop.get(gpt_variant(prompt, "H2"), {}).get("bal"))
        part_b_rows.append(
            f"        <tr{hl}><td>{method}</td><td>{desc}</td>"
            f'<td class="num">{g.get("wins_vs_H2", "—")}</td><td class="num">{g.get("losses_vs_H2", "—")}</td>'
            f'<td class="num">{q["wins_vs_H2"] if q else "—"}</td><td class="num">{q["losses_vs_H2"] if q else "—"}</td>'
            f'<td>{badge_html(q["wins_vs_H2"], q["losses_vs_H2"], q["ties_vs_H2"]) if q else "—"}</td>'
            f'<td class="num">{gbal}</td><td class="num">{qwen_bal(q["variant"]) if q else "—"}</td></tr>'
        )

    part_c_methods = [
        ("XStag", "p_explore rises when stagnating", "P5plusA5T50XStag"),
        ("XCov", "p_explore rises when coverage low", "P5plusA5T50XCov"),
        ("XRefine", "Hard switch by refine rules", "P5plusA5T50XRefine"),
        ("XHAP", "Adaptive pursuit on exploit vs explore arms", "P5plusA5T50XHAP"),
        ("XInt", "Exploit 3 LLM turns after improvement", "P5plusA5T50XInt"),
    ]
    part_c_rows: list[str] = []
    for method, desc, prompt in part_c_methods:
        q = qrow_by_method(report, "c", method)
        g = gpt_h2h.get(method, {})
        gbal = fbal(drop.get(gpt_variant(prompt, "H2"), {}).get("bal"))
        part_c_rows.append(
            f"        <tr><td>{method}</td><td>{desc}</td>"
            f'<td class="num">{g.get("wins_vs_H2", "—")}</td><td class="num">{g.get("losses_vs_H2", "—")}</td>'
            f'<td class="num">{q["wins_vs_H2"] if q else "—"}</td><td class="num">{q["losses_vs_H2"] if q else "—"}</td>'
            f'<td>{badge_html(q["wins_vs_H2"], q["losses_vs_H2"], q["ties_vs_H2"]) if q else "—"}</td>'
            f'<td class="num">{gbal}</td><td class="num">{qwen_bal(q["variant"]) if q else "—"}</td></tr>'
        )

    q_a_beats = sum(1 for r in report["part_a"] if int(r["wins_vs_H2"]) > int(r["losses_vs_H2"]))
    q_b_beats = sum(1 for r in report["part_b"] if int(r["wins_vs_H2"]) > int(r["losses_vs_H2"]))
    arms = str(report["complete_arms"])

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
    <p><em>Bal. norm:</em> {METRIC_NOTE}.</p>
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
    <p><em>Bal. norm:</em> {METRIC_NOTE}.</p>
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
    <p><em>Bal. norm:</em> {METRIC_NOTE}.</p>
    <p><strong>Part C verdict:</strong> GPT: 0/5 beat baseline. <strong>Qwen:</strong> 0/5 beat H2; adaptive X policies remain close but do not clearly beat fixed X50/H2.</p>
  </div>"""

    # --- Part D ---
    d_vals = {
        "RC": drop.get("ptllm_tpe__P5plusA5T50X50RC__H2"),
        "RS": drop.get("ptllm_tpe__P5plusA5T50X50RS__H2"),
        "H2": h2_gpt,
        "H0": drop.get("ptllm_tpe__P5plusA5T50X50__H0"),
    }
    h0_bal = d_vals["H0"]["bal"] if d_vals["H0"] else 0
    rs_bal = d_vals["RS"]["bal"] if d_vals["RS"] else 0
    rc_bal = d_vals["RC"]["bal"] if d_vals["RC"] else 0
    h2_bal = d_vals["H2"]["bal"] if d_vals["H2"] else 0

    part_d_block = f"""  <h3>Part D — Similar-task retrieval <span class="badge badge-part">H2 mix fixed</span> <span class="badge badge-part">GPT only</span></h3>
  <div class="card">
    <p>Independent bake-off: <strong>RC</strong> (<code>P5plusA5T50X50RC</code>, cascade retrieve) vs <strong>RS</strong> (<code>P5plusA5T50X50RS</code>, per-algorithm GP surrogate landscape). Same X50/H2 hybrid as baseline; difference is cold-start experience from held-out similar tasks. Launcher: <code>run_p5plus_a5_t50_x50_retrieve_h2_eval.sh</code>.</p>
    <p><strong>Coverage note:</strong> GPT RC/RS are <em>not</em> full algo-correct 40/40 yet (RC 31/40, RS 30/40, H2 30/40 on disk). Head-to-head below uses only tasks where <em>both</em> arms have algo-correct trajectories. <strong>Qwen RC/RS have not been run</strong> (no <code>ptllm_tpe_qwen__P5plusA5T50X50RC/RS__H2</code> dirs).</p>
    <table>
      <thead>
        <tr>
          <th>Method</th><th>What changes</th><th class="num">Algo-correct tasks</th>
          <th class="num">vs H2 W–L–T</th><th class="num">overlap n</th>
          <th class="num">vs H0 W–L–T</th><th class="num">overlap n</th>
          <th class="num">Bal. norm</th>
          <th>vs H2</th>
        </tr>
      </thead>
      <tbody>
        <tr class="highlight">
          <td><strong>RC</strong></td><td>Cascade similar-task retrieve (hard filters + Spearman reweight)</td>
          <td class="num">31/40</td>
          <td class="num">11–8–2</td><td class="num">21</td>
          <td class="num">9–20–1</td><td class="num">30</td>
          <td class="num">{fbal(rc_bal)}</td>
          <td><span class="badge badge-yes">Beat H2</span></td>
        </tr>
        <tr class="highlight">
          <td><strong>RS</strong></td><td>Per-algo GP surrogate landscape retrieve</td>
          <td class="num">30/40</td>
          <td class="num">11–9–1</td><td class="num">21</td>
          <td class="num">10–19–1</td><td class="num">30</td>
          <td class="num">{fbal(rs_bal)}</td>
          <td><span class="badge badge-yes">Beat H2</span></td>
        </tr>
        <tr>
          <td>RC vs RS</td><td>Direct comparison (either arm may have solo coverage)</td>
          <td class="num">—</td>
          <td class="num">17–10–3</td><td class="num">30</td>
          <td class="num">—</td><td class="num">—</td>
          <td class="num">—</td>
          <td><span class="badge badge-yes">RC &gt; RS</span></td>
        </tr>
        <tr class="baseline">
          <td><strong>H2 (reference)</strong></td><td>No similar-task retrieve</td>
          <td class="num">30/40</td>
          <td class="num">—</td><td class="num">—</td>
          <td class="num">—</td><td class="num">—</td>
          <td class="num">{fbal(h2_bal)}</td>
          <td><span class="badge badge-part">Reference</span></td>
        </tr>
        <tr>
          <td><strong>H0</strong></td><td>50/50 mix (Part A best on GPT tasks)</td>
          <td class="num">30/40</td>
          <td class="num">12–8–1</td><td class="num">21</td>
          <td class="num">—</td><td class="num">—</td>
          <td class="num">{fbal(h0_bal)}</td>
          <td><span class="badge badge-yes">Beat H2</span></td>
        </tr>
      </tbody>
    </table>
    <p><em>Bal. norm:</em> {METRIC_NOTE} (GPT partial coverage 30–31 tasks for RC/RS/H2/H0).</p>
    <p><strong>Part D verdict:</strong> On the <strong>21-task overlap with H2</strong>, both RC and RS edge H2 on task wins (RC 11–8–2, RS 11–9–1), and <strong>RC beats RS</strong> 17–10–3 on 30 shared tasks. On bal norm (drop-20%), ordering is <strong>H0 ({fbal(h0_bal)}) &gt; RS ({fbal(rs_bal)}) &gt; H2 ({fbal(h2_bal)}) &gt; RC ({fbal(rc_bal)})</strong>. However, <strong>both retrieval arms lose badly to H0 on task wins</strong> (RC 9–20–1, RS 10–19–1) — similar-task retrieval does not beat the simpler 50/50 mix. Not recommended over H0; finish GPT 40/40 repair before drawing final conclusions. Qwen RC/RS eval still TODO.</p>
  </div>"""

    # --- GPT leaderboard (39-arm summary order, drop20 metrics) ---
    gpt_lb: list[str] = []
    gpt_rows = list(csv.DictReader((P5 / "gpt_p5_p5plus_summary.csv").open(encoding="utf-8")))
    ranked = []
    for r in gpt_rows:
        m = drop.get(r["variant"])
        if m:
            ranked.append((m["bal"], r, m))
    ranked.sort(key=lambda x: -x[0])
    for i, (bal, r, m) in enumerate(ranked[:15], 1):
        label = r["label"].replace("gpt ptllm_tpe ", "")
        part = "A" if "/H" in label and "TRdy" not in label and " X" not in label.split("/")[0][-3:] else (
            "B" if "TRdy" in label or "TContrast" in label or "TStag" in label else (
                "C" if " X" in label else "Prior ablation"
            )
        )
        if "A2" in label or "T50" in label and "X50" not in label:
            part = "Prior ablation"
        hl = ' class="highlight"' if i == 1 else (' class="baseline"' if "X50 / H2" in label and "RC" not in label and "RS" not in label else "")
        gpt_lb.append(
            f'        <tr{hl}><td class="num">{i}</td><td>{label}</td><td>{part}</td>'
            f'<td class="num">{m["avg_rank"]:.2f}</td><td class="num">{bal:.3f}</td>'
            f'<td class="num">{m["avg_rank_cls"]:.2f}</td><td class="num">{m["avg_rank_rgs"]:.2f}</td></tr>'
        )

    gpt_lb_block = f"""  <h2>6. Full leaderboard (top 15 of 39 arms)</h2>
  <div class="card">
    <p>Metrics: {METRIC_NOTE}; avg rank vs all methods in the global seed-1 table. Sorted by bal. norm (drop-20%). GPT coverage varies by arm (see <code>datasets</code> column in CSV).</p>
    <table>
      <thead>
        <tr>
          <th class="num">#</th><th>Method</th><th>Part</th>
          <th class="num">Avg rank</th><th class="num">Bal. norm</th><th class="num">Cls rank</th><th class="num">Rgs rank</th>
        </tr>
      </thead>
      <tbody>
{chr(10).join(gpt_lb)}
      </tbody>
    </table>
    <p>Full table: <code>gpt_p5_p5plus_summary.csv</code> (39 rows); bal. norm from <code>mean_norm_table_pooled_drop20pct_seed1.csv</code>.</p>
  </div>"""

    # --- Qwen leaderboard ---
    qwen_summary = list(csv.DictReader((P5 / "qwen_p5_p5plus_summary.csv").open(encoding="utf-8")))
    q_ranked = []
    for r in qwen_summary:
        v = qwen_disk_variant(r["variant"])
        m = drop.get(v)
        if m:
            q_ranked.append((m["bal"], r, m))
    q_ranked.sort(key=lambda x: -x[0])
    q_lb: list[str] = []
    base_v = report["base_variant"]
    for i, (bal, r, m) in enumerate(q_ranked[:15], 1):
        display = r["label"].split(" ", 3)[-1] if " " in r["label"] else r["label"]
        part = "A"
        if "TRdy" in r["variant"] or "TContrast" in r["variant"] or "TStag" in r["variant"]:
            part = "B"
        elif "T50X" in r["variant"] and "X50" not in r["variant"]:
            part = "C"
        elif "P5plusA2" in r["variant"] or "P5plusT50" in r["variant"] or "P5plusA5T40" in r["variant"]:
            part = "Prior"
        hl = ' class="highlight"' if i == 1 else (' class="baseline"' if r["variant"] == base_v else "")
        q_lb.append(
            f'        <tr{hl}><td class="num">{i}</td><td>{display}</td><td>{part}</td>'
            f'<td class="num">{m["avg_rank"]:.2f}</td><td class="num">{bal:.3f}</td>'
            f'<td class="num">{m["avg_rank_cls"]:.2f}</td><td class="num">{m["avg_rank_rgs"]:.2f}</td></tr>'
        )

    qwen_lb_block = f"""  <h2>6b. Qwen s1b_attr_t2 leaderboard (top 15 of 37 arms)</h2>
  <div class="card">
    <p>Model: <code>llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2</code> on <code>real_hpo_v2_mltest20_algo40_p5_s1b_attr_t2_p5plus</code>.
    Metrics: {METRIC_NOTE}. Sorted by bal. norm (drop-20%).
    All arms below: <strong>algo-correct 40/40</strong> task slots after the Aug 2026 repair pass.</p>
    <table>
      <thead>
        <tr>
          <th class="num">#</th><th>Method</th><th>Part</th>
          <th class="num">Avg rank</th><th class="num">Bal. norm</th><th class="num">Cls rank</th><th class="num">Rgs rank</th>
        </tr>
      </thead>
      <tbody>
{chr(10).join(q_lb)}
      </tbody>
    </table>
    <p>Full table: <code>analysis/p5plus/qwen_p5_p5plus_summary.csv</code> (37 rows, all 40/40); bal. norm from <code>mean_norm_table_pooled_drop20pct_seed1.csv</code>.</p>
    <div class="callout callout-ok">
      <strong>Qwen vs GPT on dynamic gates:</strong> On Qwen, <strong>HPUCB_arm</strong> and <strong>TRdyFanova</strong> both beat the H2 baseline head-to-head — unlike GPT where only H0/HDUCB beat H2 and Fanova did not.
      The repaired Qwen sweep changes the Part B story: readiness+fANOVA is worth a confirmatory follow-up on Qwen even though it failed on GPT.
    </div>
  </div>"""

    # Replace sections via regex
    html = re.sub(
        r"  <h3>Part A — LLM call frequency.*?</div>\n\n  <h3>Part B",
        part_a_block + "\n\n  <h3>Part B",
        html,
        count=1,
        flags=re.DOTALL,
    )
    html = re.sub(
        r"  <h3>Part B — Region insertion timing.*?</div>\n\n  <h3>Part C",
        part_b_block + "\n\n  <h3>Part C",
        html,
        count=1,
        flags=re.DOTALL,
    )
    html = re.sub(
        r"  <h3>Part C — Explore / exploit.*?</div>\n\n  <h3>Part D",
        part_c_block + "\n\n  <h3>Part D",
        html,
        count=1,
        flags=re.DOTALL,
    )
    html = re.sub(
        r"  <h3>Part D — Similar-task retrieval.*?</div>\n\n  <h2>6\. Full leaderboard",
        part_d_block + "\n\n  <h2>6. Full leaderboard",
        html,
        count=1,
        flags=re.DOTALL,
    )
    html = re.sub(
        r"  <h2>6\. Full leaderboard \(top 15 of 39 arms\)</h2>.*?</div>\n\n  \n  <h2>6b\. Qwen",
        gpt_lb_block + "\n\n  \n  <h2>6b. Qwen",
        html,
        count=1,
        flags=re.DOTALL,
    )
    html = re.sub(
        r"  <h2>6b\. Qwen s1b_attr_t2 leaderboard \(top 15 of 37 arms\)</h2>.*?</div>\n\n  <h2>7\. Conclusions",
        qwen_lb_block + "\n\n  <h2>7. Conclusions",
        html,
        count=1,
        flags=re.DOTALL,
    )

    html = html.replace("drop worst 10%", "drop worst 20%")
    html_path.write_text(html, encoding="utf-8")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Refresh p5plus_dynamic_gates_senior_report.html bal-norm columns from drop-20% pooled table."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
P5 = REPO / "benchmark_data/real_hpo_v2_mltest20_algo40/analysis/p5plus"
DROP20 = (
    REPO
    / "benchmark_data/real_hpo_v2_mltest20_algo40/analysis/avg_rank/mean_norm_table_pooled_drop20pct_seed1.csv"
)
METRIC_NOTE = (
    "pooled trial normalization (cls+rgs, <strong>drop worst 20%</strong>), seed 1, "
    "from <code>mean_norm_table_pooled_drop20pct_seed1.csv</code>"
)
QWEN_DISK_PREFIX = "ptllm_tpe_qwen_s1b_attr_t2__"


def load_drop20() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in csv.DictReader(DROP20.open(encoding="utf-8")):
        out[row["variant"]] = {
            "bal": float(row["mean_balanced_norm"]),
            "avg_rank": float(row["avg_rank"]),
            "avg_rank_cls": float(row["avg_rank_cls"]),
            "avg_rank_rgs": float(row["avg_rank_rgs"]),
            "datasets": int(row["datasets"]),
        }
    return out


def fbal(x: float | None) -> str:
    return f"{x:.3f}" if x is not None else "—"


def gpt_variant(prompt: str, hybrid: str = "H2") -> str:
    return f"ptllm_tpe__{prompt}__{hybrid}" if "__" not in prompt else f"ptllm_tpe__{prompt}"


def qwen_disk_variant(report_variant: str) -> str:
    return report_variant.replace("ptllm_tpe_qwen__", QWEN_DISK_PREFIX, 1)


def badge_html(w, l, t) -> str:
    if w == "—":
        return "—"
    w, l, t = int(w), int(l), int(t)
    if w > l:
        return '<span class="badge badge-yes">Beat</span>'
    if w == l:
        return '<span class="badge badge-tie">Tie</span>'
    return '<span class="badge badge-no">No</span>'


def qrow_by_method(report: dict, part: str, method: str) -> dict | None:
    for row in report[f"part_{part.lower()}"]:
        if row["method"] == method:
            return row
    return None


def main() -> None:
    drop = load_drop20()
    report = json.loads((P5 / "qwen_dynamic_gates_report_data.json").read_text(encoding="utf-8"))
    html_path = P5 / "p5plus_dynamic_gates_senior_report.html"
    html = html_path.read_text(encoding="utf-8")

    gpt_h2h: dict[str, dict] = {}
    for row in csv.DictReader((P5 / "gpt_p5_dynamic_gates_vs_baseline.csv").open(encoding="utf-8")):
        gpt_h2h[row["method"]] = row

    def gpt_bal(method: str, prompt: str = "P5plusA5T50X50") -> str:
        v = gpt_variant(prompt, method) if method != prompt else gpt_variant(method)
        if method in drop:
            return fbal(drop[method]["bal"])
        row = drop.get(gpt_variant(prompt, method) if "__" not in method else gpt_variant(method))
        return fbal(row["bal"]) if row else "—"

    def qwen_bal(report_variant: str) -> str:
        row = drop.get(qwen_disk_variant(report_variant))
        return fbal(row["bal"]) if row else "—"

    # --- Part A ---
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
        g = gpt_h2h.get(method, {})
        hl = ' class="highlight"' if q and int(q["wins_vs_H2"]) > int(q["losses_vs_H2"]) else ""
        gbal = fbal(drop.get(gpt_variant("P5plusA5T50X50", method), {}).get("bal"))
        part_a_rows.append(
            f"        <tr{hl}><td><strong>{method}</strong></td><td>{mix}</td>"
            f'<td class="num">{g.get("wins_vs_H2", "—")}</td><td class="num">{g.get("losses_vs_H2", "—")}</td><td class="num">{g.get("ties", "—")}</td>'
            f'<td class="num">{q["wins_vs_H2"] if q else "—"}</td><td class="num">{q["losses_vs_H2"] if q else "—"}</td><td class="num">{q["ties_vs_H2"] if q else "—"}</td>'
            f'<td>{badge_html(q["wins_vs_H2"], q["losses_vs_H2"], q["ties_vs_H2"]) if q else "—"}</td>'
            f'<td class="num">{gbal}</td><td class="num">{qwen_bal(q["variant"]) if q else "—"}</td></tr>'
        )
    h2_gpt = drop.get(gpt_variant("P5plusA5T50X50", "H2"))
    h2_qwen = drop.get(qwen_disk_variant(report["base_variant"]))
    part_a_rows.append(
        '        <tr class="baseline"><td><strong>H2 (baseline)</strong></td><td>70% / 30%</td>'
        '<td class="num">—</td><td class="num">—</td><td class="num">—</td><td class="num">—</td><td class="num">—</td><td class="num">—</td>'
        '<td><span class="badge badge-part">Reference</span></td>'
        f'<td class="num">{fbal(h2_gpt["bal"] if h2_gpt else None)}</td>'
        f'<td class="num">{fbal(h2_qwen["bal"] if h2_qwen else None)}</td></tr>'
    )

    part_b_methods = [
        ("TRdyStrength", "Readiness latch + confidence-weighted language", "P5plusA5TRdyStrengthX50"),
        ("TRdyFanova", "Readiness + fANOVA HP filter", "P5plusA5TRdyFanovaX50"),
        ("TRdy", "Readiness latch (replaces trial-50 clock)", "P5plusA5TRdyX50"),
        ("TRace", "Readiness + statistical racing on bins", "P5plusA5TRaceX50"),
        ("TRdyTurbo", "Readiness + TuRBO trust region", "P5plusA5TRdyTurboX50"),
        ("TContrast", "Contrast-based insertion trigger", "P5plusA5TContrastX50"),
        ("TStag", "Insert on stagnation", "P5plusA5TStagX50"),
    ]
    part_b_rows: list[str] = []
    for method, desc, prompt in part_b_methods:
        q = qrow_by_method(report, "b", method)
        g = gpt_h2h.get(method, {})
        hl = ' class="highlight"' if q and int(q["wins_vs_H2"]) > int(q["losses_vs_H2"]) else ""
        gbal = fbal(drop.get(gpt_variant(prompt, "H2"), {}).get("bal"))
        part_b_rows.append(
            f"        <tr{hl}><td>{method}</td><td>{desc}</td>"
            f'<td class="num">{g.get("wins_vs_H2", "—")}</td><td class="num">{g.get("losses_vs_H2", "—")}</td>'
            f'<td class="num">{q["wins_vs_H2"] if q else "—"}</td><td class="num">{q["losses_vs_H2"] if q else "—"}</td>'
            f'<td>{badge_html(q["wins_vs_H2"], q["losses_vs_H2"], q["ties_vs_H2"]) if q else "—"}</td>'
            f'<td class="num">{gbal}</td><td class="num">{qwen_bal(q["variant"]) if q else "—"}</td></tr>'
        )

    part_c_methods = [
        ("XStag", "p_explore rises when stagnating", "P5plusA5T50XStag"),
        ("XCov", "p_explore rises when coverage low", "P5plusA5T50XCov"),
        ("XRefine", "Hard switch by refine rules", "P5plusA5T50XRefine"),
        ("XHAP", "Adaptive pursuit on exploit vs explore arms", "P5plusA5T50XHAP"),
        ("XInt", "Exploit 3 LLM turns after improvement", "P5plusA5T50XInt"),
    ]
    part_c_rows: list[str] = []
    for method, desc, prompt in part_c_methods:
        q = qrow_by_method(report, "c", method)
        g = gpt_h2h.get(method, {})
        gbal = fbal(drop.get(gpt_variant(prompt, "H2"), {}).get("bal"))
        part_c_rows.append(
            f"        <tr><td>{method}</td><td>{desc}</td>"
            f'<td class="num">{g.get("wins_vs_H2", "—")}</td><td class="num">{g.get("losses_vs_H2", "—")}</td>'
            f'<td class="num">{q["wins_vs_H2"] if q else "—"}</td><td class="num">{q["losses_vs_H2"] if q else "—"}</td>'
            f'<td>{badge_html(q["wins_vs_H2"], q["losses_vs_H2"], q["ties_vs_H2"]) if q else "—"}</td>'
            f'<td class="num">{gbal}</td><td class="num">{qwen_bal(q["variant"]) if q else "—"}</td></tr>'
        )

    q_a_beats = sum(1 for r in report["part_a"] if int(r["wins_vs_H2"]) > int(r["losses_vs_H2"]))
    q_b_beats = sum(1 for r in report["part_b"] if int(r["wins_vs_H2"]) > int(r["losses_vs_H2"]))
    arms = str(report["complete_arms"])

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
    <p><em>Bal. norm:</em> {METRIC_NOTE}.</p>
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
    <p><em>Bal. norm:</em> {METRIC_NOTE}.</p>
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
    <p><em>Bal. norm:</em> {METRIC_NOTE}.</p>
    <p><strong>Part C verdict:</strong> GPT: 0/5 beat baseline. <strong>Qwen:</strong> 0/5 beat H2; adaptive X policies remain close but do not clearly beat fixed X50/H2.</p>
  </div>"""

    # --- Part D ---
    d_vals = {
        "RC": drop.get("ptllm_tpe__P5plusA5T50X50RC__H2"),
        "RS": drop.get("ptllm_tpe__P5plusA5T50X50RS__H2"),
        "H2": h2_gpt,
        "H0": drop.get("ptllm_tpe__P5plusA5T50X50__H0"),
    }
    h0_bal = d_vals["H0"]["bal"] if d_vals["H0"] else 0
    rs_bal = d_vals["RS"]["bal"] if d_vals["RS"] else 0
    rc_bal = d_vals["RC"]["bal"] if d_vals["RC"] else 0
    h2_bal = d_vals["H2"]["bal"] if d_vals["H2"] else 0

    part_d_block = f"""  <h3>Part D — Similar-task retrieval <span class="badge badge-part">H2 mix fixed</span> <span class="badge badge-part">GPT only</span></h3>
  <div class="card">
    <p>Independent bake-off: <strong>RC</strong> (<code>P5plusA5T50X50RC</code>, cascade retrieve) vs <strong>RS</strong> (<code>P5plusA5T50X50RS</code>, per-algorithm GP surrogate landscape). Same X50/H2 hybrid as baseline; difference is cold-start experience from held-out similar tasks. Launcher: <code>run_p5plus_a5_t50_x50_retrieve_h2_eval.sh</code>.</p>
    <p><strong>Coverage note:</strong> GPT RC/RS are <em>not</em> full algo-correct 40/40 yet (RC 31/40, RS 30/40, H2 30/40 on disk). Head-to-head below uses only tasks where <em>both</em> arms have algo-correct trajectories. <strong>Qwen RC/RS have not been run</strong> (no <code>ptllm_tpe_qwen__P5plusA5T50X50RC/RS__H2</code> dirs).</p>
    <table>
      <thead>
        <tr>
          <th>Method</th><th>What changes</th><th class="num">Algo-correct tasks</th>
          <th class="num">vs H2 W–L–T</th><th class="num">overlap n</th>
          <th class="num">vs H0 W–L–T</th><th class="num">overlap n</th>
          <th class="num">Bal. norm</th>
          <th>vs H2</th>
        </tr>
      </thead>
      <tbody>
        <tr class="highlight">
          <td><strong>RC</strong></td><td>Cascade similar-task retrieve (hard filters + Spearman reweight)</td>
          <td class="num">31/40</td>
          <td class="num">11–8–2</td><td class="num">21</td>
          <td class="num">9–20–1</td><td class="num">30</td>
          <td class="num">{fbal(rc_bal)}</td>
          <td><span class="badge badge-yes">Beat H2</span></td>
        </tr>
        <tr class="highlight">
          <td><strong>RS</strong></td><td>Per-algo GP surrogate landscape retrieve</td>
          <td class="num">30/40</td>
          <td class="num">11–9–1</td><td class="num">21</td>
          <td class="num">10–19–1</td><td class="num">30</td>
          <td class="num">{fbal(rs_bal)}</td>
          <td><span class="badge badge-yes">Beat H2</span></td>
        </tr>
        <tr>
          <td>RC vs RS</td><td>Direct comparison (either arm may have solo coverage)</td>
          <td class="num">—</td>
          <td class="num">17–10–3</td><td class="num">30</td>
          <td class="num">—</td><td class="num">—</td>
          <td class="num">—</td>
          <td><span class="badge badge-yes">RC &gt; RS</span></td>
        </tr>
        <tr class="baseline">
          <td><strong>H2 (reference)</strong></td><td>No similar-task retrieve</td>
          <td class="num">30/40</td>
          <td class="num">—</td><td class="num">—</td>
          <td class="num">—</td><td class="num">—</td>
          <td class="num">{fbal(h2_bal)}</td>
          <td><span class="badge badge-part">Reference</span></td>
        </tr>
        <tr>
          <td><strong>H0</strong></td><td>50/50 mix (Part A best on GPT tasks)</td>
          <td class="num">30/40</td>
          <td class="num">12–8–1</td><td class="num">21</td>
          <td class="num">—</td><td class="num">—</td>
          <td class="num">{fbal(h0_bal)}</td>
          <td><span class="badge badge-yes">Beat H2</span></td>
        </tr>
      </tbody>
    </table>
    <p><em>Bal. norm:</em> {METRIC_NOTE} (GPT partial coverage 30–31 tasks for RC/RS/H2/H0).</p>
    <p><strong>Part D verdict:</strong> On the <strong>21-task overlap with H2</strong>, both RC and RS edge H2 on task wins (RC 11–8–2, RS 11–9–1), and <strong>RC beats RS</strong> 17–10–3 on 30 shared tasks. On bal norm (drop-20%), ordering is <strong>H0 ({fbal(h0_bal)}) &gt; RS ({fbal(rs_bal)}) &gt; H2 ({fbal(h2_bal)}) &gt; RC ({fbal(rc_bal)})</strong>. However, <strong>both retrieval arms lose badly to H0 on task wins</strong> (RC 9–20–1, RS 10–19–1) — similar-task retrieval does not beat the simpler 50/50 mix. Not recommended over H0; finish GPT 40/40 repair before drawing final conclusions. Qwen RC/RS eval still TODO.</p>
  </div>"""

    # --- GPT leaderboard (39-arm summary order, drop20 metrics) ---
    gpt_lb: list[str] = []
    gpt_rows = list(csv.DictReader((P5 / "gpt_p5_p5plus_summary.csv").open(encoding="utf-8")))
    ranked = []
    for r in gpt_rows:
        m = drop.get(r["variant"])
        if m:
            ranked.append((m["bal"], r, m))
    ranked.sort(key=lambda x: -x[0])
    for i, (bal, r, m) in enumerate(ranked[:15], 1):
        label = r["label"].replace("gpt ptllm_tpe ", "")
        part = "A" if "/H" in label and "TRdy" not in label and " X" not in label.split("/")[0][-3:] else (
            "B" if "TRdy" in label or "TContrast" in label or "TStag" in label else (
                "C" if " X" in label else "Prior ablation"
            )
        )
        if "A2" in label or "T50" in label and "X50" not in label:
            part = "Prior ablation"
        hl = ' class="highlight"' if i == 1 else (' class="baseline"' if "X50 / H2" in label and "RC" not in label and "RS" not in label else "")
        gpt_lb.append(
            f'        <tr{hl}><td class="num">{i}</td><td>{label}</td><td>{part}</td>'
            f'<td class="num">{m["avg_rank"]:.2f}</td><td class="num">{bal:.3f}</td>'
            f'<td class="num">{m["avg_rank_cls"]:.2f}</td><td class="num">{m["avg_rank_rgs"]:.2f}</td></tr>'
        )

    gpt_lb_block = f"""  <h2>6. Full leaderboard (top 15 of 39 arms)</h2>
  <div class="card">
    <p>Metrics: {METRIC_NOTE}; avg rank vs all methods in the global seed-1 table. Sorted by bal. norm (drop-20%). GPT coverage varies by arm (see <code>datasets</code> column in CSV).</p>
    <table>
      <thead>
        <tr>
          <th class="num">#</th><th>Method</th><th>Part</th>
          <th class="num">Avg rank</th><th class="num">Bal. norm</th><th class="num">Cls rank</th><th class="num">Rgs rank</th>
        </tr>
      </thead>
      <tbody>
{chr(10).join(gpt_lb)}
      </tbody>
    </table>
    <p>Full table: <code>gpt_p5_p5plus_summary.csv</code> (39 rows); bal. norm from <code>mean_norm_table_pooled_drop20pct_seed1.csv</code>.</p>
  </div>"""

    # --- Qwen leaderboard ---
    qwen_summary = list(csv.DictReader((P5 / "qwen_p5_p5plus_summary.csv").open(encoding="utf-8")))
    q_ranked = []
    for r in qwen_summary:
        v = qwen_disk_variant(r["variant"])
        m = drop.get(v)
        if m:
            q_ranked.append((m["bal"], r, m))
    q_ranked.sort(key=lambda x: -x[0])
    q_lb: list[str] = []
    base_v = report["base_variant"]
    for i, (bal, r, m) in enumerate(q_ranked[:15], 1):
        display = r["label"].split(" ", 3)[-1] if " " in r["label"] else r["label"]
        part = "A"
        if "TRdy" in r["variant"] or "TContrast" in r["variant"] or "TStag" in r["variant"]:
            part = "B"
        elif "T50X" in r["variant"] and "X50" not in r["variant"]:
            part = "C"
        elif "P5plusA2" in r["variant"] or "P5plusT50" in r["variant"] or "P5plusA5T40" in r["variant"]:
            part = "Prior"
        hl = ' class="highlight"' if i == 1 else (' class="baseline"' if r["variant"] == base_v else "")
        q_lb.append(
            f'        <tr{hl}><td class="num">{i}</td><td>{display}</td><td>{part}</td>'
            f'<td class="num">{m["avg_rank"]:.2f}</td><td class="num">{bal:.3f}</td>'
            f'<td class="num">{m["avg_rank_cls"]:.2f}</td><td class="num">{m["avg_rank_rgs"]:.2f}</td></tr>'
        )

    qwen_lb_block = f"""  <h2>6b. Qwen s1b_attr_t2 leaderboard (top 15 of 37 arms)</h2>
  <div class="card">
    <p>Model: <code>llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2</code> on <code>real_hpo_v2_mltest20_algo40_p5_s1b_attr_t2_p5plus</code>.
    Metrics: {METRIC_NOTE}. Sorted by bal. norm (drop-20%).
    All arms below: <strong>algo-correct 40/40</strong> task slots after the Aug 2026 repair pass.</p>
    <table>
      <thead>
        <tr>
          <th class="num">#</th><th>Method</th><th>Part</th>
          <th class="num">Avg rank</th><th class="num">Bal. norm</th><th class="num">Cls rank</th><th class="num">Rgs rank</th>
        </tr>
      </thead>
      <tbody>
{chr(10).join(q_lb)}
      </tbody>
    </table>
    <p>Full table: <code>analysis/p5plus/qwen_p5_p5plus_summary.csv</code> (37 rows, all 40/40); bal. norm from <code>mean_norm_table_pooled_drop20pct_seed1.csv</code>.</p>
    <div class="callout callout-ok">
      <strong>Qwen vs GPT on dynamic gates:</strong> On Qwen, <strong>HPUCB_arm</strong> and <strong>TRdyFanova</strong> both beat the H2 baseline head-to-head — unlike GPT where only H0/HDUCB beat H2 and Fanova did not.
      The repaired Qwen sweep changes the Part B story: readiness+fANOVA is worth a confirmatory follow-up on Qwen even though it failed on GPT.
    </div>
  </div>"""

    # Replace sections via regex
    html = re.sub(
        r"  <h3>Part A — LLM call frequency.*?</div>\n\n  <h3>Part B",
        part_a_block + "\n\n  <h3>Part B",
        html,
        count=1,
        flags=re.DOTALL,
    )
    html = re.sub(
        r"  <h3>Part B — Region insertion timing.*?</div>\n\n  <h3>Part C",
        part_b_block + "\n\n  <h3>Part C",
        html,
        count=1,
        flags=re.DOTALL,
    )
    html = re.sub(
        r"  <h3>Part C — Explore / exploit.*?</div>\n\n  <h3>Part D",
        part_c_block + "\n\n  <h3>Part D",
        html,
        count=1,
        flags=re.DOTALL,
    )
    html = re.sub(
        r"  <h3>Part D — Similar-task retrieval.*?</div>\n\n  <h2>6\. Full leaderboard",
        part_d_block + "\n\n  <h2>6. Full leaderboard",
        html,
        count=1,
        flags=re.DOTALL,
    )
    html = re.sub(
        r"  <h2>6\. Full leaderboard \(top 15 of 39 arms\)</h2>.*?</div>\n\n  \n  <h2>6b\. Qwen",
        gpt_lb_block + "\n\n  \n  <h2>6b. Qwen",
        html,
        count=1,
        flags=re.DOTALL,
    )
    html = re.sub(
        r"  <h2>6b\. Qwen s1b_attr_t2 leaderboard \(top 15 of 37 arms\)</h2>.*?</div>\n\n  <h2>7\. Conclusions",
        qwen_lb_block + "\n\n  <h2>7. Conclusions",
        html,
        count=1,
        flags=re.DOTALL,
    )

    html = html.replace("drop worst 10%", "drop worst 20%")
    html_path.write_text(html, encoding="utf-8")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
