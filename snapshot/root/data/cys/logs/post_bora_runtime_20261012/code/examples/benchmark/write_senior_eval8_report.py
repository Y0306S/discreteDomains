#!/usr/bin/env python3
"""Regenerate eval8 senior CSV with 8-method-only ranks and HTML report."""

from __future__ import annotations

import csv
import math
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "benchmark_data/real_hpo_v2/analysis/eval8/all_valid_best/eval8_senior_top5_baselines.csv"
OUT_CSV = REPO / "benchmark_data/real_hpo_v2/analysis/eval8/all_valid_best/eval8_senior_top5_baselines_rank8.csv"
OUT_HTML = REPO / "benchmark_data/real_hpo_v2/analysis/eval8/all_valid_best/eval8_senior_report.html"

DATASETS = ["yeast", "car", "PhishingWebsites", "segment", "boston", "tecator", "socmob", "pol"]
CLS = {"yeast", "car", "PhishingWebsites", "segment"}

SCHEME = {
    "ptllm_tpe (P16/H2)": {
        "prompt": "P16 — TCS + 双阶段 LLM（Analysis Agent → Optimizer Agent）",
        "hybrid": "H2 — 70% LLM / 30% TPE（warm-start 后概率混合）",
        "logic": (
            "确定性 TCS 汇总全部 trial 轨迹；先由 Analysis 做六段诊断与建议，"
            "再由 Optimizer 输出完整 JSON 配置。每轮 LLM 历史步 2 次 API 调用。"
        ),
    },
    "ptllm_tpe (P6/H5)": {
        "prompt": "P6 — top5/worst5/recent10（无 run summary）",
        "hybrid": "H5 — 前 10 trial 全 LLM，之后 100% TPE",
        "logic": "历史块：最优5 + 最差5 + 最近10；无运行摘要。前期靠 LLM 探索，后期交给 TPE 收敛。",
    },
    "ptllm_tpe (P5/H0)": {
        "prompt": "P5 — top5/worst5/recent10 + run summary",
        "hybrid": "H0 — 50% LLM / 50% TPE",
        "logic": "经典历史提示：最优/最差/最近 + 运行级摘要；与 TPE 五五混合。",
    },
    "ptllm_tpe (P5/H4)": {
        "prompt": "P5 — 同上",
        "hybrid": "H4 — 前 20 trial 全 LLM，之后 50/50 混合",
        "logic": "P5 提示 + 更长 LLM 独占阶段，再进入 LLM/TPE 混合。",
    },
    "ptllm_tpe (P8/H0)": {
        "prompt": "P8 — top6/worst4/recent10 + summary（接近生产默认）",
        "hybrid": "H0 — 50% LLM / 50% TPE",
        "logic": "略宽的最优池 + 摘要；混合调度与 P5/H0 相同。",
    },
    "smac": {
        "prompt": "—",
        "hybrid": "—",
        "logic": "OpenBox SMAC：概率随机森林代理 + EI 采集；纯贝叶斯优化基线，无 LLM。",
    },
    "tpe": {
        "prompt": "—",
        "hybrid": "—",
        "logic": "Optuna TPE：树结构 Parzen 估计器；经典 HPO 基线。",
    },
    "cmaes": {
        "prompt": "—",
        "hybrid": "—",
        "logic": "CMA-ES 进化策略；连续空间局部/refine 基线。",
    },
}


def competition_ranks(scores: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(scores, key=lambda x: x[1], reverse=True)
    ranks: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        avg = (i + 1 + j + 1) / 2
        for k in range(i, j + 1):
            ranks[ordered[k][0]] = avg
        i = j + 1
    return ranks


def fmt_rank(r: float) -> str:
    return str(int(r)) if r == int(r) else f"{r:.1f}"


TCS_SECTION_HTML = """
  <h2>2. P16 / TCS 流水线详解（排名第一方案）</h2>
  <p class="note">
    TCS（Trajectory Context Summary）来自论文 LLM-TCS-HPT 思路，在本项目中适配为
    <strong>单算法 random_forest + holdout 评估</strong>。P16 相对 P5 的核心差异不是 hybrid，
    而是<strong>用结构化 TCS 替代采样历史块</strong>，并拆成 Analysis + Optimizer 两次 LLM 调用。
  </p>

  <div class="flow-diagram">
    <div class="flow-step"><span class="tag code">代码</span> ① TCS 汇总器<br><small>读取全部 trial → 确定性文本报告（无 LLM）</small></div>
    <div class="flow-arrow">→</div>
    <div class="flow-step"><span class="tag llm">LLM #1</span> ② Analysis Agent<br><small>读 TCS → 六段诊断与建议（自然语言，无 JSON）</small></div>
    <div class="flow-arrow">→</div>
    <div class="flow-step"><span class="tag llm">LLM #2</span> ③ Optimizer Agent<br><small>读 TCS + Analysis → Rationale + JSON 配置</small></div>
    <div class="flow-arrow">→</div>
    <div class="flow-step"><span class="tag eval">评估</span> ④ MindWare 评估<br><small>valid 分数写回 KD / TPE study</small></div>
  </div>

  <p class="note">Warm-start 前 4 个 trial 不带历史；从第 5 个 trial 起，每次轮到 LLM 时走上述三步（TCS 免费，2 次 API）。</p>

  <h3>2.1 TCS 汇总器（确定性，非 LLM）</h3>
  <p>输入：当前任务全部成功 trial 的 <code>(config, validation_score)</code>。输出：固定格式的文本块，包含以下 <strong>6 个子节</strong>：</p>
  <table class="detail-table">
    <thead><tr><th>子节</th><th>内容</th><th>作用</th></tr></thead>
    <tbody>
      <tr>
        <td><strong>CURRENT SITUATION</strong></td>
        <td>已评估 trial 数、历史最优/最新分数、<code>rel_score</code>（任务内 min-max 归一化，1.0=最好）、近期趋势（IMPROVING/DECLINING/STAGNANT）、距上次刷新最优已过去多少 trial</td>
        <td>让 Agent 判断当前处于探索、停滞还是退化</td>
      </tr>
      <tr>
        <td><strong>LATEST EXPERIMENT</strong></td>
        <td>上一轮 trial 的完整超参 JSON + rel_score + rank</td>
        <td>锚定「刚发生了什么」</td>
      </tr>
      <tr>
        <td><strong>PREVIOUS EXPERIMENT COMPARISON</strong></td>
        <td>与上上轮的分差、每个超参 changed/unchanged</td>
        <td>显式对比相邻两步</td>
      </tr>
      <tr>
        <td><strong>DETAILED HYPERPARAMETER ANALYSIS</strong></td>
        <td>每个超参：已试唯一值数、最优/最差均值、当前值、impact range；未试过的标 NEVER TESTED</td>
        <td>比 P5 的 top/worst 列表更系统的边际统计</td>
      </tr>
      <tr>
        <td><strong>TOP / WORST TRIALS</strong></td>
        <td>历史最优 5 与最差 5 的 config 参考</td>
        <td>提供可模仿与需避开的锚点</td>
      </tr>
      <tr>
        <td><strong>EVALUATED CONFIGURATIONS</strong></td>
        <td>所有已评估配置的完整列表（禁止重复）</td>
        <td>硬性去重约束</td>
      </tr>
    </tbody>
  </table>

  <h3>2.2 Analysis Agent（LLM 第 1 次调用）</h3>
  <div class="agent-box">
    <h4>输入</h4>
    <ul>
      <li><strong>System：</strong>角色 = HPO 分析专家；目标 = 提升 validation 分数；必须考虑搜索空间中<strong>全部</strong>超参；<strong>不得输出完整配置</strong></li>
      <li><strong>User：</strong>完整 TCS 文本 + 分析任务说明</li>
    </ul>
    <h4>必须输出的六段结构（自然语言）</h4>
    <ol class="compact">
      <li><strong>PROBLEM DIAGNOSIS</strong> — 什么阻碍了进一步提升？</li>
      <li><strong>HYPERPARAMETER IMPACT ASSESSMENT</strong> — 逐个超参：改大/改小/help/hurt</li>
      <li><strong>PRIMARY ACTION</strong> — 优先改哪个超参、为什么</li>
      <li><strong>SPECIFIC RECOMMENDATION</strong> — 主超参的目标值或方向（如「min_samples_split → 10」）</li>
      <li><strong>REASONING</strong> — 为何优于其他改法</li>
      <li><strong>EXPECTED OUTCOME</strong> — 预期 rel_score 提升幅度</li>
    </ol>
    <h4>输出物</h4>
    <p>纯文本分析报告（无 JSON）。该报告会原样传给 Optimizer Agent。</p>
  </div>

  <h3>2.3 Optimizer Agent（LLM 第 2 次调用）</h3>
  <div class="agent-box optimizer">
    <h4>输入</h4>
    <ul>
      <li><strong>System：</strong>角色 = 单算法 HPO Optimizer；固定算法（如 random_forest）；读 TCS + Analysis 报告；最大化 validation 分数</li>
      <li><strong>User（按顺序拼接）：</strong>
        <ol class="compact">
          <li>Dataset profile（样本量、特征数、任务类型、指标名等）</li>
          <li>Fixed algorithm</li>
          <li>Search space（每个超参的类型、范围、choices）</li>
          <li>完整 <strong>TCS 文本</strong>（与 Analysis 看到的是同一份）</li>
          <li><strong>Analysis Agent Report:</strong> + 上一步六段分析全文</li>
          <li>输出格式说明</li>
        </ol>
      </li>
    </ul>
    <h4>必须输出的结构</h4>
    <pre class="format-block">Rationale: &lt;一句话说明下一步配置的理由&gt;

JSON:
{"config": {"&lt;超参名&gt;": "&lt;值&gt;", ...}}</pre>
    <h4>约束</h4>
    <ul>
      <li>必须为 Search space 中<strong>每一个</strong>可调超参赋值</li>
      <li>不得返回 TCS「EVALUATED CONFIGURATIONS」中已出现过的配置</li>
      <li>超参名必须与 Search space 完全一致</li>
      <li>解析后由 KnowledgeDatabase 做 schema 校验、越界 clip、去重；失败则 reprompt 或随机 fallback</li>
    </ul>
    <h4>与 Analysis 的关系（实测，1,620 个 history trial）</h4>
    <ul>
      <li>约 <strong>99.6%</strong> 会改动 Analysis 点名的「主超参」</li>
      <li>仅约 <strong>45%</strong> 主超参取值与 SPECIFIC RECOMMENDATION <strong>完全一致</strong></li>
      <li>约 <strong>83%</strong> 会同时改动 2 个以上超参 — Optimizer 把 Analysis 当作<strong>战略参考</strong>，综合 section 2 的全局评估后给出<strong>完整配置</strong>，而非机械复制 section 4</li>
    </ul>
  </div>

  <h3>2.4 与 P5 历史提示的对比</h3>
  <table class="detail-table">
    <thead><tr><th>维度</th><th>P5 / P6 / P8（历史块）</th><th>P16（TCS 双 Agent）</th></tr></thead>
    <tbody>
      <tr><td>上下文来源</td><td>窗口内采样：top-K、worst-K、recent-K（+ 可选 summary）</td><td>确定性全轨迹 TCS + 超参边际分析</td></tr>
      <tr><td>LLM 调用次数</td><td>每步 1 次</td><td>每步 2 次（Analysis + Optimizer）</td></tr>
      <tr><td>推理结构</td><td>直接 Rationale + JSON</td><td>先六段分析，再 Rationale + JSON</td></tr>
      <tr><td>同 hybrid H2 表现</td><td>P5/H2 avg_rank 全池 ~30（8 法池靠后）</td><td>P16/H2 当前最佳</td></tr>
    </tbody>
  </table>

  <h2>3. 各方案逻辑说明（简表）</h2>
"""


def main() -> None:
    rows = list(csv.DictReader(SRC.open(encoding="utf-8")))

    for ds in DATASETS:
        scores = [(r["display_name"], float(r[f"score_{ds}"])) for r in rows]
        ranks = competition_ranks(scores)
        for r in rows:
            r[f"rank8_{ds}"] = fmt_rank(ranks[r["display_name"]])

    for r in rows:
        rank_vals = [float(r[f"rank8_{ds}"]) for ds in DATASETS]
        cls_r = [float(r[f"rank8_{ds}"]) for ds in DATASETS if ds in CLS]
        rgs_r = [float(r[f"rank8_{ds}"]) for ds in DATASETS if ds not in CLS]
        r["avg_rank8"] = f"{sum(rank_vals) / len(rank_vals):.2f}"
        r["avg_rank8_cls"] = f"{sum(cls_r) / len(cls_r):.2f}"
        r["avg_rank8_rgs"] = f"{sum(rgs_r) / len(rgs_r):.2f}"

    rows.sort(key=lambda r: float(r["avg_rank8"]))
    for i, r in enumerate(rows, 1):
        r["rank8"] = str(i)

    fieldnames = list(rows[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    html_rows = ""
    for r in rows:
        ds_cells = "".join(f"<td>{r[f'rank8_{ds}']}</td>" for ds in DATASETS)
        html_rows += f"""
    <tr>
      <td><strong>{r['rank8']}</strong></td>
      <td>{r['display_name']}</td>
      <td>{r['role']}</td>
      <td>{r['avg_rank8']}</td>
      <td>{r['avg_rank8_cls']}</td>
      <td>{r['avg_rank8_rgs']}</td>
      <td>{r['mean_balanced_norm']}</td>
      {ds_cells}
    </tr>"""

    scheme_cards = ""
    for r in rows:
        sch = SCHEME.get(r["display_name"], {})
        scheme_cards += f"""
    <article class="card">
      <div class="rank">#{r['rank8']}</div>
      <h3>{r['display_name']}</h3>
      <p class="meta">avg_rank8 = <strong>{r['avg_rank8']}</strong> · cls {r['avg_rank8_cls']} · rgs {r['avg_rank8_rgs']}</p>
      <dl>
        <dt>Prompt / 上下文</dt><dd>{sch.get('prompt', '—')}</dd>
        <dt>Hybrid 调度</dt><dd>{sch.get('hybrid', '—')}</dd>
        <dt>核心逻辑</dt><dd>{sch.get('logic', '—')}</dd>
      </dl>
    </article>"""

    ds_headers = "".join(f"<th>{ds}</th>" for ds in DATASETS)
    winner = rows[0]["display_name"]
    winner_avg = rows[0]["avg_rank8"]

    tcs_html = TCS_SECTION_HTML

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Eval8 Senior Report — Top 5 LLM + Baselines</title>
  <style>
    :root {{
      --ink: #1a2420; --muted: #5c6b64; --paper: #f8f6f0; --line: #d4cfc0;
      --accent: #0d5c4d; --accent2: #9a3412; --card: #fffef9;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; font-family: "PingFang SC", "Noto Sans SC", "Segoe UI", sans-serif;
      color: var(--ink); background: linear-gradient(160deg, #f0f7f4, var(--paper));
      line-height: 1.6;
    }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 40px 24px 64px; }}
    header {{
      background: var(--card); border: 1px solid var(--line); border-radius: 16px;
      padding: 32px; margin-bottom: 28px; box-shadow: 0 8px 24px rgba(0,0,0,.06);
    }}
    h1 {{ margin: 0 0 8px; font-size: 1.75rem; }}
    .subtitle {{ color: var(--muted); margin: 0; }}
    .pills {{ margin-top: 16px; display: flex; flex-wrap: wrap; gap: 8px; }}
    .pill {{
      font-size: 0.85rem; padding: 4px 12px; border-radius: 999px;
      background: #e8f2ed; color: var(--accent);
    }}
    h2 {{ font-size: 1.25rem; margin: 32px 0 12px; color: var(--accent); }}
    p.note {{ color: var(--muted); font-size: 0.95rem; }}
    .callout {{
      background: #fff7e6; border-left: 4px solid var(--accent2);
      padding: 14px 18px; border-radius: 8px; margin: 16px 0;
    }}
    table {{
      width: 100%; border-collapse: collapse; font-size: 0.88rem;
      background: var(--card); border-radius: 12px; overflow: hidden;
      border: 1px solid var(--line);
    }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: center; }}
    th {{ background: #1e3d34; color: #fff; font-weight: 600; }}
    th:nth-child(1), th:nth-child(2), td:nth-child(1), td:nth-child(2) {{ text-align: left; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #f5faf7; }}
    .grid {{ display: grid; gap: 16px; }}
    .card {{
      background: var(--card); border: 1px solid var(--line); border-radius: 12px;
      padding: 20px 22px; position: relative;
    }}
    .card .rank {{
      position: absolute; top: 16px; right: 18px; font-size: 1.5rem;
      font-weight: 700; color: var(--accent); opacity: .35;
    }}
    .card h3 {{ margin: 0 0 8px; font-size: 1.1rem; padding-right: 48px; }}
    .card .meta {{ margin: 0 0 12px; font-size: 0.9rem; color: var(--muted); }}
    dl {{ margin: 0; display: grid; grid-template-columns: 110px 1fr; gap: 6px 12px; font-size: 0.92rem; }}
    dt {{ color: var(--muted); font-weight: 600; }}
    dd {{ margin: 0; }}
    footer {{ margin-top: 40px; font-size: 0.85rem; color: var(--muted); }}
    .flow-diagram {{
      display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
      padding: 18px; background: #eef6f2; border-radius: 12px; border: 1px dashed #9db4aa;
      margin: 16px 0; font-size: 0.88rem;
    }}
    .flow-step {{
      flex: 1 1 140px; background: var(--card); border: 1px solid var(--line);
      border-radius: 10px; padding: 12px 14px; line-height: 1.45;
    }}
    .flow-arrow {{ color: var(--accent); font-weight: 700; font-size: 1.2rem; }}
    .tag {{
      display: inline-block; font-size: 0.72rem; font-weight: 700;
      padding: 2px 8px; border-radius: 4px; margin-bottom: 6px;
    }}
    .tag.code {{ background: #e8e8e8; color: #444; }}
    .tag.llm {{ background: #dbeafe; color: #1e40af; }}
    .tag.eval {{ background: #fce7f3; color: #9d174d; }}
    .detail-table {{ font-size: 0.9rem; margin: 12px 0; }}
    .detail-table td:first-child {{ white-space: nowrap; font-weight: 600; }}
    .agent-box {{
      background: #f8fbfa; border: 1px solid #c5ddd4; border-radius: 12px;
      padding: 18px 22px; margin: 12px 0 24px;
    }}
    .agent-box.optimizer {{ border-color: #b8cce8; background: #f7f9fc; }}
    .agent-box h4 {{ margin: 14px 0 6px; font-size: 0.95rem; color: var(--accent); }}
    .agent-box h4:first-child {{ margin-top: 0; }}
    ol.compact {{ margin: 6px 0; padding-left: 1.3rem; }}
    ol.compact li {{ margin: 4px 0; }}
    pre.format-block {{
      background: #1e2a26; color: #e8f3ea; padding: 14px 16px; border-radius: 8px;
      font-size: 0.85rem; overflow-x: auto; margin: 8px 0;
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>Eval8 基准测试 — Senior 汇报</h1>
    <p class="subtitle">Top 5 LLM 方案 + 3 个经典基线 · 8 数据集 · 80 trials · random_forest · gpt-4o-mini</p>
    <div class="pills">
      <span class="pill">生成日期：{date.today().isoformat()}</span>
      <span class="pill">8/8 完整覆盖</span>
      <span class="pill">排名池：本表 8 个方法互比（rank 1–8）</span>
      <span class="pill">主指标：avg_rank8（越低越好）</span>
    </div>
  </header>

  <div class="callout">
    <strong>排名说明：</strong>本报告 <code>rank8</code> / <code>rank8_*</code> / <code>avg_rank8</code>
    仅在<strong>下表 8 个方法</strong>之间重新计算（1=最好）。
    源 CSV 中 <code>avg_rank</code> 为全量 63 变体池排名，二者不可直接对比。
    分类：balanced accuracy；回归：negated MSE（均为越高越好）。
  </div>

  <h2>1. 总览排行榜（8 法互比）</h2>
  <table>
    <thead>
      <tr>
        <th>Rank</th><th>方法</th><th>类型</th>
        <th>avg_rank8</th><th>cls</th><th>rgs</th><th>balanced_norm</th>
        {ds_headers}
      </tr>
    </thead>
    <tbody>{html_rows}</tbody>
  </table>

{tcs_html}
  <p class="note">LLM 方案使用 <code>ptllm_tpe</code>（LLM 提议与 TPE 混合调度）；基线为纯 classical 优化器。</p>
  <div class="grid">{scheme_cards}</div>

  <h2>4. 实验设置</h2>
  <ul>
    <li><strong>数据集（8）：</strong>yeast, car, PhishingWebsites, segment（cls）；boston, tecator, socmob, pol（rgs）</li>
    <li><strong>预算：</strong>80 trials，seed=1，holdout validation</li>
    <li><strong>mean_balanced_norm：</strong>任务内归一化后的 cls/rgs 均衡均值（0–1，越高越好）</li>
    <li><strong>入选：</strong>8/8 完整的 Top-5 LLM + smac / tpe / cmaes</li>
  </ul>

  <h2>5. 主要结论</h2>
  <ul>
    <li><strong>{winner}</strong> 在 8 法互比中排名第 1（avg_rank8 = {winner_avg}）。</li>
    <li>Top 5 均为 LLM+TPE；三个经典基线在本池中排名靠后（6–8）。</li>
    <li>P16（TCS 双 Agent）相对 P5 历史提示 + 同 hybrid 方案优势显著，是当前最佳配置。</li>
    <li>进行中：P16 LLM 比例消融、P30 单 Agent 合并消融、Qwen SFT 训练路线评估。</li>
  </ul>

  <footer>
    CSV：<code>eval8_senior_top5_baselines_rank8.csv</code> ·
    HTML：<code>eval8_senior_report.html</code>
  </footer>
</main>
</body>
</html>"""

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_HTML}")
    for r in rows:
        print(f"  {r['rank8']}. {r['display_name']:22} avg_rank8={r['avg_rank8']}")


if __name__ == "__main__":
    main()
