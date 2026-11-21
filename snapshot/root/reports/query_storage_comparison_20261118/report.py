import collections,csv,json,math,statistics,time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

ROOT=Path(__file__).resolve().parent
def read_lines(name):
    p=ROOT/name
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []
def geo(values):return math.exp(statistics.mean(map(math.log,values))) if values else None
def number(value):return f'{value:.3f}' if value is not None else 'failed / unavailable'
def main():
    assert json.loads((ROOT/'benchmark_state.json').read_text())['state']=='complete'
    assert json.loads((ROOT/'cold_probe_state.json').read_text())['state']=='complete'
    assert json.loads((ROOT/'hybrid_probe_state.json').read_text())['state']=='complete'
    samples=json.loads((ROOT/'samples.json').read_text());calls=read_lines('timings.jsonl');explain=read_lines('explain.jsonl');cache=read_lines('cache_probe.jsonl')
    def times(i,label,phase):
        return [r['result']['runtime'] for r in calls if r['sample']==i and r['storage']==label and r['phase']==phase and r['result']['status']=='completed']
    table=[]
    for s in samples:
        i=s['id'];a=times(i,'ram','repeated');b=times(i,'lustre','repeated')
        ca=times(i,'ram','private_file_cache_advice');cb=times(i,'lustre','private_file_cache_advice')
        row=dict(task=s['task'],source_method=s['historical_method'],historical_seconds=s['historical_seconds'],ram_repeats=len(a),lustre_repeats=len(b),ram_median=statistics.median(a) if a else None,lustre_median=statistics.median(b) if b else None,ram_mean=statistics.mean(a) if a else None,lustre_mean=statistics.mean(b) if b else None,ram_stdev=statistics.stdev(a) if len(a)>1 else None,lustre_stdev=statistics.stdev(b) if len(b)>1 else None)
        row['ram_over_lustre']=row['ram_median']/row['lustre_median'] if a and b else None
        row['ram_vs_lustre_percent']=100*(row['ram_over_lustre']-1) if row['ram_over_lustre'] else None
        row['ram_vs_historical_ratio']=row['ram_median']/s['historical_seconds'] if a else None
        row['lustre_vs_historical_ratio']=row['lustre_median']/s['historical_seconds'] if b else None
        row['cache_probe_ram']=ca[0] if ca else None;row['cache_probe_lustre']=cb[0] if cb else None
        row['cache_probe_ratio']=ca[0]/cb[0] if ca and cb else None
        for label in ['ram','lustre']:
            records=[x for x in calls if x['sample']==i and x['storage']==label and x['phase']=='private_file_cache_advice']
            row['cache_probe_'+label+'_status']=records[0]['result']['status'] if records else None
        ha=times(i,'ram_lustre_temp','hybrid_repeated');hb=times(i,'lustre','hybrid_repeated')
        row['hybrid_ram_median']=statistics.median(ha) if ha else None
        row['hybrid_lustre_median']=statistics.median(hb) if hb else None
        row['hybrid_ratio']=statistics.median(ha)/statistics.median(hb) if ha and hb else None
        row['hybrid_repeats']=len(ha)
        ee={e['storage']:e for e in explain if e['sample']==i}
        row['same_plan_shape']=ee['ram'].get('shape_sha256')==ee['lustre'].get('shape_sha256') if all('shape_sha256' in ee.get(l,{}) for l in ['ram','lustre']) else None
        row['same_result']=ee['ram'].get('result_sha256')==ee['lustre'].get('result_sha256') if all('result_sha256' in ee.get(l,{}) for l in ['ram','lustre']) else None
        row['hybrid_same_result']=ee['ram_lustre_temp'].get('result_sha256')==ee['lustre'].get('result_sha256') if all('result_sha256' in ee.get(l,{}) for l in ['ram_lustre_temp','lustre']) else None
        row['hybrid_same_plan_shape']=ee['ram_lustre_temp'].get('shape_sha256')==ee['lustre'].get('shape_sha256') if all('shape_sha256' in ee.get(l,{}) for l in ['ram_lustre_temp','lustre']) else None
        for label in ['ram','lustre']:
            plan=ee.get(label,{}).get('plan',{}).get('Plan',{})
            row[label+'_shared_read_blocks']=plan.get('Shared Read Blocks')
            row[label+'_shared_hit_blocks']=plan.get('Shared Hit Blocks')
            row[label+'_temp_read_blocks']=plan.get('Temp Read Blocks')
            row[label+'_temp_written_blocks']=plan.get('Temp Written Blocks')
        table.append(row)
    with (ROOT/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    valid=[r for r in table if r['ram_over_lustre'] is not None];ratios=[r['ram_over_lustre'] for r in valid]
    cold=[r['cache_probe_ratio'] for r in table if r['cache_probe_ratio'] is not None]
    previous=[r for r in valid if r['source_method']!='BOLT30_warm_start']
    exceptions=[r for r in calls if r['result']['status']!='completed']
    stats=dict(generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),samples=len(samples),historical_baseline_samples=len(previous),repeated_calls=sum(r['phase']=='repeated' for r in calls),recorded_calls=len(calls),noncompleted_calls=exceptions,explain_errors=[e for e in explain if 'error' in e],equal_results=sum(r['same_result'] is True for r in table),equal_plan_shapes=sum(r['same_plan_shape'] is True for r in table),warm_ram_over_lustre_geomean=geo(ratios),warm_ram_over_lustre_min=min(ratios),warm_ram_over_lustre_max=max(ratios),warm_within_5_percent=sum(abs(r-1)<=.05 for r in ratios),warm_within_10_percent=sum(abs(r-1)<=.1 for r in ratios),warm_mean_ram_seconds=statistics.mean(r['ram_mean'] for r in valid),warm_mean_lustre_seconds=statistics.mean(r['lustre_mean'] for r in valid),cold_probe_ram_over_lustre_geomean=geo(cold),cold_probe_ram_over_lustre_min=min(cold),cold_probe_ram_over_lustre_max=max(cold),cold_probe_resident_fraction_before=[c['before']['resident_fraction'] for c in cache],cold_probe_resident_fraction_after=[c['after_advice']['resident_fraction'] for c in cache],historical_comparison_ram_geomean=geo([r['ram_vs_historical_ratio'] for r in previous]),historical_comparison_lustre_geomean=geo([r['lustre_vs_historical_ratio'] for r in previous]),historical_comparison_ram_ratio_range=[min(r['ram_vs_historical_ratio'] for r in previous),max(r['ram_vs_historical_ratio'] for r in previous)],cache_probe_errors=[c for c in cache if c['after_advice']['failures']],spill_plans=sum(bool(r['ram_temp_written_blocks'] or r['lustre_temp_written_blocks']) for r in table))
    hybrid=[r for r in table if r['hybrid_ratio'] is not None]
    stats.update(hybrid_plans=len(hybrid),hybrid_repetitions=3,hybrid_ram_over_lustre_geomean=geo([r['hybrid_ratio'] for r in hybrid]),hybrid_same_result=sum(r['hybrid_same_result'] is True for r in hybrid),hybrid_same_plan_shape=sum(r['hybrid_same_plan_shape'] is True for r in hybrid))
    stats.update(cold_probe_successful_pairs=len(cold),cold_probe_noncompleted=[r for r in exceptions if r['phase']=='private_file_cache_advice'])
    (ROOT/'summary.json').write_text(json.dumps(stats,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(13,8),gridspec_kw={'width_ratios':[1.25,1]})
    y=np.arange(len(table));axes[0].scatter([r['ram_over_lustre'] for r in table],y,label='Warm: median of 5',color='#0072B2',s=35)
    axes[0].scatter([r['cache_probe_ratio'] for r in table],y,label='Private-file cache advice: one replay',color='#D55E00',marker='x',s=38)
    hy=[i for i,r in enumerate(table) if r['hybrid_ratio'] is not None]
    axes[0].scatter([table[i]['hybrid_ratio'] for i in hy],hy,label='RAM tables + Lustre spill: median of 3',color='#009E73',marker='s',s=30)
    axes[0].axvline(1,color='black',lw=1);axes[0].axvspan(.95,1.05,color='gray',alpha=.15)
    axes[0].set_yticks(y,[r['task'] for r in table]);axes[0].invert_yaxis();axes[0].set_xlabel('RAM runtime / Lustre runtime (lower = RAM faster)');axes[0].legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,-.10));axes[0].grid(axis='x',alpha=.2)
    old=np.array([r['historical_seconds'] for r in previous]);a=np.array([r['ram_median'] for r in previous]);b=np.array([r['lustre_median'] for r in previous])
    axes[1].scatter(old,a,label='New RAM',color='#0072B2');axes[1].scatter(old,b,label='New Lustre',color='#D55E00',marker='x')
    lo=min(old.min(),a.min(),b.min())*.8;hi=max(old.max(),a.max(),b.max())*1.2;axes[1].plot([lo,hi],[lo,hi],color='black',lw=1)
    axes[1].set_xscale('log');axes[1].set_yscale('log');axes[1].set_xlabel('Historical saved runtime (seconds)');axes[1].set_ylabel('New replay runtime (seconds)');axes[1].legend();axes[1].grid(alpha=.2)
    ticks=[.2,.5,1,2,5]
    axes[1].set_xticks(ticks,[str(t) for t in ticks]);axes[1].set_yticks(ticks,[str(t) for t in ticks])
    axes[1].xaxis.set_minor_formatter(NullFormatter());axes[1].yaxis.set_minor_formatter(NullFormatter())
    fig.suptitle('Query storage comparison — same machine, saved executable plans')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(ROOT/f'comparison.{ext}',dpi=180,bbox_inches='tight')
    pct=lambda v:f'{100*(v-1):+.1f}%'
    lines=[
        '# Query storage investigation and timing comparison',
        '',f"Generated {stats['generated_at']}",'',
        f"Previous runs did use RAM. In the matched eighteen-plan replay, RAM warm runtimes were **{100*(1-stats['warm_ram_over_lustre_geomean']):.1f}% lower** than Lustre runtimes overall (geometric mean of per-plan median ratios), so the two backends did not give identical runtime. The cache-advice and historical comparisons are detailed below. This is a bounded sample, not a full test200 rerun.",
        '', '## Verified historical storage', '',
        '| Run / record | Database storage | Temporary-file storage / qualification |',
        '|---|---|---|',
        '| Oct 4 train800 restart | Five local block-storage copies; one remote RAM copy | Contemporary audit explicitly warned against pooling absolute runtimes across storage backends. |',
        '| Oct 12 LOLBO/TPE test200 allocation | Six local block-storage copies and eight remote RAM copies | Saved pre-sequential plans establish this allocation; LOLBO completed Oct 13 and TPE Oct 22. Later config edits do not retroactively make every timing RAM-backed. |',
        '| Oct 15 Random Search recovery | Eight RAM database copies on LLMOptimizer4 | SQL spill moved to Lustre after RAM DiskFull errors. |',
        '| Oct 27 baseline storage migration record | Six local copies changed from block storage to RAM | Before/after plan snapshots retained; prior measured results retain their original conditions. |',
        '| Nov 6 FSBO pilot and original test200 pool | Fourteen RAM copies and two warmed Lustre copies | Original test200 workers subsequently configured Lustre spill tablespaces. |',
        '| Nov 7 FSBO test200 expansion | Four additional local RAM copies; total eighteen RAM and two Lustre copies across twenty workers | The additional RAM databases also used disk-backed spill tablespaces. |',
        '| SemanticOPT query evaluation setup | Its provisioner creates private databases under the Lustre campaign directory; retained data directories resolve to Lustre | This setup differs from the RAM-backed FSBO pool. Directory inventory is preparation evidence, not a simultaneous-worker count. |',
        '| Current paused ORPO+FSBO campaign, live verification | Fourteen RAM copies and ten Lustre copies | Six Lustre copies on LLMOptimizer2; fourteen RAM plus four Lustre copies on LLMOptimizer4. RAM workers still use Lustre spill tablespaces. |',
        '', 'Evidence: [historical inventory](historical_inventory.json), [live endpoints](live_inventory.json), [sample endpoint provenance](historical_sample_storage.json). The ten LOLBO/TPE sample endpoints map to five local block-storage copies and five remote RAM copies in the contemporaneous plan; individual FSBO sample endpoints are not recorded in their history metadata. Historical source documents are listed below.',
        '', '## Matched replay design', '',
        '- Eighteen saved plans: fifteen held-out test queries spanning fifteen CEB families, five plans each from completed FSBO, LOLBO and TPE evaluations; plus three saved warm-start plans for failure-affected tasks CEB_11A121/122/123.',
        '- The fifteen baseline plans were distinct actual successful SQL calls, excluding Bao imports and cache hits. Selection used the median distinct plan among historical runtimes from 0.05 through 8 seconds. This deliberately bounded sample is not the full 200-query evaluation or a random estimate of campaign performance.',
        '- One verified physical PostgreSQL base backup cloned into private RAM and Lustre directories on LLMOptimizer2. Both copies passed pg_verifybackup with the same manifest. Production database instances and optimization journals were not migrated or resumed.',
        '- Same PostgreSQL 16.15, data, statistics, executable SQL/hints, three CPU cores (0–2), 1 GiB shared_buffers, 64 MiB work_mem, up to two parallel workers, JIT on, 8 GiB temporary-file limit. Full pg_settings comparison differed only in paths, ports and sockets. Results are wall-clock seconds, using the production RuntimeOracle with read-only sessions.',
        '- Test clusters use the default empty temp_tablespaces setting: spill follows the database directory (RAM for the RAM test, Lustre for the Lustre test). Historical RAM-plus-Lustre-spill arrangements are therefore a separate condition for spill-heavy plans.',
        '- One first pass per storage, then five timed repetitions per plan with shuffled query order and alternating storage order. No optimizer or LLM inference work is included. Timeouts are capped at 60 seconds; failed/censored outcomes are never substituted into successful-runtime averages.',
        '- Separately, both private test servers were restarted per plan to clear shared_buffers, and POSIX_FADV_DONTNEED was applied only to the private Lustre database relation files. mincore checked resulting OS-cache residency. This is an advisory cache test, not proof that remote Lustre/server/controller caches are cold. No global cache dropping or production-server restart occurred.',
        '- EXPLAIN ANALYZE/BUFFERS and separate result-hash checks were run outside the timed repetitions. Data counts for title/name/movie_info/cast_info matched.',
        '- An additional matched test covers every sampled plan that spilled in EXPLAIN: RAM tables with a private Lustre temporary tablespace versus Lustre tables with Lustre spill, one warm-up and three paired repetitions, plus separate hybrid EXPLAIN and result checks. Only the test clone catalog was changed to create/drop this temporary tablespace.',
        '', '## Results', '',
        f"- Warm repetitions: **{stats['repeated_calls']} calls**; geometric mean of per-plan RAM/Lustre median ratios **{stats['warm_ram_over_lustre_geomean']:.3f}**, or **{pct(stats['warm_ram_over_lustre_geomean'])}** RAM runtime relative to Lustre.",
        f"- Mean SQL runtime over these equally weighted plans: RAM **{stats['warm_mean_ram_seconds']:.3f} s**, Lustre **{stats['warm_mean_lustre_seconds']:.3f} s**. This is a sample average, not the campaign's evaluation-time average.",
        f"- **{stats['warm_within_5_percent']}/{len(table)}** plans within ±5%; **{stats['warm_within_10_percent']}/{len(table)}** within ±10%. Per-plan warm ratios range **{min(ratios):.3f}–{max(ratios):.3f}**.",
        f"- Private-file cache-advice replay: **{stats['cold_probe_successful_pairs']}/{len(table)} successful pairs**, geometric mean RAM/Lustre ratio **{stats['cold_probe_ram_over_lustre_geomean']:.3f}** ({pct(stats['cold_probe_ram_over_lustre_geomean'])}); per-plan range **{min(cold):.3f}–{max(cold):.3f}**. Only one replay per storage per plan in this diagnostic. Non-completed calls: **{len(stats['cold_probe_noncompleted'])}**, retained separately in summary.json and excluded from the successful-pair ratio.",
        f"- Lustre client-cache residency after advice: **{min(stats['cold_probe_resident_fraction_after'])*100:.2f}%–{max(stats['cold_probe_resident_fraction_after'])*100:.2f}%** of examined relation pages. Files already warmed by prior reads explain why an ordinary replay can hide storage-read costs.",
        f"- EXPLAIN checks: **{stats['equal_plan_shapes']}/{len(table)}** matching plan shapes; **{stats['equal_results']}/{len(table)}** matching unordered query-result hashes. **{stats['spill_plans']}** sampled plans wrote temporary blocks in EXPLAIN, so spill-heavy behavior is only covered to that extent.",
        f"- Matching the historical RAM-table/Lustre-spill arrangement on **{stats['hybrid_plans']} spilling plans**, three paired repetitions each: RAM/Lustre geometric mean ratio **{stats['hybrid_ram_over_lustre_geomean']:.3f}** ({pct(stats['hybrid_ram_over_lustre_geomean'])}). Both conditions use Lustre for spill. Hybrid validation matched **{stats['hybrid_same_result']}/{stats['hybrid_plans']}** query results and **{stats['hybrid_same_plan_shape']}/{stats['hybrid_plans']}** plan shapes. This subgroup ratio is not the full eighteen-query ratio.",
        f"- Recorded timing calls: **{len(calls)}**, non-completed **{len(exceptions)}**; EXPLAIN errors **{len(stats['explain_errors'])}**. These short tests cannot certify that intermittent storage faults are repaired.",
        '', '## Comparison with the saved earlier test timings', '',
        f"For the fifteen completed baseline plans, new RAM medians / saved historical timings have geometric mean **{stats['historical_comparison_ram_geomean']:.3f}** ({pct(stats['historical_comparison_ram_geomean'])}); new Lustre medians / historical timings **{stats['historical_comparison_lustre_geomean']:.3f}** ({pct(stats['historical_comparison_lustre_geomean'])}). The RAM/historical per-plan range is **{stats['historical_comparison_ram_ratio_range'][0]:.3f}–{stats['historical_comparison_ram_ratio_range'][1]:.3f}**.",
        '', 'This historical comparison is descriptive: the earlier measurements were single executions under different machine, CPU and concurrent-load conditions, and some used different spill placement. It does not attribute those differences solely to RAM. Full per-query values are in comparison.csv.',
        '', '## Interpretation and recommendation', '',
        'RAM use is established historically. Runtime equality must be assessed from the controlled ratios above, not inferred from database folder names or one aggregate average. Warm-cache agreement does not establish equivalence for initial reads, spill-heavy queries or a heavily contended full campaign.',
        '', 'For a paper comparison, keep base-data and spill placement, CPU allocation, cache policy and timing protocol consistent across methods. Preserve historical storage boundaries; do not relabel or silently merge old and new seconds. A production RAM migration still needs its own recorded decision, and this investigation did not perform one.',
        '', 'A RAM-table migration would bypass Lustre for base-table reads. If spill files and optimization journals remain on Lustre, those operations still depend on the same storage service. RAM placement is a workaround for selected I/O paths, not a repair of the shared filesystem.',
        '', 'The campaign remains paused with all saved scores intact. Successful isolated reads do not resolve the previously observed I/O errors/short reads on the production data paths. The RAM probe establishes a feasible alternative to investigate, not proof of storage repair.',
        '', '## Per-query timings', '',
        '| Query | Saved method | Historical s | RAM median s | Lustre median s | RAM/Lustre | Cache-advice RAM/Lustre | Hybrid RAM/Lustre |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for r in table:lines.append(f"| {r['task']} | {r['source_method']} | {number(r['historical_seconds'])} | {number(r['ram_median'])} | {number(r['lustre_median'])} | {number(r['ram_over_lustre'])} | {number(r['cache_probe_ratio'])} | {number(r['hybrid_ratio']) if r['hybrid_ratio'] is not None else 'not sampled (no observed spill)'} |")
    lines+=['','## Artifacts and source documents','',
        '- [Timing plot](comparison.png), [PDF plot](comparison.pdf), [per-query CSV](comparison.csv), [machine-readable summary](summary.json).',
        '- samples.json records exact executable SQL, plans, source hashes and original timings. timings.jsonl contains every timed call, including unsuccessful calls if any. explain.jsonl contains execution plans and result hashes.',
        '- provision.log and clusters.json record backup verification and isolated server settings. environment.json and setting_differences.json record PostgreSQL configuration checks. cache_probe.jsonl records cache-residency checks.',
        '- /root/reports/queryopt_health_20261005/README.md',
        '- /root/data/cys/logs/query_test200_baselines_20261012/plan.before_sequential_unique.json and stage.lolbo.complete.json / stage.tpe.complete.json',
        '- /root/data/cys/logs/query_random_storage_20261015/README.md',
        '- /root/data/logs/query_fsbo_pilot_20261106_v1/README.md',
        '- /root/data/logs/query_fsbo_test200_50s_20261106_v1/storage_configuration.json',
        '- /root/data/logs/query_fsbo_local_expansion_20261107_v1/README.md and plan.json',
        '- /root/data/cys/baseline/campaigns/semanticopt_eval200_20261110/code/host.py:98 and historical_semanticopt_storage.json',
        '- /root/data/cys/baseline/campaigns/query_our_fsbo_vae80_test200_20261117/resources.json',
        '', 'Temporary test cluster cleanup is recorded separately in cleanup.json. The scripts and measurement evidence are retained for review.', '']
    (ROOT/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
