"""Snapshot completed BOLT queries and compare saved incumbents at equal budgets."""
import csv
import hashlib
import json
import pickle
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
BASE = Path('/root/data/cys/baseline/campaigns')
BOLT = BASE / 'query_bolt30_lolbo_total800_20261123'
OUR = BASE / 'query_our_fsbo_vae80_test200_20261117'
sys.path.insert(0, str(OUR / 'batch_code'))


def read(path, key):
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=30) as db:
        db.execute('BEGIN')
        raw = db.execute('SELECT value FROM kv WHERE key=?', (key,)).fetchone()[0]
    return pickle.loads(raw)['rows']


def curve(rows):
    assert len(rows) == 800
    best = np.inf
    result = []
    for row in rows:
        value = float(row['runtime'])
        assert value > 0 and np.isfinite(value)
        if not row['censored']:
            best = min(best, value)
        result.append(best)
    assert np.isfinite(result[29:]).all()
    return np.asarray(result)


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    tasks = sorted(p.parent.name for p in (BOLT / 'runs').glob('*/complete.json'))
    data = {'LLM + FSBO': [], 'BOLT / LOLBO': []}
    records = []
    evidence = {}
    warm_sum = 0.0
    for task in tasks:
        a = read(OUR / 'orpo_batch10/runs' / task / 'runtime.sqlite', 'optimization_state')
        b = read(BOLT / 'runs' / task / 'runtime.sqlite', 'lolbo_state')
        warm = lambda rows: [(r['plan'], r['runtime'], r['censored']) for r in rows[:30]]
        assert warm(a) == warm(b), task
        ca, cb = curve(a), curve(b)
        np.testing.assert_array_equal(ca[:30], cb[:30])
        warm_sum += ca[29]
        data['LLM + FSBO'].append(ca)
        data['BOLT / LOLBO'].append(cb)
        records.append(dict(task=task, shared_warm_best_seconds=float(ca[29]),
                            our_at_400=float(ca[399]), bolt_at_400=float(cb[399]),
                            our_at_800=float(ca[799]), bolt_at_800=float(cb[799]),
                            our_over_bolt_at_800=float(ca[799]/cb[799]),
                            our_reduction_percent=100*float(1-ca[799]/cb[799])))
        evidence[task] = {m: hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
                          for m, rows in [('LLM + FSBO', a), ('BOLT / LOLBO', b)]}
    assert tasks
    data = {m: np.asarray(v) for m, v in data.items()}
    a, b = data['LLM + FSBO'], data['BOLT / LOLBO']
    summary = dict(generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                   completed_queries=len(tasks), tasks=tasks, total_queries=200,
                   total_evaluations_per_query=800, shared_warm_evaluations_per_query=30,
                   shared_warm_summed_best_seconds=warm_sum, methods={}, paired={})
    for m, values in data.items():
        summary['methods'][m] = {
            f'at_{n}': dict(summed_best_runtime_seconds=float(values[:, n-1].sum()),
                             normalized_summed_runtime=float(values[:, n-1].sum()/warm_sum))
            for n in [30, 400, 800]}
    for n in [400, 800]:
        av, bv = a[:, n-1], b[:, n-1]
        equal = np.isclose(av, bv, rtol=1e-9, atol=1e-12)
        summary['paired'][f'at_{n}'] = dict(
            our_wins=int(((av < bv) & ~equal).sum()),
            bolt_wins=int(((bv < av) & ~equal).sum()), ties=int(equal.sum()),
            summed_runtime_reduction_percent=100*float(1-av.sum()/bv.sum()),
            summed_runtime_speedup=float(bv.sum()/av.sum()),
            geometric_mean_our_over_bolt=float(np.exp(np.log(av/bv).mean())),
            median_our_over_bolt=float(np.median(av/bv)))
    summary['notes'] = [
        'Cohort fixed by BOLT completion at snapshot time; it is not a random sample of all 200 queries.',
        'Both methods use identical 30 saved warm measurements and 770 further counted evaluations per query.',
        'Only uncensored measured runtimes improve incumbents; timeouts remain counted slots.',
        'Sum query incumbents first, then divide by the shared sum of best warm30 runtimes; lower is better.',
        'Current BOLT continuation runs the LOLBO optimizer. Historical standalone LOLBO is a separate experiment.',
        'SQL execution dates, host contention and storage conditions can differ; one saved run per method, no uncertainty bands.',
        'Running campaign journals are read only; no unfinished query is extrapolated.'
    ]
    write_csv('per_query.csv', records)
    write_csv('convergence.csv', [dict(method=m, queries=len(tasks), total_evaluations=n,
                                     summed_best_runtime_seconds=float(v[:, n-1].sum()),
                                     normalized_summed_runtime=float(v[:, n-1].sum()/warm_sum))
                                for m,v in data.items() for n in range(30,801)])
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 11})
    fig, ax = plt.subplots(figsize=(8.5,5.5))
    for (m,v), color in zip(data.items(), ['#B33D84','#6A51A3']):
        ax.step(np.arange(30,801), v[:,29:].sum(axis=0)/warm_sum, where='post',
                label=m, color=color, lw=2.2)
    ax.axvline(400, color='#999999', ls=':', lw=1)
    ax.set(xlim=(0,800), xlabel='Total evaluations per query (30 warm starts included)',
           ylabel='Summed best runtime / summed warm30 best',
           title=f'LLM + FSBO vs BOLT / LOLBO · {len(tasks)} completed queries')
    ax.grid(axis='y', alpha=.2)
    ax.legend(frameon=False)
    fig.text(.5,.015,'Same queries and initialization; lower is better. Completion-selected cohort; historical execution conditions differ.',
             ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1))
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/f'normalized_summed_runtime.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'row_digests.json').write_text(json.dumps(evidence,indent=2)+'\n')
    np.savez_compressed(OUT/'curves.npz',tasks=np.asarray(tasks),**data)
    paired=summary['paired']['at_800']
    text = [f'# Query comparison: {len(tasks)} completed BOLT queries', '', summary['generated_at'], '',
            f"At 800 total evaluations, LLM + FSBO wins {paired['our_wins']} queries; BOLT / LOLBO wins {paired['bolt_wins']}; ties {paired['ties']}.",
            f"LLM + FSBO reduces summed best runtime by {paired['summed_runtime_reduction_percent']:.3f}% relative to BOLT / LOLBO.", '',
            *['- '+s for s in summary['notes']], '',
            'Outputs: normalized_summed_runtime.png/.pdf/.svg, per_query.csv, convergence.csv, summary.json, curves.npz, row_digests.json.',
            'Refresh with python3 build_report.py.']
    (OUT/'README.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
