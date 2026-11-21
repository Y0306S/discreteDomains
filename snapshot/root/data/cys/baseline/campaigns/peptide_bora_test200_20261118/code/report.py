"""Publish full-budget peptide BORA curves without extra oracle calls."""
import csv
from common import ROOT, atomic, config, read


def report():
    cfg = config()
    summaries, curves = [], []
    for task in cfg['tasks']:
        result = read(ROOT/'runs'/task/'result.json')
        rows = read(ROOT/'runs'/task/'observations.json')
        assert result['status'] == 'complete' and result['evaluations'] == cfg['fresh_budget']
        assert len(rows) == 100+cfg['fresh_budget'] and len({r['sequence'] for r in rows}) == len(rows)
        best = min(r['mic'] for r in rows[:100])
        initial = best
        curves.append(dict(task=task,evaluations=0,best_mic=best))
        for n,row in enumerate(rows[100:],1):
            assert row['evaluation'] == n
            best = min(best,row['mic'])
            if n % 50 == 0:
                curves.append(dict(task=task,evaluations=n,best_mic=best))
        summaries.append(dict(task=task,initial_mic=initial,best_mic=best,
            improvement_pct=100*(initial-best)/initial,wall_seconds=result['elapsed_seconds']))
    for name,rows in [('paired_results.csv',summaries),('convergence.csv',curves)]:
        with open(ROOT/name,'w') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    atomic(ROOT/'summary.json',dict(status='complete',audits_passed=True,tasks=summaries))
    (ROOT/'RESULTS.md').write_text('BORA peptide adaptation: 200 held-out tasks, 4,000 fresh evaluations per task.\n\n'
        'See paired_results.csv, convergence.csv, config.json and README.md for scores, timing and protocol adaptations.\n')
