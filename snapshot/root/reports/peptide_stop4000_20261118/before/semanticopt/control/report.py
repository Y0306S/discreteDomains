"""Audit retained observations and export actual MIC curves at matched budgets."""
import csv
import hashlib
import json
import math
import importlib.util
import os
from pathlib import Path
import sqlite3
import sys

if importlib.util.find_spec('Levenshtein') is None:
    python='/root/data/cys/envs/bolt_clean_venv/bin/python'
    if Path(sys.executable).resolve()==Path(python).resolve():
        raise RuntimeError('Report environment is missing Levenshtein')
    os.execv(python,[python,str(Path(__file__).resolve())])

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from protocol import canonical, peptide_measure


def report():
    cfg=json.loads((ROOT/'extension.json').read_text())
    imports=json.loads((ROOT/'evidence/import_audit.json').read_text())
    checks={a['task']:a for a in imports['tasks']}
    summaries=[];curves=[];fsbo_curves={};comparisons=[]
    for task in cfg['tasks']:
        path=ROOT/'peptide/runs'/task/'observations.sqlite'
        db=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
        raw=db.execute('SELECT n,value FROM records ORDER BY n').fetchall()
        old=hashlib.sha256(json.dumps(raw[:1000],separators=(',',':')).encode()).hexdigest()
        assert old==checks[task]['source_prefix_sha256'],task+' imported history changed'
        assert len(raw)==5000 and [n for n,_ in raw]==list(range(5000))
        initial=json.loads(db.execute("SELECT value FROM state WHERE key='initial'").fetchone()[0])
        reference=json.loads(Path('/root/data/cys/baseline/campaigns/semanticopt_eval200_20261110/peptide/initialization.json').read_text())[task]['reference']
        rows=[json.loads(v) for _,v in raw]
        seen={canonical(r['configuration']) for r in initial}
        best=min(r['objective'] for r in initial);initial_best=best
        curve=[best];fallback=0
        for i,row in enumerate(rows,1):
            assert row['evaluation']==i and math.isfinite(row['objective'])
            assert peptide_measure(row['configuration']['sequence'],reference)['feasible']
            key=canonical(row['configuration']);assert key not in seen;seen.add(key)
            assert row['status']=='success' and row['feasible']
            best=min(best,row['objective']);curve.append(best)
            fallback+=row['proposal_source']=='random_fallback'
        for budget in range(0,5001,50):
            curves.append(dict(task=task,evaluations=budget,best_mic=curve[budget]))
        summaries.append(dict(task=task,initial_mic=initial_best,mic_at_1000=curve[1000],
            mic_at_5000=curve[5000],relative_improvement_after_1000=(curve[1000]-curve[5000])/curve[1000],
            random_fallback_evaluations=fallback,
            additional_fallback_evaluations=sum(r['proposal_source']=='random_fallback' for r in rows[1000:])))
        for arm in ['author_frozen','adapted_frozen','adapted_online']:
            path=Path(cfg['fsbo_experiment'])/'evaluation'/arm/task/'result.json'
            if not path.exists():continue
            result=json.loads(path.read_text())
            assert result['status']=='complete'
            best=result['initialization_best_mic'];fresh=0;values=[(0,best)]
            group=[];key=None
            def commit():
                nonlocal fresh,best
                fresh+=sum(not r['cached'] for r in group)
                best=min([best]+[r['mic'] for r in group if r['feasible']])
                values.append((fresh,best))
            for row in result['observations']:
                if row['phase']=='initialization':continue
                current=(row['phase'],row['elapsed'])
                if key is not None and current!=key:
                    commit();group=[]
                key=current;group.append(row)
            if group:commit()
            assert fresh==result['fresh_evaluations']
            fsbo_curves[task,arm]=values
            matched=min(5000,fresh)
            comparisons.append(dict(task=task,fsbo_arm=arm,matched_fresh_budget=matched,
                semanticopt_best_mic=curve[matched],fsbo_best_mic=values[-1][1],
                fsbo_actual_fresh_evaluations=fresh,fsbo_wall_seconds=result['wall_seconds']))
        db.close()
    for name,data in [('paired_results.csv',summaries),('convergence.csv',curves),('fsbo_comparison.csv',comparisons)]:
        if not data:continue
        with (ROOT/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    (ROOT/'summary.json').write_text(json.dumps(dict(status='complete',audits_passed=True,tasks=summaries),indent=2)+'\n')
    content=['# SemanticOPT continuation: same ten peptide tasks','',
        'All ten tasks reached 5,000 distinct scored proposals, excluding 100 initial observations. '
        'The first 1,000 saved observations per task were preserved exactly. Lower predicted MIC is better.','',
        '| Task | MIC at 1,000 | MIC at 5,000 | Further improvement |',
        '|---|---:|---:|---:|']
    for row in summaries:
        content.append(f"| {row['task']} | {row['mic_at_1000']:.4f} | {row['mic_at_5000']:.4f} | {100*row['relative_improvement_after_1000']:.2f}% |")
    content+=['','Seed 42; one trajectory per task. Original SemanticOPT generation and labeled random-fallback '
        'policy retained. Hardware changed to two A16 cards per replica; runtime is not directly comparable '
        'with the original hardware. FSBO uses seed 1. Offline FSBO/VAE training costs are separate.',
        '', 'See paired_results.csv and convergence.csv for measured values.']
    (ROOT/'RESULTS.md').write_text('\n'.join(content)+'\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,5,figsize=(17,6),sharex=True)
        for task,ax in zip(cfg['tasks'],axes.flat):
            selected=[r for r in curves if r['task']==task]
            ax.step([r['evaluations'] for r in selected],[r['best_mic'] for r in selected],where='post',label='SemanticOPT')
            for arm in ['author_frozen','adapted_frozen','adapted_online']:
                values=fsbo_curves.get((task,arm))
                if values:ax.step([v[0] for v in values],[v[1] for v in values],where='post',label=arm)
            ax.axvline(1000,color='grey',ls=':',lw=1)
            ax.set_title(task);ax.set_ylabel('Best predicted MIC ↓');ax.set_xlabel('Fresh evaluations')
        axes.flat[0].legend(fontsize=7)
        fig.tight_layout()
        for suffix in ['png','pdf','svg']:fig.savefig(ROOT/f'comparison.{suffix}',dpi=160)
        plt.close(fig)
    except ImportError:
        (ROOT/'plot_unavailable.txt').write_text('Install matplotlib and rerun control/report.py to render plots.\n')


if __name__=='__main__':report()
