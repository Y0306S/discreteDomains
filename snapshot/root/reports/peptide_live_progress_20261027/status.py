#!/usr/bin/env python3
"""Read current peptide progress and audit completed histories without changing runs."""

import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile
import time

from Levenshtein import distance

ROOT = Path(__file__).resolve().parent
CAMPAIGN = Path('/root/data/logs/peptide_test200_baselines_20261027_v2')
TRANSFER = Path('/root/data/logs/peptide_transfer_train800_20261027_v2')


def read(path):
    return json.loads(path.read_text())


def main():
    result = dict(checked_at=time.time(), methods={})
    for method in ('tpe', 'lolbo'):
        folder = CAMPAIGN / method
        recipe = read(folder / 'recipe.json')
        recipe_hash = hashlib.sha256((folder / 'recipe.json').read_bytes()).hexdigest()
        completed, active, issues = [], [], []
        archives = {p.name[:-7]: p for p in (folder / 'runs').glob('*.tar.gz')}
        for task, path in sorted(archives.items()):
            try:
                with tarfile.open(path, 'r:gz') as packed:
                    manifest = json.load(packed.extractfile('manifest.json'))
                    raw = packed.extractfile('history.json').read()
                history = json.loads(raw)
                identity = history['identity']
                assert identity == manifest['identity']
                assert identity['recipe_sha256'] == recipe_hash and identity['task'] == task
                assert hashlib.sha256(raw).hexdigest() == manifest['members']['history.json']
                assert task in recipe['test'] and history['status'] == 'complete'
                assert history['reference'] == recipe['references'][task]
                assert history['initialization'] == 100
                rows = history['observations']
                assert len(rows) == 100 + history['proposals'] + history['recenter_proposals']
                assert history['proposals'] <= 5000 and history['fresh_evaluations'] <= 5000
                assert history['proposals'] == 5000 or history['fresh_evaluations'] == 5000
                assert history['fresh_evaluations'] == history['acquisition_fresh'] + history['recenter_fresh']
                assert history['fresh_evaluations'] + history['cache_hits'] == history['proposals'] + history['recenter_proposals']
                feasible = []
                for row in rows:
                    sequence = row['sequence']
                    assert sequence and len(sequence) <= 30 and sequence.count('C') <= 1
                    assert not set(sequence) - set('ACDEFGHIKLMNPQRSTVWY')
                    assert math.isfinite(row['score']) and row['score'] < 0
                    assert math.isclose(row['mic'], -row['score'], abs_tol=1e-5, rel_tol=1e-6)
                    allowed = 1 - distance(sequence, history['reference']) / len(history['reference']) >= 0.75 - 1e-8
                    assert row['feasible'] == allowed
                    if allowed:
                        feasible.append(row['mic'])
                assert math.isclose(min(feasible), history['best_feasible_mic'], abs_tol=1e-5, rel_tol=1e-6)
                initial = min(r['mic'] for r in rows[:100])
                history['relative_improvement'] = (initial - history['best_feasible_mic']) / initial
                completed.append({k:history[k] for k in ('proposals','fresh_evaluations','cache_hits',
                    'recenter_fresh','best_feasible_mic','relative_improvement','seconds')})
            except Exception as exc:
                issues.append(dict(task=task, error=repr(exc)))
        for path in (folder / 'runs').glob('*/progress.json'):
            if path.parent.name not in archives:
                value = read(path)
                value['seconds_since_progress_file'] = time.time() - path.stat().st_mtime
                value['task'] = path.parent.name
                active.append(value)
        all_rows = completed + active
        result['methods'][method] = dict(completed=len(archives), audited_completed=len(completed),
            total=200, active=len(active), issues=issues,
            proposals=sum(r['proposals'] for r in all_rows),
            fresh_evaluations=sum(r['fresh_evaluations'] for r in all_rows),
            budget_progress_percent=100*(len(archives) + sum(max(r['proposals'],r['fresh_evaluations'])/5000 for r in active))/200,
            improved_completed=sum(r['relative_improvement']>1e-6 for r in completed),
            median_completed_relative_improvement=statistics.median(r['relative_improvement'] for r in completed) if completed else None,
            median_completed_seconds=statistics.median(r['seconds'] for r in completed) if completed else None,
            median_active_proposals=statistics.median(r['proposals'] for r in active) if active else None,
            oldest_active_progress_seconds=max((r['seconds_since_progress_file'] for r in active),default=0),
            active_tasks=[dict(task=r['task'],proposals=r['proposals'],fresh_evaluations=r['fresh_evaluations'],
                age_seconds=r['seconds_since_progress_file']) for r in active])
    progress = read(TRANSFER / 'corpus/status.json')
    logs = []
    for line in (TRANSFER / 'export.log').read_text().splitlines():
        try:
            value=json.loads(line)
            if value.get('status')=='exporting':logs.append(value)
        except ValueError:
            pass
    if len(logs)>20:
        delta=logs[-1]['elapsed_seconds']-logs[-21]['elapsed_seconds']
        progress['recent_seconds_per_task']=delta/20
        progress['remaining_hours_at_recent_pace']=(800-progress['completed'])*delta/20/3600
    result['transfer']=progress
    temporary=ROOT/'status.json.tmp'
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False))
    temporary.replace(ROOT/'status.json')
    concise=dict(result,methods={method:{k:v for k,v in values.items() if k!='active_tasks'} for method,values in result['methods'].items()})
    print(json.dumps(concise),flush=True)


if __name__=='__main__':
    main()
