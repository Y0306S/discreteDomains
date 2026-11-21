"""Check saved FSBO observations without loading or modifying live model states."""

import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import statistics
import time

from Levenshtein import distance

ROOT = Path(__file__).resolve().parent
CAMPAIGN = Path('/root/data/logs/peptide_transfer_test200_20261028_v1')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            result.update(chunk)
    return result.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def main():
    recipe_path = CAMPAIGN / 'fsbo/recipe.json'
    recipe = json.loads(recipe_path.read_text())
    identity = digest(recipe_path)
    errors = []
    if set(recipe['train']) & set(recipe['test']):
        errors.append('Source/test task overlap')
    changed = [path for path, expected in recipe['code'].items() if digest(Path(path)) != expected]
    errors.extend('Pinned code changed: ' + path for path in changed)
    tasks, fits = [], []
    for path in sorted((CAMPAIGN / 'fsbo/runs').glob('*/history.json')):
        history = json.loads(path.read_text())
        task, rows = history['task'], history['observations']
        issues = []
        if task not in recipe['test'] or history['identity']['recipe_sha256'] != identity:
            issues.append('Wrong task or recipe identity')
        acquired = [r for r in rows if r['phase'] == 'acquisition']
        initial = [r for r in rows if r['phase'] == 'initialization']
        fresh = [r for r in acquired if not r['cached']]
        if (len(initial) != recipe['initialization'] or len(acquired) != history['proposals']
                or len(fresh) != history['fresh_evaluations']
                or len(acquired) - len(fresh) != history['cache_hits']
                or history['proposals'] > recipe['proposals']
                or history['fresh_evaluations'] > recipe['fresh_budget']):
            issues.append('Proposal/fresh/cache accounting mismatch')
        known = {}
        for row in rows:
            sequence, score = row['sequence'], row['score']
            if not math.isfinite(score) or row['mic'] != -score:
                issues.append('Nonfinite or reversed objective')
            if (not sequence or len(sequence) > 30 or sequence.count('C') > 1
                    or set(sequence) - set('ACDEFGHIKLMNPQRSTVWY')):
                issues.append('Invalid decoder peptide')
            feasible = 1.0 - distance(sequence, history['reference']) / len(history['reference']) >= recipe['similarity']
            if bool(row['feasible']) != feasible:
                issues.append('Recorded feasibility differs from actual constraint')
            if sequence in known and known[sequence] != score:
                issues.append('Cached sequence changed score')
            if row['phase'] == 'acquisition' and row['cached'] != (sequence in known):
                issues.append('Cache flag does not match earlier observations')
            known[sequence] = score
            if row.get('fit'):
                fit = row['fit']
                if fit['checkpoint_sha256'] != recipe['fsbo_weights_sha256']:
                    issues.append('Wrong FSBO parent checkpoint')
                fits.append(fit)
        best = min(r['mic'] for r in rows if r['feasible'])
        if best != history['best_feasible_mic']:
            issues.append('Incorrect feasible incumbent')
        errors.extend(task + ': ' + issue for issue in set(issues))
        task_fits = [r['fit'] for r in acquired if r.get('fit')]
        tasks.append(dict(task=task, proposals=history['proposals'], fresh=len(fresh),
            fresh_feasible=sum(r['feasible'] for r in fresh),
            improved=best < min(r['mic'] for r in initial),
            cache_hits=history['cache_hits'], recorded_seconds=history['seconds'],
            latest_fit=task_fits[-1], history_age_seconds=time.time() - path.stat().st_mtime,
            issues=sorted(set(issues))))
    total_fresh = sum(t['fresh'] for t in tasks)
    total_seconds = sum(t['recorded_seconds'] for t in tasks)
    total_fit = sum(f['seconds'] for f in fits)
    with sqlite3.connect('file:/root/data/logs/peptide_transfer_train800_20261027_v2/rgpe/models.sqlite?mode=ro',
                         uri=True, timeout=10) as db:
        metadata = [json.loads(row[0]) for row in db.execute('SELECT metadata FROM models')]
    unverified = [r['task'] for r in metadata if not r['reload_verified']]
    errors.extend('Unverified source GP: ' + task for task in unverified)
    result = dict(checked=time.time(), completed_fsbo=len(list((CAMPAIGN / 'fsbo/runs').glob('*.tar.gz'))),
        active_tasks=len(tasks), total_proposals=sum(t['proposals'] for t in tasks),
        fresh_evaluations=total_fresh, fresh_feasible=sum(t['fresh_feasible'] for t in tasks),
        fresh_feasible_percent=100 * sum(t['fresh_feasible'] for t in tasks) / total_fresh,
        tasks_without_feasible_new=sum(t['fresh_feasible'] == 0 for t in tasks),
        improved_tasks=sum(t['improved'] for t in tasks), fit_rounds=len(fits),
        fits_at_cap=sum(f['adaptation_steps'] == recipe['fsbo_target_epochs'] for f in fits),
        latest_fit_median_seconds=statistics.median(t['latest_fit']['seconds'] for t in tasks),
        recorded_fit_fraction=total_fit / total_seconds,
        rgpe_complete=len(metadata), code_hashes_match=not changed, integrity_errors=errors, tasks=tasks)
    write(ROOT / 'audit.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'tasks'}))


if __name__ == '__main__':
    main()
