"""Summarize isolated, fixed-candidate transfer tests; never changes a campaign."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
SRC = Path('/root/data/cys/tmp/query_vae_control_20261121')
NEW = Path('/root/data/cys/baseline/campaigns/query_our_fsbo_vae80_test200_20261117')
NAMES = ['original80', 'retrained80', 'original788']


def summarize(rows):
    rho = [r['spearman'] for r in rows if r['spearman'] is not None]
    return dict(tasks=len(rows), mean_spearman=float(np.mean(rho)),
                median_spearman=float(np.median(rho)),
                mean_normalized_rmse=float(np.mean([r['rmse_normalized'] for r in rows])),
                mean_predictive_nll=float(np.mean([r['predictive_nll'] for r in rows])),
                geomean_selection_regret_ratio=float(np.exp(np.mean(np.log([
                    r['selection_regret_ratio'] for r in rows])))))


def pair(rows, numerator, denominator):
    a = {r['task']: r for r in rows if r['model'] == numerator}
    b = {r['task']: r for r in rows if r['model'] == denominator}
    tasks = sorted(a.keys() & b.keys())
    assert len(tasks) == 15
    ratios = np.array([a[t]['selected_query_runtime'] / b[t]['selected_query_runtime'] for t in tasks])
    rng = np.random.default_rng(123)
    interval = np.exp(np.quantile(rng.choice(np.log(ratios), (10000, len(tasks))).mean(axis=1), [.025, .975]))
    return dict(tasks=len(tasks), numerator=numerator, denominator=denominator,
                geomean_selected_runtime_ratio=float(np.exp(np.log(ratios).mean())),
                task_bootstrap_95pct_ratio_interval=interval.tolist(),
                wins_over_5pct=int((ratios < .95).sum()), losses_over_5pct=int((ratios > 1.05).sum()),
                within_5pct=int(((ratios >= .95) & (ratios <= 1.05)).sum()),
                mean_spearman_difference=float(np.mean([a[t]['spearman'] - b[t]['spearman'] for t in tasks])))


def main():
    adapted = dict(models={}, per_task=[], status='complete')
    for name in NAMES:
        for shard in range(2):
            p = SRC / f'target_probe_{name}_{shard}.json'
            d = json.loads(p.read_text())
            assert d.get('status') == 'complete', str(p)
            adapted['models'].update(d['models'])
            adapted['per_task'].extend(d['per_task'])
    assert len(adapted['per_task']) == 45
    assert len({(r['task'], r['model']) for r in adapted['per_task']}) == 45
    for name in NAMES:
        rows = [r for r in adapted['per_task'] if r['model'] == name]
        assert len(rows) == 15
        for r in rows:
            assert all(np.isfinite(r[k]) for k in ['spearman', 'selection_regret_ratio', 'predictive_nll', 'rmse_normalized'])
        adapted['models'][name]['summary'] = summarize(rows)
    (OUT / 'target_probe_adapted.json').write_text(json.dumps(adapted, indent=2))
    unadapted = json.loads((SRC / 'target_probe_no_adaptation.json').read_text())
    assert unadapted['status'] == 'complete' and len(unadapted['per_task']) == 45
    checks = {}
    target_tasks = {r['task'] for r in adapted['per_task']}
    for name, mp in [('original80', SRC / 'original/corpus/manifest.json'),
                     ('retrained80', NEW / 'training/corpus/manifest.json'),
                     ('original788', Path('/root/data/logs/query_transfer_train800_20261106_v1/corpus/manifest.json'))]:
        source = set(json.loads(mp.read_text())['source_tasks'])
        overlap = sorted(source & target_tasks)
        assert not overlap
        checks[name] = dict(source_tasks=len(source), target_task_overlap=overlap)
    result = dict(status='complete', oracle_calls=0, production_promoted=False,
                  training_original80=json.loads((SRC / 'original/fsbo/status.json').read_text()),
                  training_retrained80=json.loads((NEW / 'training/fsbo/status.json').read_text()),
                  source_disjointness=checks,
                  preparation=json.loads((SRC / 'matched80_preparation.json').read_text()),
                  target_manifest=json.loads((SRC / 'target_probe/manifest.json').read_text()),
                  after_native_target_adaptation=adapted, without_target_gradient_updates=unadapted,
                  paired_comparisons={})
    for stage, data in [('after_adaptation', adapted), ('before_adaptation', unadapted)]:
        result['paired_comparisons'][stage] = {
            f'{a}_over_{b}': pair(data['per_task'], a, b)
            for a, b in [('retrained80', 'original80'), ('retrained80', 'original788'), ('original80', 'original788')]}
    result['adaptation_effect'] = {}
    for name in NAMES:
        tagged = [dict(r, model='after') for r in adapted['per_task'] if r['model'] == name]
        tagged += [dict(r, model='before') for r in unadapted['per_task'] if r['model'] == name]
        result['adaptation_effect'][name] = pair(tagged, 'after', 'before')
    result['limitations'] = [
        '15 held-out queries spanning 15 templates, not held-out templates; one meta-training seed.',
        'Ranking saved BOLT30 plans, not a new SQL or Bayesian optimization convergence experiment.',
        'Small support sets contain up to 10 observations; production fitting later uses more data.',
        'Models rank identical candidate outcomes; historical runtime and warm-pool coverage still limit generalization.',
        'These test-query results must not select production checkpoints or tune the models.',
        'Source minmax normalization uses all 80 source tasks, including its 16 validation tasks, matching production policy.'
    ]
    evidence = [SRC / 'matched80.py', SRC / 'target_probe.py', SRC / 'target_probe_shard.py',
                SRC / 'target_probe_no_adaptation.py', SRC / 'run_shards.py', SRC / 'target_probe/manifest.json']
    result['code_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence}
    for p in evidence:
        if p.suffix == '.py':
            shutil.copyfile(p, OUT / p.name)
    shutil.copyfile(SRC / 'target_probe_no_adaptation.json', OUT / 'target_probe_no_adaptation.json')
    (OUT / 'controlled_results.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(dict(summaries={n: adapted['models'][n]['summary'] for n in NAMES},
                          paired=result['paired_comparisons']), indent=2))


if __name__ == '__main__':
    main()
