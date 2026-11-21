"""Read archived LOLBO histories by phase and snapshot available RGPE histories.

No model loading, decoder calls, oracle calls, or production writes.
"""
import collections
import csv
import hashlib
import json
from pathlib import Path
import tarfile

OUT = Path(__file__).resolve().parent
SUMMARY = Path('/root/reports/peptide_convergence_20261101/task_summary.csv')


def count_rows(rows):
    counts = collections.defaultdict(collections.Counter)
    for row in rows:
        c = counts[row['phase']]
        c['rows'] += 1
        c['feasible'] += bool(row['feasible'])
        c['fresh'] += not row['cached']
        c['fresh_feasible'] += bool(row['feasible']) and not row['cached']
        seq = row['sequence']
        c['structural_invalid'] += bool(not seq or len(seq) > 30 or
            set(seq) - set('ACDEFGHIKLMNPQRSTVWY') or seq.count('C') > 1)
    return dict(counts)


def main():
    total = collections.defaultdict(collections.Counter)
    tasks = []
    with SUMMARY.open() as f:
        records = [r for r in csv.DictReader(f) if r['method'] == 'LOLBO']
    for i, record in enumerate(records, 1):
        with tarfile.open(record['archive'], 'r|gz') as archive:
            for member in archive:
                if member.name == 'history.json':
                    payload = archive.extractfile(member).read()
                    break
            else:
                raise ValueError('Missing history')
        assert hashlib.sha256(payload).hexdigest() == record['history_sha256']
        history = json.loads(payload)
        counts = count_rows(history['observations'][100:])
        for phase, values in counts.items():
            total[phase].update(values)
        tasks.append(dict(task=record['task'], phases=counts))
        if i % 25 == 0:
            print(f'Checked {i}/{len(records)} LOLBO histories', flush=True)
    rgpe = collections.Counter()
    paths = sorted(Path('/root/data/logs/peptide_transfer_test200_20261028_v1/rgpe/runs').glob('*/history.json'))
    for path in paths:
        history = json.loads(path.read_text())
        for c in count_rows(history['observations'][100:]).values():
            rgpe.update(c)
    def rates(c):
        return dict(c, feasible_percent=100*c['feasible']/c['rows'] if c['rows'] else None,
                    fresh_feasible_percent=100*c['fresh_feasible']/c['fresh'] if c['fresh'] else None)
    output = dict(lolbo_tasks=len(tasks), lolbo_phases={k:rates(v) for k,v in total.items()},
                  rgpe_partial_snapshot=dict(tasks=len(paths), **rates(rgpe)),
                  checks='LOLBO history SHA256 checked against prior full archive audit; feasibility flags read as recorded; alphabet/length/cysteine checked',
                  oracle_calls=0, decoder_calls=0, tasks=tasks)
    (OUT/'phase_audit.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k != 'tasks'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
