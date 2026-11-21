"""Summarize durable task progress and check the recorded worker processes."""
import argparse
import json
import os
from pathlib import Path
import time

from resources import snapshot

CAMPAIGN = Path('/root/data/logs/peptide_bo_train800_20261027_v5b')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True)
    args = parser.parse_args()
    workers = json.loads((CAMPAIGN / 'workers' / (args.host + '.json')).read_text())
    live = []
    errors = []
    assignments = []
    for row in workers:
        try:
            command = Path(f'/proc/{row["pid"]}/cmdline').read_bytes().decode().replace('\0', ' ')
            assert 'peptide_bo_v5.py' in command and str(CAMPAIGN) in command
            assert os.sched_getaffinity(row['pid']) == {row['cpu']}
            live.append(row['local_index'])
            children = Path(f'/proc/{row["pid"]}/task/{row["pid"]}/children').read_text().split()
            for child in children:
                child_command = Path(f'/proc/{child}/cmdline').read_bytes().decode().split('\0')
                if '--task' in child_command:
                    assert os.sched_getaffinity(int(child)) == {row['cpu']}
                    assignments.append(dict(worker=row['local_index'], cpu=row['cpu'], pid=int(child),
                        task=child_command[child_command.index('--task') + 1]))
        except (OSError, AssertionError):
            errors.append(dict(worker=row['local_index'], problem='missing or incorrect affinity'))
        with Path(row['log']).open('rb') as stream:
            stream.seek(row.get('resume_log_offset', 0))
            log = stream.read().decode(errors='replace')
        if 'Traceback' in log or '"task_error"' in log:
            errors.append(dict(worker=row['local_index'], problem='error in worker log'))
    tasks = []
    for path in (CAMPAIGN / 'runs').glob('*/progress.json'):
        try:
            value = json.loads(path.read_text())
            tasks.append(dict(task=value['identity']['task'], rounds=value['rounds'],
                proposals=value['proposals'], fresh_evaluations=value['fresh_evaluations'],
                e2e_updates=value['e2e_updates'], age_seconds=time.time()-path.stat().st_mtime))
        except OSError:
            pass
    resources = snapshot()
    resources.pop('processes')
    report = dict(host=args.host, checked_at=time.time(), live_workers=len(live), workers=live,
        assignments=assignments, errors=errors,
        started_tasks=len(tasks), completed_tasks=len(list((CAMPAIGN / 'runs').glob('*.tar.gz'))),
        running_acquisition_proposals=sum(value['proposals'] for value in tasks),
        running_fresh_evaluations=sum(value['fresh_evaluations'] for value in tasks),
        resources=resources, tasks=sorted(tasks, key=lambda value:value['task']))
    temporary = CAMPAIGN / 'workers' / (args.host + '.status.json.tmp')
    temporary.write_text(json.dumps(report))
    os.replace(temporary, temporary.with_suffix(''))
    print(json.dumps(report))

if __name__ == '__main__':
    main()
