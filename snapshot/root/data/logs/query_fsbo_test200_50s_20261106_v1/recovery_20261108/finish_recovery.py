"""Wait for this single bounded batch, validate it, and return w19 to its queue."""
import json
import os
from pathlib import Path
import select
import sqlite3
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def atomic(path, value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(temp,path)


def main():
    ledger=json.loads((ROOT/'current_workers.json').read_text())
    worker=next(w for w in ledger['workers'] if w['name']=='w19')
    assert '--stop-after-batches' in worker['actual_command']
    report=HERE/'bounded_check_validation.json'
    atomic(HERE/'handoff_status.json',dict(status='waiting_for_one_batch',pilot_pid=worker['pid'],at=time.time()))
    process=Path('/proc')/str(worker['pid'])
    if process.exists():
        fields=(process/'stat').read_text().split()
        assert fields[21]==str(worker['process_start_ticks'])
        if fields[2]!='Z':
            deadline=time.monotonic()+4*3600
            while process.exists():
                fields=(process/'stat').read_text().split()
                if fields[2]=='Z' or fields[21]!=str(worker['process_start_ticks']):
                    break
                if time.monotonic()>deadline:
                    raise TimeoutError('Bounded recovery batch did not finish within four hours')
                time.sleep(10)
    state=json.loads((HERE/'w19.json').read_text())
    assert state['status']=='bounded_check_complete',state
    run=ROOT/'runs/CEB_7A102'
    status=json.loads((run/'status.json').read_text())
    assert status['status']=='paused' and status['proposals']==230
    assert status['unresolved_database_calls']==0 and status['error'] is None
    before=json.loads((HERE/'CEB_7A102_before.json').read_text())
    with sqlite3.connect(f'file:{run}/artifacts.sqlite3?mode=ro',uri=True) as connection:
        after=dict(connection.execute('SELECT name,sha256 FROM artifacts'))
    assert all(after.get(n)==h for n,h in before['all_artifact_hashes'].items())
    assert (run/'checkpoint.pt').stat().st_mtime>before['at']
    validation=dict(passed=True,at=time.time(),task='CEB_7A102',proposals_before=220,
        proposals_after=230,all_prior_artifact_hashes_unchanged=True,unresolved_database_calls=0,
        sql_timeout_seconds=50,inference_timeout_seconds=1800)
    atomic(report,validation)
    subprocess.run(['/root/data/cys/envs/bolt_clean_venv/bin/python','-B',str(HERE/'restart_workers.py'),
        '--host','LLMOptimizer4','--worker','w19'],check=True)
    atomic(HERE/'handoff_status.json',dict(status='full_queue_resumed',validation=str(report),at=time.time()))
    print(json.dumps(validation),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        atomic(HERE/'handoff_error.json',dict(error=repr(exc),at=time.time()))
        raise
