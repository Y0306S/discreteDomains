"""Launch only validated local additions to the existing frozen query FSBO campaign."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PYTHON = '/root/data/cys/envs/bolt_clean_venv/bin/python'


def alive(entry):
    proc = Path('/proc')/str(entry['pid'])
    try:
        return str(ROOT/'run_worker.py').encode() in (proc/'cmdline').read_bytes() and (proc/'stat').read_text().split()[2] != 'Z'
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-batch', action='store_true')
    args = parser.parse_args()
    with (ROOT/'launch.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'LAUNCH_HOLD').exists():
            raise RuntimeError('Launch held by user; see LAUNCH_HOLD. No new workers started.')
        plan = json.loads((ROOT/'plan.json').read_text())
        reserve = json.loads((ROOT/'reservation.json').read_text())
        pilot = json.loads((ROOT/'pilot_LLMOptimizer2.json').read_text())
        assert pilot['status'] == 'complete'
        cg = Path('/sys/fs/cgroup')
        quota,period = map(int,(cg/'cpu.max').read_text().split())
        assert quota/period >= reserve['orpo_cpu_cores']+reserve['existing_overhead_cpu_cores']+reserve['query_worker_cpu_cores']
        free = int((cg/'memory.max').read_text())-int((cg/'memory.current').read_text())
        assert free/2**30 >= reserve['orpo_memory_gib']+reserve['headroom_gib']+8
        campaign = Path(reserve['campaign'])
        assert hashlib.sha256((campaign/'recipe.json').read_bytes()).hexdigest() == reserve['recipe_sha256']
        if not args.pilot_batch:
            validation=json.loads((ROOT/'batch_validation.json').read_text())
            assert validation['passed']
        record = ROOT/('batch_process.json' if args.pilot_batch else 'production_processes.json')
        if record.exists():
            raise RuntimeError('Launch record exists; inspect it before restarting')
        if not args.pilot_batch:
            current_workers=json.loads((campaign/'current_workers.json').read_text())
            if set(w['name'] for w in current_workers['workers']) & set(w['name'] for w in plan['workers']):
                raise RuntimeError('Local worker names already registered')
        if args.pilot_batch:
            selected=plan['workers'][:1]
        else:
            previous=json.loads((ROOT/'batch_process.json').read_text())
            if any(alive(e) for e in previous['workers']):
                raise RuntimeError('Bounded batch pilot is still running')
            selected=plan['workers']
        launched=[]
        for w in selected:
            command=['taskset','-c',str(w['cpus'][0]),PYTHON,'-u',str(ROOT/'run_worker.py'),'--worker',w['name']]
            if args.pilot_batch:
                command+=['--stop-after-batches','1']
            log=ROOT/'workers'/w['name']/('batch.log' if args.pilot_batch else 'production.log')
            with log.open('a') as out:
                process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            launched.append(dict(w,pid=process.pid,actual_command=command,command=command,log=str(log),started_at=time.time(),process_start_ticks=Path(f'/proc/{process.pid}/stat').read_text().split()[21]))
        value=dict(at=time.time(),workers=launched,host='LLMOptimizer2',recipe_sha256=reserve['recipe_sha256'])
        record.write_text(json.dumps(value,indent=2)+'\n')
        if not args.pilot_batch:
            current=campaign/'current_workers.json'
            old=json.loads(current.read_text())
            if set(w['name'] for w in old['workers']) & set(w['name'] for w in launched):
                raise RuntimeError('Unexpected existing local worker names')
            backup=ROOT/'current_workers.before_expansion.json';backup.write_text(json.dumps(old,indent=2)+'\n')
            old['workers'].extend(launched);old.update(at=time.time(),operational_expansion=str(ROOT))
            temp=current.with_suffix('.expansion.tmp');temp.write_text(json.dumps(old,indent=2)+'\n');os.replace(temp,current)
        print(json.dumps(value,indent=2))


if __name__=='__main__':
    main()
