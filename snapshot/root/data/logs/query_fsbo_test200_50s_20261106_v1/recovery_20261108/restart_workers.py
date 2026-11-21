"""One-shot restart of stopped query evaluators; preserve active processes."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(temporary, path)


def live(worker):
    process = Path('/proc')/str(worker['pid'])
    try:
        fields = (process/'stat').read_text().split()
        return fields[2]!='Z' and fields[21]==str(worker['process_start_ticks'])
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', choices=['LLMOptimizer2','LLMOptimizer4'], required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--worker')
    args = parser.parse_args()
    assert (socket.gethostname()=='cs-94bac-0693e-server')==(args.host=='LLMOptimizer2')
    evidence = json.loads((HERE/'CEB_7A102_before.json').read_text())
    assert evidence['passed'] and evidence['pending_database_proposals']==0
    assert hashlib.sha256((ROOT/'recipe.json').read_bytes()).hexdigest()==evidence['recipe_sha256']
    assert not (ROOT/'STOP').exists()
    with (HERE/'restart.lock').open('a') as launch_lock:
        fcntl.flock(launch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger=json.loads((ROOT/'current_workers.json').read_text())
        restarted=[]
        for worker in ledger['workers']:
            if worker['host']!=args.host or (args.pilot and worker['name']!='w19'):
                continue
            if args.worker and worker['name']!=args.worker:
                continue
            if live(worker):
                continue
            endpoint=f"127.0.0.1:{worker['db_port']}:imdb"
            path=Path('/tmp')/('ceb_baseline_db_'+hashlib.sha256(endpoint.encode()).hexdigest()[:16]+'.lock')
            with path.open('a') as handle:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            env=dict(os.environ,PGPASSWORD='imdb',PGCONNECT_TIMEOUT='5',PGOPTIONS='-c statement_timeout=5000')
            command=['/root/data/cys/envs/pg16/bin/psql','-h','127.0.0.1','-p',str(worker['db_port']),
                     '-U','imdb','-d','imdb','-X','-A','-t','-v','ON_ERROR_STOP=1','-c',
                     "SELECT count(*) FROM pg_stat_activity WHERE datname='imdb' AND pid<>pg_backend_pid(); SHOW temp_tablespaces; SELECT 1;"]
            result=subprocess.run(command,env=env,text=True,capture_output=True,timeout=10,check=True)
            assert result.stdout.splitlines()==['0','fsbo_tmp_'+worker['name'],'1'], result.stdout
            cg=Path('/sys/fs/cgroup')
            free=int((cg/'memory.max').read_text())-int((cg/'memory.current').read_text())
            minimum=232 if args.host=='LLMOptimizer2' else 32
            assert free>=minimum*2**30, f'Memory headroom unavailable: {free/2**30}'
            command=['taskset','-c',str(worker['cpus'][0]),'/root/data/cys/envs/bolt_clean_venv/bin/python',
                     '-u','-B',str(HERE/'resilient_worker.py'),'--worker',worker['name']]
            if args.pilot:
                command+=['--task','CEB_7A102','--stop-after-batches','1']
            log=HERE/(worker['name']+('.pilot' if args.pilot else '.resume')+f'.{int(time.time())}.log')
            with log.open('a') as output:
                process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
                    start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',
                    MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
            old=dict(worker)
            worker.update(previous_pid=worker['pid'],pid=process.pid,command=command,actual_command=command,
                log=str(log),started_at=time.time(),
                process_start_ticks=Path(f'/proc/{process.pid}/stat').read_text().split()[21],
                recovery_evidence=str(HERE/'CEB_7A102_before.json'))
            restarted.append(dict(before=old,after=dict(worker)))
            atomic(ROOT/'current_workers.json',ledger)
            atomic(HERE/(args.host+('.'+args.worker if args.worker else '')+('.pilot' if args.pilot else '.restart')+'.json'),
                   dict(at=time.time(),workers=restarted))
        print(json.dumps(dict(host=args.host,restarted=[dict(worker=x['after']['name'],pid=x['after']['pid']) for x in restarted])))


if __name__=='__main__':
    main()
