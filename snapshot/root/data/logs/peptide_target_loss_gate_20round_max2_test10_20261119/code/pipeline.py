"""Durable four-worker launch of the same-ten peptide target-gate experiment."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/root/data/cys/envs/bolt_clean_venv/bin/python'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); tmp.replace(path)


def process_identity(pid):
    try:
        path=Path('/proc')/str(pid)
        stat=path.joinpath('stat').read_text().split()
        if stat[2] in ('Z','T'):return None
        return dict(pid=pid,start_ticks=stat[21],argv=[v.decode() for v in path.joinpath('cmdline').read_bytes().split(b'\0') if v])
    except OSError:return None


def verify_manifest():
    for name,digest in read(ROOT/'code_manifest.json').items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest, 'Dependency changed: '+name


def status(**values):
    write(ROOT/'pipeline_status.json',dict(updated_at=time.time(),host=socket.gethostname(),
        **process_identity(os.getpid()),**values))


def launch_worker(slot, tasks):
    policy=read(ROOT/'execution_policy.json')
    for task in tasks:
        result=ROOT/'evaluation/adapted_online'/task/'result.json'
        if result.exists() and read(result)['status']=='complete':continue
        verify_manifest()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(slot['gpu']),OMP_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONWARNINGS='ignore',PYTHONDONTWRITEBYTECODE='1')
        # A previous controller can leave a healthy evaluator. Adopt it, using
        # exact argv matching, instead of launching a duplicate evaluation.
        adopted=None
        for path in Path('/proc').glob('[0-9]*'):
            identity=process_identity(int(path.name))
            if not identity:continue
            argv=identity['argv']
            if str(ROOT/'code/evaluate.py') in argv and '--task' in argv and argv[argv.index('--task')+1]==task and '--max-events' not in argv:
                adopted=identity;break
        started=time.time()
        if adopted:
            write(ROOT/'workers'/f'slot{slot["slot"]}.json',dict(status='adopted',task=task,slot=slot,**adopted))
            while process_identity(adopted['pid'])==adopted:time.sleep(10)
            if not result.exists() or read(result)['status']!='complete':
                raise RuntimeError('Adopted evaluator exited without completion: '+task)
            continue
        argv=['taskset','-c',','.join(map(str,slot['cpus'])),PYTHON,'-u',str(ROOT/'code/evaluate.py'),
              '--task',task,'--arm','adapted_online']
        with (ROOT/('eval_'+task+'.log')).open('ab') as log:
            process=subprocess.Popen(argv,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            identity=process_identity(process.pid)
            write(ROOT/'workers'/f'slot{slot["slot"]}.json',dict(status='running',task=task,slot=slot,
                started_at=started,**identity))
            code=process.wait()
        write(ROOT/'workers'/f'slot{slot["slot"]}.json',dict(status='complete' if code==0 else 'failed',
            task=task,slot=slot,started_at=started,finished_at=time.time(),returncode=code,**identity))
        if code or not result.exists() or read(result)['status']!='complete':
            raise RuntimeError(f'{task} failed (exit {code}); inspect eval_{task}.log')
        write(ROOT/'timings'/(task+'.json'),dict(seconds=time.time()-started,gpu=slot['gpu'],cpus=slot['cpus']))


def main():
    with (ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_manifest()
        cfg=read(ROOT/'config.json'); execution=read(ROOT/'execution_policy.json')
        assert len(cfg['tasks'])==10 and cfg['batch_size']==50 and cfg['proposals']==cfg['fresh_budget']==4000
        assert cfg['stagnation_rounds']==20 and cfg['max_retrain_attempts']==2
        assert read(ROOT/'validation/retrain_schedule.json')['passed']
        assert read(ROOT/'validation/scheduling_gpu.json')['passed']
        assert read(ROOT/'validation/live_gpu.json')['passed']
        assert read(ROOT/'validation/saved_pep999_gate.json')['passed']
        workers=execution['workers']; order=execution.get('task_order',cfg['tasks'])
        assert set(order)==set(cfg['tasks']) and len(order)==len(cfg['tasks'])
        assignments=[order[i::len(workers)] for i in range(len(workers))]
        status(status='running',stage='same10_target_loss_rollback_evaluation',assignments=assignments,
            reused_offline_assets=True)
        errors=[]
        with ThreadPoolExecutor(max_workers=len(workers)) as pool:
            futures={pool.submit(launch_worker,slot,tasks):slot for slot,tasks in zip(workers,assignments)}
            for future in as_completed(futures):
                try:future.result()
                except Exception:
                    errors.append(dict(slot=futures[future],error=traceback.format_exc()))
                    write(ROOT/'worker_errors.json',errors)
        if errors:raise RuntimeError('Worker failures; see worker_errors.json')
        status(status='running',stage='paired_comparison_report')
        verify_manifest()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        with (ROOT/'report.log').open('ab') as log:
            subprocess.run([PYTHON,str(ROOT/'code/report.py')],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        status(status='complete',stage='complete',results=str(ROOT/'RESULTS.md'))


if __name__=='__main__':
    try:main()
    except BaseException:
        status(status='failed',stage='failed',error=traceback.format_exc())
        raise
