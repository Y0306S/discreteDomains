"""Durable staged experiment driver; run with the system Python, no GPU allocation."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
MODERN='/root/data/cys/envs/bolt_clean_venv/bin/python'
EXACT='/root/data/cys/baseline/envs/bl_fsbo_exact/bin/python'


def read(path):
    return json.loads(Path(path).read_text())


def write(value):
    value.update(updated_at=time.time(),pid=os.getpid())
    temporary=ROOT/'pipeline_status.json.tmp'
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(ROOT/'pipeline_status.json')


def command(stage,script,args=(),gpu=None,python=MODERN):
    import hashlib
    pinned=read(ROOT/'code_manifest.json')
    for path,expected in pinned.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
            raise ValueError('Experiment dependency changed: '+path)
    started=time.monotonic()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONWARNINGS='ignore',PYTHONDONTWRITEBYTECODE='1')
    with (ROOT/(stage+'.log')).open('ab') as output:
        subprocess.run([python,'-u',str(ROOT/'code'/script),*map(str,args)],env=env,cwd=ROOT,
            stdout=output,stderr=subprocess.STDOUT,check=True)
    (ROOT/'timings').mkdir(exist_ok=True)
    (ROOT/'timings'/(stage+'.json')).write_text(json.dumps(dict(elapsed_seconds=time.monotonic()-started,gpu=gpu,python=python)))


def main():
    import fcntl
    with (ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg=read(ROOT/'config.json')
        policy=read(ROOT/'run_policy.json') if (ROOT/'run_policy.json').exists() else {}
        active_arms=policy.get('active_arms',cfg['arms'])
        assert active_arms and len(set(active_arms))==len(active_arms)
        assert set(active_arms)<=set(cfg['arms'])
        validation=read(ROOT/'validation.json')
        assert validation['passed'], 'Real VAE/FSBO/recovery validation must pass before the driver starts'
        import hashlib
        for name in ('common.py','offline.py','evaluate.py','train_fsbo.py'):
            path=ROOT/'code'/name
            assert hashlib.sha256(path.read_bytes()).hexdigest()==validation['code_sha256'][str(path)]
        write(dict(status='running',stage='offline_vae_training'))
        process=read(ROOT/'vae_process.json')['pid']
        while not (ROOT/'vae/status.json').exists() or read(ROOT/'vae/status.json')['status']!='complete':
            try:
                stat=Path('/proc')/str(process)/'stat'
                if not stat.exists() or stat.read_text().split()[2]=='Z':
                    command('vae_resume','offline.py',['train'],gpu=0)
            except FileNotFoundError:
                command('vae_resume','offline.py',['train'],gpu=0)
            progress=read(ROOT/'vae/status.json') if (ROOT/'vae/status.json').exists() else {}
            write(dict(status='running',stage='offline_vae_training',progress=progress))
            time.sleep(30)
        write(dict(status='running',stage='refresh_train800_embeddings'))
        if not (ROOT/'corpus/manifest.json').exists():
            command('encoding','offline.py',['export'],gpu=0)
        write(dict(status='running',stage='fsbo_meta_training_100k_steps'))
        if not (ROOT/'fsbo/status.json').exists() or read(ROOT/'fsbo/status.json')['status']!='complete':
            command('fsbo_meta','train_fsbo.py',python=EXACT)
        write(dict(status='running',stage=f'heldout_evaluation_{len(cfg["tasks"])}_tasks_{len(active_arms)}_arms',active_arms=active_arms))
        assignments=[cfg['tasks'][i::4] for i in range(4)]
        def worker(gpu,tasks):
            for task in tasks:
                index=cfg['tasks'].index(task)
                # Rotate order to balance temporal load and checkpoint warming.
                offset=index%len(active_arms)
                arms=active_arms[offset:]+active_arms[:offset]
                for arm in arms:
                    result=ROOT/'evaluation'/arm/task/'result.json'
                    # Adopt retained evaluators after a controller-only restart.
                    while True:
                        running=False
                        for process in Path('/proc').glob('[0-9]*'):
                            try:
                                argv=(process/'cmdline').read_bytes().split(b'\0')
                                if (str(ROOT/'code/evaluate.py').encode() in argv and
                                        b'--task' in argv and b'--arm' in argv and
                                        argv[argv.index(b'--task')+1]==task.encode() and
                                        argv[argv.index(b'--arm')+1]==arm.encode()):
                                    running=True
                                    break
                            except (OSError,IndexError):
                                continue
                        if not running:
                            break
                        time.sleep(5)
                    if result.exists() and read(result)['status']=='complete':
                        continue
                    command('eval_'+task+'_'+arm,'evaluate.py',['--task',task,'--arm',arm],gpu=gpu)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(worker,gpu,tasks) for gpu,tasks in enumerate(assignments)]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        write(dict(status='running',stage='audit_and_report'))
        command('report','report.py',['report'])
        assert read(ROOT/'report_status.json')['status']=='complete'
        write(dict(status='complete',stage='complete',results=str(ROOT/'RESULTS.md')))


if __name__=='__main__':
    try:
        main()
    except BaseException:
        write(dict(status='failed',error=traceback.format_exc()))
        raise
