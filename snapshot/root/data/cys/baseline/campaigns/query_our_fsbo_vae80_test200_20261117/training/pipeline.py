"""Durable sequential retraining and promotion; stages are restartable."""
import fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'training'))
from train import atomic,load,digest
VAE_PY='/root/data/cys/envs/bolt_clean_venv/bin/python'
FSBO_PY='/root/data/cys/baseline/envs/bl_fsbo_exact/bin/python'
def main():
    lock=(ROOT/'training/pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    record=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19],at=time.time())
    atomic(ROOT/'training/pipeline.process.json',record)
    config_sha=digest(ROOT/'training/config.json')
    for stage,python,gpu,done in [
        ('vae',VAE_PY,'3',ROOT/'training/vae/status.json'),
        ('export',VAE_PY,'3',ROOT/'training/corpus/manifest.json'),
        ('fsbo',FSBO_PY,'',ROOT/'training/fsbo/status.json')]:
        if done.exists() and load(done).get('status')=='complete':
            print('Already complete',stage,flush=True);continue
        atomic(ROOT/'training/pipeline_status.json',dict(status='running',stage=stage,process=record,config_sha256=config_sha,at=time.time()))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='123')
        with (ROOT/'training'/f'{stage}.log').open('ab') as log:
            cmd=[python,'-u',str(ROOT/'training/gpu_reembed.py')] if stage=='export' else [python,'-u',str(ROOT/'training/train.py'),stage]
            p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            atomic(ROOT/'training'/f'{stage}.process.json',dict(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().rsplit(') ',1)[1].split()[19],at=time.time()))
            code=p.wait()
        if code:raise RuntimeError(f'{stage} exited {code}; see training/{stage}.log')
        assert load(done)['status']=='complete'
    atomic(ROOT/'training/pipeline_status.json',dict(status='trained',stage='promotion',process=record,config_sha256=config_sha,at=time.time()))
    # Promotion fails closed unless real training, service, SQL and batch checks pass.
    command=[VAE_PY,'-u',str(ROOT/'training/promote.py')]
    if '--resume-validated' in sys.argv:command.append('--resume-validated')
    with (ROOT/'training/promotion.log').open('ab') as log:code=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    if code:raise RuntimeError('Promotion failed; see training/promotion.log; old results remain preserved')
    atomic(ROOT/'training/pipeline_status.json',dict(status='complete',stage='rerun_started',process=record,config_sha256=config_sha,at=time.time()))
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        atomic(ROOT/'training/pipeline_status.json',dict(status='failed',error=repr(exc),traceback=traceback.format_exc(),at=time.time()));raise
