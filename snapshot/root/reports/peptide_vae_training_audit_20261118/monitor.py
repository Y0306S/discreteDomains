"""Durable 10-minute read-only training monitor; writes files, no chat push."""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import time
import traceback

OUT=Path(__file__).resolve().parent
ROOT=Path('/root/data/logs/peptide_vae_fsbo_source_replay_test10_20261118')
LOOP=Path('/root/.codex/loops/peptide-vae-training-10m')
INTERVAL=600


def read(path,default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def check(previous):
    result=dict(checked_at=time.time(),alerts=[],pipeline=read(ROOT/'pipeline_status.json',{}),
        vae=read(ROOT/'vae/status.json',{}),gate=read(ROOT/'reconstruction_gate.json'),
        corpus=read(ROOT/'corpus/status.json'),fsbo=read(ROOT/'fsbo/status.json'))
    pid=result['pipeline'].get('pid')
    result['controller_alive']=False
    if pid:
        try:
            args=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
            result['controller_alive']=str(ROOT/'code/pipeline.py').encode() in args
            result['controller_process_state']=(Path('/proc')/str(pid)/'stat').read_text().split()[2]
            if result['controller_process_state'] in ['Z','T']:
                result['alerts'].append('Controller process is stopped or zombie: '+result['controller_process_state'])
        except OSError:pass
    if result['pipeline'].get('status') not in ['complete'] and not result['controller_alive']:
        result['alerts'].append('Controller is not alive; review pipeline_status.json and pipeline.log')
    if result['pipeline'].get('status')=='failed':result['alerts'].append('Pipeline reports failure')
    signature=(result['vae'].get('attempt'),result['vae'].get('index'))
    oldvae=(previous or {}).get('vae',{})
    oldsignature=(oldvae.get('attempt'),oldvae.get('index'))
    if previous and signature==oldsignature and result['vae'].get('status')=='training':
        result['alerts'].append('No offline VAE task progress since previous 10-minute check')
    if signature!=oldsignature and (ROOT/'vae/state.pt').exists():
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONWARNINGS='ignore')
        path=OUT/'latest_snapshot_check.json'
        run=subprocess.run(['/root/data/cys/envs/bolt_clean_venv/bin/python',str(OUT/'check_snapshot.py'),str(path)],
            env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
        if run.returncode:result['alerts'].append('Snapshot reconstruction audit failed: '+run.stdout[-1500:])
        else:result['snapshot_reconstruction']=read(path)
    elif (previous and previous.get('snapshot_reconstruction') and
            previous['snapshot_reconstruction'].get('attempt')==result['vae'].get('attempt')):
        result['snapshot_reconstruction']=previous['snapshot_reconstruction']
    if 'snapshot_reconstruction' in result:
        baseline=read(OUT/'snapshot_reconstruction.json')['author']
        audit=result['snapshot_reconstruction']
        result['decoder_retention_fraction']=audit['feasible']/baseline['feasible']
        if result['decoder_retention_fraction']<.8:result['alerts'].append('Decoder retention below configured end-of-training gate')
        if audit['reconstruction_loss']>1.25*baseline['reconstruction_loss']:
            result['alerts'].append('Reconstruction loss exceeds configured end-of-training gate')
    result['completed_tasks']=len(list((ROOT/'evaluation/adapted_online').glob('*/result.json')))
    states=[]
    for path in (ROOT/'evaluation/adapted_online').glob('*/status.json'):
        d=read(path);states.append(dict(task=path.parent.name,status=d['status'],rounds=d['rounds'],fresh_evaluations=d['fresh_evaluations'],best_feasible_mic=d['best_feasible_mic']))
    result['evaluation_tasks']=states
    for path in ROOT.glob('evaluation/adapted_online/*/source_refresh/round_*/meta_status.json'):
        d=read(path)
        if d['status']=='complete' and (d['initialization']!='scratch' or not d['all_tasks_sampled'] or d['source_tasks']!=800):
            result['alerts'].append('Invalid completed source-meta checkpoint: '+str(path))
    return result


def main():
    LOOP.mkdir(parents=True,exist_ok=True)
    with (LOOP/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ticks=(Path('/proc')/str(os.getpid())/'stat').read_text().split()[21]
        state=dict(loop_id='peptide-vae-training-10m',job='Read-only peptide VAE/FSBO progress, reconstruction and training health checks',
            targets=[str(ROOT)],interval_seconds=INTERVAL,timezone='Asia/Shanghai',
            mechanism='detached Python process with exclusive flock; writes file reports, no automatic chat notifications',
            schedule_id=f'pid:{os.getpid()}',pid=os.getpid(),start_ticks=ticks,state='active',iteration=0,started_at=time.time())
        due=time.monotonic();previous=read(OUT/'monitor_latest.json')
        while not (LOOP/'STOP').exists():
            state['last_start']=time.time();write(LOOP/'state.json',state)
            try:
                value=check(previous);previous=value
                write(OUT/'monitor_latest.json',value)
                with (OUT/'monitor_history.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
                state['last_result']='alerts' if value['alerts'] else 'healthy'
                state['last_alerts']=value['alerts']
            except Exception:
                state['last_result']='check_failed';state['last_error']=traceback.format_exc()
            state['iteration']+=1;state['last_finish']=time.time()
            due+=INTERVAL
            while due<=time.monotonic():due+=INTERVAL
            state['next_due']=time.time()+due-time.monotonic();write(LOOP/'state.json',state)
            while time.monotonic()<due and not (LOOP/'STOP').exists():time.sleep(min(30,max(0,due-time.monotonic())))
        state.update(state='stopped',next_due=None,stopped_at=time.time());write(LOOP/'state.json',state)


if __name__=='__main__':main()
