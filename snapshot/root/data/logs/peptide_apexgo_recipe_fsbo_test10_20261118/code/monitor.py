"""Existing durable10-minute monitoring job, retargeted to APEXGo recipe test."""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'monitor'
LOOP=Path('/root/.codex/loops/peptide-vae-training-10m')
INTERVAL=600
COMPLETED_CACHE={}


def read(path,default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)


def evaluation_status(path):
    """Completed atomic results supersede the final running checkpoint status."""
    final=path.with_name('result.json')
    if not final.exists():
        return dict(read(path),status_source=str(path))
    signature=(final.stat().st_mtime_ns,final.stat().st_size)
    cached=COMPLETED_CACHE.get(str(final))
    if cached and cached[0]==signature:
        return cached[1]
    value=read(final)
    rows=value.get('observations',[])
    initial=[r['mic'] for r in rows if r['phase']=='initialization' and r['feasible']]
    best=min(initial) if initial else float('inf')
    last_round=0
    for row in rows:
        if row['phase']!='initialization' and row['feasible'] and row['mic']<best-1e-8:
            best=row['mic'];last_round=row['round']
    status={k:v for k,v in value.items() if k not in ('observations','adaptation_events','fit_events')}
    status.update(status_source=str(final),last_improvement_round=last_round,
                  rounds_since_last_improvement=value['rounds']-last_round,
                  plateau_definition='Completed measured history; flat tail does not establish convergence')
    COMPLETED_CACHE[str(final)]=(signature,status)
    return status


def check(previous):
    pipeline=read(ROOT/'pipeline_status.json',{})
    result=dict(checked_at=time.time(),pipeline=pipeline,alerts=[],vae=read(ROOT/'vae/status.json'),
        corpus=read(ROOT/'corpus/status.json'),fsbo=read(ROOT/'fsbo/meta_status.json'),evaluation=[])
    pid=pipeline.get('pid');alive=False
    if pid:
        try:
            args=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
            alive=str(ROOT/'code/pipeline.py').encode() in args
            alive=alive and (Path('/proc')/str(pid)/'stat').read_text().split()[2] not in ('Z','T')
        except OSError:pass
    result['controller_alive']=alive
    if pipeline.get('status')!='complete' and not alive:result['alerts'].append('Pipeline controller is not running')
    if pipeline.get('status')=='failed':result['alerts'].append('Pipeline failed; inspect pipeline.log')
    vae=result['vae'] or {};old=(previous or {}).get('vae') or {}
    if vae.get('step')!=old.get('step') and (ROOT/'vae/state.pt').exists():
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONWARNINGS='ignore')
        run=subprocess.run(['/root/data/cys/envs/bolt_clean_venv/bin/python',str(ROOT/'code/probe_snapshot.py')],env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
        if run.returncode:result['alerts'].append('Snapshot probe failed: '+run.stdout[-1200:])
    result['source_decoder_probe']=read(OUT/'snapshot_latest.json')
    for path in (ROOT/'evaluation/adapted_online').glob('*/status.json'):
        d=evaluation_status(path);result['evaluation'].append(dict(task=path.parent.name,**d))
    result['completed_tasks']=len(list((ROOT/'evaluation/adapted_online').glob('*/result.json')))
    result['report']=read(ROOT/'report_status.json')
    result['comparison']=read(ROOT/'summary.json') if pipeline.get('status')=='complete' else None
    if pipeline.get('status')=='complete':
        write(OUT/'completion.json',dict(completed_at=time.time(),results=str(ROOT/'RESULTS.md'),plot=str(ROOT/'comparison.png'),summary=str(ROOT/'summary.json')))
    return result


def main():
    OUT.mkdir(exist_ok=True);LOOP.mkdir(parents=True,exist_ok=True)
    with (LOOP/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(loop_id='peptide-vae-training-10m',job='APEXGo recipe VAE/FSBO training and paired results monitoring',
            targets=[str(ROOT)],interval_seconds=INTERVAL,timezone='Asia/Shanghai',
            mechanism='detached Python process with exclusive flock; file reports, no automatic chat notifications',
            schedule_id=f'pid:{os.getpid()}',pid=os.getpid(),start_ticks=(Path('/proc')/str(os.getpid())/'stat').read_text().split()[21],
            state='active',iteration=0,started_at=time.time())
        due=time.monotonic();previous=read(OUT/'latest.json')
        while not (LOOP/'STOP').exists():
            state['last_start']=time.time();write(LOOP/'state.json',state)
            try:
                value=check(previous);previous=value;write(OUT/'latest.json',value)
                with (OUT/'history.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
                state['last_result']='alerts' if value['alerts'] else 'healthy';state['last_alerts']=value['alerts']
            except Exception:
                state['last_result']='check_failed';state['last_error']=traceback.format_exc()
            state['iteration']+=1;state['last_finish']=time.time();due+=INTERVAL
            while due<=time.monotonic():due+=INTERVAL
            state['next_due']=time.time()+due-time.monotonic();write(LOOP/'state.json',state)
            while time.monotonic()<due and not (LOOP/'STOP').exists():time.sleep(min(30,max(0,due-time.monotonic())))
        state.update(state='stopped',next_due=None,stopped_at=time.time());write(LOOP/'state.json',state)


if __name__=='__main__':main()
