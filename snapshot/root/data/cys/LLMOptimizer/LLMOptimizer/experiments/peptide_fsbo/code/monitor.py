"""Existing peptide-vae-training-10m loop, monitoring the new same10 run."""
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
LOOP=Path('/root/.codex/loops/peptide-vae-training-10m')
OUT=ROOT/'monitor'
INTERVAL=600


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def check():
    pipeline=read(ROOT/'pipeline_status.json',{})
    result=dict(checked_at=time.time(),pipeline=pipeline,alerts=[],evaluation=[],controller_alive=False)
    # One read-only remote check verifies stored PID AND start ticks and argv.
    script='''import json
from pathlib import Path
root=Path("/root/data/logs/peptide_target_loss_gate_20round_max2_test10_20261119")
records=[]
for path in [root/"pipeline_status.json"]+list((root/"workers").glob("*.json")):
 d=json.loads(path.read_text());pid=d.get("pid");alive=False
 if pid:
  try:
   p=Path("/proc")/str(pid);stat=p.joinpath("stat").read_text().split();args=p.joinpath("cmdline").read_bytes()
   alive=(stat[21]==str(d.get("start_ticks")) and stat[2] not in ("Z","T") and str(root/"code").encode() in args)
  except OSError:pass
 records.append(dict(file=path.name,pid=pid,alive=alive,status=d.get("status"),task=d.get("task")))
print(json.dumps(records))
'''
    remote=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','LLMOptimizer3','python3','-'],
        input=script,capture_output=True,text=True,timeout=40)
    if remote.returncode:
        result['alerts'].append('Host verification unavailable: '+remote.stderr[-500:])
        result['controller_alive']=None
    else:
        result['processes']=json.loads(remote.stdout)
        result['controller_alive']=next((p['alive'] for p in result['processes'] if p['file']=='pipeline_status.json'),False)
        if pipeline.get('status')!='complete' and not result['controller_alive']:
            result['alerts'].append('Pipeline controller is not running')
        for process in result['processes']:
            if process['status'] in ('running','adopted') and process['file']!='pipeline_status.json' and not process['alive']:
                result['alerts'].append('Worker is not running: '+str(process['task']))
    if pipeline.get('status')=='failed':result['alerts'].append('Pipeline failed; inspect pipeline.log')
    for task in read(ROOT/'config.json')['tasks']:
        directory=ROOT/'evaluation/adapted_online'/task
        final=directory/'result.json';saved=directory/'status.json'
        source=final if final.exists() else saved
        if not source.exists():
            result['evaluation'].append(dict(task=task,status='queued'));continue
        value=read(source)
        entry={k:v for k,v in value.items() if k not in ('observations','fit_events','adaptation_events')}
        decisions=[read(p) for p in sorted((directory/'vae_updates').glob('*/decision.json'))]
        schedule=value.get('retrain_schedule',{})
        if schedule.get('attempts',0)>2:
            result['alerts'].append(task+': retraining attempt limit exceeded')
        if any(e.get('trigger_nonimproving_rounds',20)<20 for e in decisions):
            result['alerts'].append(task+': retraining occurred before20 nonimproving rounds')
        entry.update(task=task,status_source=str(source),update_attempts=max(len(decisions),schedule.get('attempts',0)),
            accepted_updates=sum(e['accepted'] for e in decisions),rejected_updates=sum(not e['accepted'] for e in decisions))
        if decisions:
            event=decisions[-1];entry['last_gate']=event
            for e in decisions:
                if e['accepted'] and e['target_loss_after']['mean_target_loss']>e['target_loss_before']['mean_target_loss']+1e-12:
                    result['alerts'].append(task+': accepted worsening target loss')
                if not e['accepted'] and e['vae_change_l2']!=0:
                    result['alerts'].append(task+': rollback weight delta is nonzero')
        result['evaluation'].append(entry)
    result['completed_tasks']=sum(e['status']=='complete' for e in result['evaluation'])
    result['total_proposals']=sum(e.get('proposals',0) for e in result['evaluation'])
    result['total_fresh_evaluations']=sum(e.get('fresh_evaluations',0) for e in result['evaluation'])
    result['summary']=read(ROOT/'summary.json')
    return result


def main():
    LOOP.mkdir(exist_ok=True,parents=True)
    with (LOOP/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(loop_id='peptide-vae-training-10m',job='Same10 peptide target-loss rollback experiment monitoring',
            targets=[str(ROOT)],interval_seconds=INTERVAL,timezone='Asia/Shanghai',
            mechanism='detached Python process with exclusive flock; read-only SSH/file checks; no automatic chat notifications',
            host=socket.gethostname(),schedule_id=f'pid:{os.getpid()}',pid=os.getpid(),
            start_ticks=(Path('/proc')/str(os.getpid())/'stat').read_text().split()[21],
            state='active',iteration=0,started_at=time.time())
        due=time.monotonic()
        while not (LOOP/'STOP').exists():
            state['last_start']=time.time();write(LOOP/'state.json',state)
            try:
                value=check();write(OUT/'latest.json',value)
                with (OUT/'history.jsonl').open('a') as stream:stream.write(json.dumps(value)+'\n')
                state.update(last_result='alerts' if value['alerts'] else 'healthy',last_alerts=value['alerts'])
            except Exception:state.update(last_result='check_failed',last_error=traceback.format_exc())
            state['iteration']+=1;state['last_finish']=time.time();due+=INTERVAL
            while due<=time.monotonic():due+=INTERVAL
            state['next_due']=time.time()+due-time.monotonic();write(LOOP/'state.json',state)
            while time.monotonic()<due and not (LOOP/'STOP').exists():time.sleep(min(30,max(0,due-time.monotonic())))
        state.update(state='stopped',next_due=None,stopped_at=time.time());write(LOOP/'state.json',state)


if __name__=='__main__':main()
