"""Bounded five-minute production throughput check after the resource rollout."""
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import urllib.request

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'batch_code'))
from query_protocol import atomic


def snapshot():
    status=json.loads((ROOT/'status.json').read_text())
    with urllib.request.urlopen('http://127.0.0.1:21811/health',timeout=25) as stream:
        health=json.load(stream)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
    return dict(at=time.time(),status=status,replicas=[{k:r.get(k) for k in ['name','ready','busy','completed','error']} for r in health['replica_health']],queued=health['queued'],gpu=gpu)


def main():
    history=[];started=time.monotonic()
    for index in range(11):
        if index:
            time.sleep(max(0,started+index*30-time.monotonic()))
        sample=snapshot();history.append(sample);atomic(HERE/'production_observations.json',history)
        print(json.dumps(dict(sample=index,at=sample['at'],measurements=sample['status']['counted_measurements'],worker_count=sample['status'].get('worker_count'),queued=sample['queued'],busy=sum(r['busy'] for r in sample['replicas']))),flush=True)
        assert not sample['status']['paused'] and sample['status']['status']=='running'
        assert sample['status']['worker_count']==21 and all(r['ready'] and not r['error'] for r in sample['replicas'])
    first,last=history[0],history[-1]
    elapsed=last['at']-first['at'];delta=last['status']['counted_measurements']-first['status']['counted_measurements']
    counters={r['name']:r['completed'] for r in first['replicas']}
    generation_delta={r['name']:r['completed']-counters[r['name']] for r in last['replicas']}
    baseline=json.loads((HERE/'baseline.json').read_text())
    rates=baseline['eval_baseline']
    requests=[]
    with sqlite3.connect('file:'+str(ROOT/'services/orpo.requests.sqlite')+'?mode=ro',uri=True,timeout=120) as db:
        for value, in db.execute('SELECT response FROM requests WHERE response IS NOT NULL AND created>?',(first['at']-600,)):
            r=json.loads(value)
            if first['at']<=r['finished_at']<=last['at']:requests.append(r)
    result=dict(at=time.time(),elapsed_seconds=elapsed,measurements_before=first['status']['counted_measurements'],measurements_after=last['status']['counted_measurements'],new_evaluations=delta,evaluations_per_hour=delta*3600/elapsed,baseline_evaluations_per_hour=rates['evaluations_per_hour'],evaluation_speed_ratio=(delta*3600/elapsed)/rates['evaluations_per_hour'],new_generations=sum(generation_delta.values()),per_replica_generations=generation_delta,generations_per_hour=sum(generation_delta.values())*3600/elapsed,mean_queue_seconds=sum(r.get('queue_wait_seconds',0) for r in requests)/max(1,len(requests)),mean_generation_seconds=sum(r['finished_at']-r['started_at'] for r in requests)/max(1,len(requests)),mean_input_tokens=sum(r['generation']['input_tokens'] for r in requests)/max(1,len(requests)),short_window=True)
    atomic(HERE/'production_rate.json',result);print('RESULT',json.dumps(result),flush=True)


if __name__=='__main__':main()
