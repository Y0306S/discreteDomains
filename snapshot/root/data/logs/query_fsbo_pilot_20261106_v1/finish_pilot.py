#!/usr/bin/env python3
"""Finish the added worker and check all twenty query workers together."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import subprocess
import threading
import time
import urllib.request

ROOT=Path(__file__).resolve().parent
PLAN=json.loads((ROOT/'plan.json').read_text())
TOKEN=Path(PLAN['token_file']).read_text().strip()


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def call(worker,mode=None):
    url='http://127.0.0.1:%d/'%worker['service_port']
    if mode:
        request=urllib.request.Request(url+'pilot',data=json.dumps(dict(mode=mode)).encode(),
                                      headers={'Authorization':'Bearer '+TOKEN,'Content-Type':'application/json'})
    else:
        request=url+'health'
    try:
        with urllib.request.urlopen(request,timeout=900) as response:return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(worker['name']+': '+exc.read().decode()) from exc


def local_resource():
    root=Path('/sys/fs/cgroup')
    return dict(at=time.time(),memory_current=int((root/'memory.current').read_text()),
                memory_maximum=int((root/'memory.max').read_text()),
                cpu={k:int(v) for k,v in (line.split() for line in (root/'cpu.stat').read_text().splitlines())})


def remote_resource():
    command=['ssh','LLMOptimizer4','python3','-c',
             "import json,pathlib; p=pathlib.Path('/sys/fs/cgroup'); print(json.dumps(dict(memory_current=int((p/'memory.current').read_text()),memory_maximum=int((p/'memory.max').read_text()),cpu={k:int(v) for k,v in (x.split() for x in (p/'cpu.stat').read_text().splitlines())})))"]
    # Quote the remote program for the shell used by ssh.
    import shlex
    command=['ssh','LLMOptimizer4',shlex.join(command[2:])]
    return json.loads(subprocess.run(command,capture_output=True,text=True,check=True,timeout=30).stdout)


def median_query(value):
    return statistics.median(x['wall_seconds'] for x in value['query']['results'] if x['repeat']>=0)


def main():
    def extra_worker(name):
        extra=next(w for w in PLAN['workers'] if w['name']==name)
        extra_path=ROOT/('pilot_extra_'+name+'.json')
        if not extra_path.exists():
            result=dict(worker=name,status='fitting',started=time.time())
            atomic(extra_path,result)
            result['fit']=call(extra,'fit')['fit']
            result['serial_query']=call(extra,'query')
            result.update(status='complete',finished=time.time());atomic(extra_path,result)
            print('Added worker Bao warmup and serial query passed: '+name,flush=True)
        extra_report=json.loads(extra_path.read_text())
        assert extra_report['status']=='complete'
        return extra_report
    with ThreadPoolExecutor(max_workers=3) as pool:
        extra_reports=list(pool.map(extra_worker,['w00','w05','w06']))
    deadline=time.monotonic()+1200
    while True:
        reports={host:json.loads((ROOT/('pilot_'+host+'.json')).read_text())
                 for host in ['LLMOptimizer2','LLMOptimizer4']}
        if any(v['status']=='failed' for v in reports.values()):
            raise ValueError('A host pilot failed')
        if all(v['status']=='complete' for v in reports.values()):break
        if time.monotonic()>deadline:raise TimeoutError('Host pilots did not finish')
        time.sleep(2)
    output=ROOT/'joint_20_query_pilot.json'
    if output.exists():raise ValueError('Joint pilot already recorded; refusing an untracked repeat')
    baselines={x['worker']:median_query(x) for v in reports.values() for x in v['serial_queries']}
    for extra_report in extra_reports:
        baselines[extra_report['worker']]=median_query(extra_report['serial_query'])
    before=remote_resource();samples=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():samples.append(local_resource());stop.wait(0.5)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    start=time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=20) as pool:
            results=list(pool.map(lambda w:call(w,'query'),PLAN['workers']))
    finally:
        stop.set();thread.join(timeout=2)
    after=remote_resource()
    health=[call(w) for w in PLAN['workers']]
    ratios={x['worker']:median_query(x)/baselines[x['worker']] for x in results}
    assert len(results)==20 and all(h['fits']>=1 and h['errors']==0 for h in health)
    assert all(h['optimizer_cpu'] in h['postgres_cpus'] and len(h['postgres_cpus'])==3 for h in health)
    assert all(h['acquisition_wrapper'] is None and not h['optimization_campaign_started'] for h in health)
    min_local_headroom=min((s['memory_maximum']-s['memory_current'])/2**30 for s in samples)
    host_ratios={host:statistics.median(ratios[w['name']] for w in PLAN['workers'] if w['host']==host)
                 for host in reports}
    value=dict(status='complete',workers=20,method='fsbo',domain='ceb_query_runtime',
               allocation=PLAN['allocation'],results=results,health=health,
               median_runtime_ratio=statistics.median(ratios.values()),runtime_ratios=ratios,
               median_runtime_ratio_by_host=host_ratios,
               local_min_memory_headroom_gib=min_local_headroom,
               local_throttled_periods=samples[-1]['cpu']['nr_throttled']-samples[0]['cpu']['nr_throttled'],
               remote_throttled_periods=after['cpu']['nr_throttled']-before['cpu']['nr_throttled'],
               remote_before=before,remote_after=after,local_resource_samples=samples,
               elapsed_seconds=time.monotonic()-start,
               optimization_campaign_started=False,acquisition_wrapper=None,
               checks=dict(all_20_query_workers=True,query_weights_and_bao_warmup=True,
                           three_core_shared_affinity=True,zero_worker_errors=True,
                           local_memory_headroom=min_local_headroom>=30,
                           remote_memory_headroom=min((x['memory_maximum']-x['memory_current'])/2**30 for x in [before,after])>=32,
                           timing_stability=all(r<=1.25 for r in host_ratios.values())))
    value['passed']=all(value['checks'].values())
    atomic(output,value)
    print(json.dumps({k:v for k,v in value.items() if k not in
                      ['results','health','local_resource_samples','remote_before','remote_after']}),flush=True)


if __name__=='__main__':main()
