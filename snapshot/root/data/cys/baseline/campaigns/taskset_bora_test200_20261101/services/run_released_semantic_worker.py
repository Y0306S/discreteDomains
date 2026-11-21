"""Additional BORA workers using released RGPE capacity and the frozen optimizer."""
import argparse,contextlib,fcntl,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import run_taskset_bora as original
from campaign_io import verify_campaign,require_live_gates
from cpu_admission import evaluation_slot as quota_slot

def main():
    p=argparse.ArgumentParser();p.add_argument('--host',required=True,choices=['LLMOptimizer2','LLMOptimizer4']);p.add_argument('--index',type=int,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    settings=json.loads((ROOT/'settings.json').read_text());signature=verify_campaign(ROOT);require_live_gates(ROOT)
    allocation=json.loads((ROOT/'services/semantic_handoff_allocation.json').read_text());pool=allocation['hosts'][a.host]
    assert os.uname().nodename==settings['cpu_pools'][a.host]['hostname']
    assert 0<=a.index<pool['workers']
    cpu=pool['cpus'][a.index%len(pool['cpus'])]
    worker=dict(name=f'{a.host}.bora.semantic_handoff.{a.index}',host=a.host,cpus=[cpu])
    if not a.execute:print(json.dumps(dict(worker=worker,identity=signature,allocation=pool,launched=False)));return
    assert os.environ.get('ML_QWEN_POOL_CONFIG')==str(ROOT/'services/pool.json')
    os.sched_setaffinity(0,{cpu});os.nice(8)
    os.environ.pop('ML_SESSION_DIR',None)
    os.environ.update(SEMANTICOPT_HOST=a.host,ML_RUN_ID=settings['run_id'],CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    settings['cpu_pools'][a.host]=dict(settings['cpu_pools'][a.host],initial_workers=min(3,len(pool['cpus'])),max_new_workers=pool['max_evaluations'])
    admission_root=ROOT/'services/semantic_handoff_runtime'
    @contextlib.contextmanager
    def admitted(root,host,settings,method):
        # Inference preparation may overlap; only one additional CPU oracle may
        # execute on each physical core. All are disjoint from existing workers.
        directory=admission_root/'core_leases'/host;directory.mkdir(parents=True,exist_ok=True)
        with (directory/f'cpu{cpu}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            with quota_slot(admission_root,host,settings,method):yield
    original.evaluation_slot=admitted
    tasks=json.loads((ROOT/'assets/tasks.json').read_text())
    (ROOT/f'services/semantic_handoff.worker.{a.host}.{a.index}.ready.json').write_text(json.dumps(dict(pid=os.getpid(),worker=worker,identity=signature,at=time.time())))
    while not (ROOT/'services/semantic_handoff.stop.json').exists():
        outstanding,ran=False,False
        for task in tasks:
            if original.status(ROOT/'results'/(task['task_id']+'.sqlite'))!='pending':continue
            outstanding=True
            try:
                if original.run_task(ROOT,task['task_id'],worker,settings,signature):ran=True;break
            except BlockingIOError:continue
            except Exception:print(traceback.format_exc(),flush=True);raise
        if not outstanding:break
        if not ran:time.sleep(10)

if __name__=='__main__':main()
