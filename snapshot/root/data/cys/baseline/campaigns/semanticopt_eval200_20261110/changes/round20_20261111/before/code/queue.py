"""Durable SemanticOPT queue honoring the audited runtime stage order."""
import os
from pathlib import Path
import signal
import subprocess
import time
from common import *

class Existing:
    def __init__(self,pid):
        self.pid=pid;self.start=(Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split()[19]
    def poll(self):
        try:
            f=(Path('/proc')/str(self.pid)/'stat').read_text().rsplit(') ',1)[1].split()
            return None if f[0]!='Z' and f[19]==self.start else 0
        except OSError:return 0
    def wait(self,timeout=60):
        deadline=time.monotonic()+timeout
        while self.poll() is None:
            if time.monotonic()>deadline:raise subprocess.TimeoutExpired(str(self.pid),timeout)
            time.sleep(1)

def spawn(args,path,env=None):
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            command=[v.decode() for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
            if command==args:
                old=Existing(int(proc.name))
                if old.poll() is None:return old
        except (OSError,UnicodeError):pass
    with path.open('ab') as log:
        return subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
            env=env,cwd=ROOT,start_new_session=True)

def stop(process):
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)

def health(url,processes,timeout=1200):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if any(p.poll() is not None for p in processes):raise RuntimeError('An inference service exited')
        try:
            value=http_json(url+'/health',timeout=5)
            if value.get('ready'):return value
        except (OSError,ValueError,RuntimeError):pass
        time.sleep(5)
    raise TimeoutError('Inference readiness timed out: '+url)

def dependency_complete():
    marker=DEPENDENCY/'orpo_rl/evaluation_complete.json'
    if not marker.exists():return False
    done=load(marker)
    if done.get('status')!='complete' or done.get('tasks')!=120:return False
    tasks=load(DEPENDENCY/'tasks_120.json')
    for task in tasks:
        path=DEPENDENCY/'orpo_rl/state'/('/'.join(task[k] for k in ['task_type','dataset','algorithm'])+'.json')
        if not path.exists():return False
        state=load(path)
        if state.get('status')!='complete' or state.get('progress',{}).get('attempts',0)<80:return False
    return load(DEPENDENCY/'queue_status.json').get('status')=='complete'

def parallel_start_authorized():
    path=ROOT/'scheduling.json'
    return path.exists() and load(path).get('start_alongside_ml') is True

def controllers(domain,smoke=False):
    processes=[]
    for host in ['LLMOptimizer2','LLMOptimizer4']:
        command=[PYTHON,'-u',str(ROOT/'code/host.py'),'--host',host,'--domain',domain]+(['--smoke'] if smoke else [])
        if host=='LLMOptimizer4':command=['ssh','-o','BatchMode=yes','-o','ServerAliveInterval=20',host]+command
        processes.append(spawn(command,ROOT/'logs'/f'{domain}.{host}.{"smoke" if smoke else "dispatch"}.log'))
    return processes

def await_controllers(processes,services,tunnel,domain):
    while any(p.poll() is None for p in processes):
        if (ROOT/'PAUSE').exists():raise RuntimeError('Campaign is paused; inspect task/controller logs')
        if any(p.poll() not in (None,0) for p in processes):
            (ROOT/'PAUSE').touch();raise RuntimeError('CPU controller failed: '+domain)
        if tunnel.poll() is not None or any(p.poll() is not None for p in services):
            (ROOT/'PAUSE').touch();raise RuntimeError('Inference transport failed; campaign paused')
        time.sleep(10)
    if any(p.poll()!=0 for p in processes):raise RuntimeError('A CPU controller failed')

def main():
    guard=lock(ROOT/'queue.lock');recipe=verify()
    count=recipe['replicas']
    preflight=load(ROOT/'offline_preflight.json')
    if not preflight.get('passed') or preflight['recipe_sha256']!=digest(ROOT/'recipe.json'):
        raise ValueError('Offline protocol and recovery checks have not passed')
    if recipe.get('runtime_policy_sha256'):
        amendment=load(ROOT/'fallback_preflight.json')
        if not amendment.get('passed') or amendment['runtime_policy_sha256']!=recipe['runtime_policy_sha256']:
            raise ValueError('Runtime fallback policy has not passed its recovery and constraint checks')
    if recipe.get('peptide_oracle_policy_sha256'):
        check=load(ROOT/'peptide_oracle_preflight.json')
        if not check.get('passed') or check.get('peptide_oracle_policy_sha256')!=recipe['peptide_oracle_policy_sha256']:
            raise ValueError('Peptide CPU oracle batch preflight has not passed')
    while not parallel_start_authorized() and not dependency_complete():
        status(ROOT/'queue_status.json',status='waiting_for_orpo_rl_fsbo',stages=['query','peptide'],
            replicas=count,reserved_replica_slots_per_gpu=2,hosts=['LLMOptimizer2','LLMOptimizer4'],dependency=str(DEPENDENCY/'orpo_rl/evaluation_complete.json'))
        time.sleep(20)
    tunnel=spawn(['ssh','-NT','-o','BatchMode=yes','-o','ControlMaster=no','-o','ControlPath=none',
        '-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3',
        '-R','127.0.0.1:19990:127.0.0.1:19990','LLMOptimizer4'],ROOT/'logs/tunnel.log')
    for stage_index,domain in enumerate(recipe['stages']):
        if (ROOT/domain/'evaluation_complete.json').exists():
            if not all_complete(domain,recipe):raise ValueError('Completion record failed task audit')
            continue
        if (ROOT/'PAUSE').exists():raise RuntimeError('Campaign paused')
        # Other workloads may legitimately own GPU memory; never stop them.
        while True:
            free=list(map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
                '--format=csv,noheader,nounits'],text=True).split()))
            loaded=True;missing={}
            for i in range(count):
                config=load(ROOT/domain/'services'/f'replica{i:02}.json')
                try:
                    h=http_json(f'http://127.0.0.1:{19360+i}/health',timeout=2)
                    ready=h.get('ready') and h.get('identities')==[recipe[domain]['identity']]
                except Exception:ready=False
                if not ready:
                    loaded=False;gpu=config['gpu'];missing[gpu]=max(missing.get(gpu,0),config['memory_gib']*1024)
            if loaded or (len(free)==4 and all(free[gpu]>=need+1024 for gpu,need in missing.items())):break
            status(ROOT/'queue_status.json',status='waiting_for_gpu_memory',stage=domain,free_mib=free)
            time.sleep(20)
        status(ROOT/'queue_status.json',status='loading_inference',stage=domain,replicas=count)
        replicas=[]
        for i in range(count):
            config=ROOT/domain/'services'/f'replica{i:02}.json';cfg=load(config)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(cfg['gpu']),HF_QWEN_DEVICE_MAP='cuda:0',
                OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
                HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            try:
                ready=http_json(f'http://127.0.0.1:{cfg["port"]}/health',timeout=2).get('ready')
            except Exception:ready=False
            if not ready:
                while True:
                    free=list(map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
                        '--format=csv,noheader,nounits'],text=True).split()))
                    if free[cfg['gpu']] >= cfg['memory_gib']*1024+1024:break
                    time.sleep(10)
            replicas.append(spawn([INFER_PYTHON,'-u',str(ROOT/'services/qwen_inference_server.py'),'replica',
                '--config',str(config)],ROOT/'logs'/f'{domain}.replica{i:02}.log',env))
            health(f'http://127.0.0.1:{cfg["port"]}',[replicas[-1]])
        for i in range(count):
            h=health(f'http://127.0.0.1:{19360+i}',replicas)
            if h['identities']!=[recipe[domain]['identity']] or not any(
                v.get('sha256')==recipe[domain]['adapter_sha256'] for v in h['adapter_verifications']):
                raise RuntimeError('Wrong adapter loaded')
        broker=spawn(['/usr/bin/python3','-u',str(ROOT/'services/qwen_inference_server.py'),'broker',
            '--config',str(ROOT/domain/'services/broker.json')],ROOT/'logs'/f'{domain}.broker.log')
        h=health('http://127.0.0.1:19990',[broker])
        for i in range(count):
            http_json('http://127.0.0.1:19990/add_replica',dict(url=f'http://127.0.0.1:{19360+i}'),timeout=30)
        h=health('http://127.0.0.1:19990',[broker])
        if len(h['replica_health'])!=count or not all(r['ready'] for r in h['replica_health']):
            raise RuntimeError('All configured inference replicas must be verified')
        atomic(ROOT/domain/'services/verified_health.json',h)
        atomic(ROOT/domain/'services/processes.json',dict(broker=broker.pid,replicas=[p.pid for p in replicas]))
        status(ROOT/'queue_status.json',status='preflight',stage=domain,replicas=count)
        services=replicas+[broker]
        smoke_path=ROOT/domain/'smoke'/recipe[domain]['smoke_task']/'status.json'
        smoke=load(smoke_path) if smoke_path.exists() else {}
        if not (smoke.get('status')=='complete' and smoke.get('evaluations')==1
                and smoke.get('adapter_sha256')==recipe[domain]['adapter_sha256']
                and smoke.get('recipe_sha256')==digest(ROOT/'recipe.json')
                and smoke.get('runtime_policy_sha256')==recipe.get('runtime_policy_sha256')):
            await_controllers(controllers(domain,smoke=True),services,tunnel,domain)
            smoke=load(smoke_path)
        if smoke.get('status')!='complete' or smoke.get('evaluations')!=1:raise RuntimeError('Live training-task preflight failed')
        atomic(ROOT/domain/'preflight.json',dict(passed=True,at=time.time(),training_task=recipe[domain]['smoke_task'],
            oracle_evaluations=1,replicas=count,adapter_sha256=recipe[domain]['adapter_sha256']))
        if recipe.get('batch_policy'):
            check=ROOT/domain/'batch_preflight.json'
            old=load(check) if check.exists() else {}
            if not (old.get('passed') and old.get('batch_policy_sha256')==recipe['batch_policy_sha256']):
                probe=spawn([PYTHON,'-u',str(ROOT/'code/batch_probe.py'),domain],
                    ROOT/'logs'/f'{domain}.batch_probe.log',dict(os.environ,CUDA_VISIBLE_DEVICES='',
                        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
                while probe.poll() is None:time.sleep(5)
                old=load(check) if check.exists() else {}
                if not (old.get('passed') and old.get('batch_policy_sha256')==recipe['batch_policy_sha256']):
                    raise RuntimeError('Live batch-50 preflight failed; see batch probe log')
        status(ROOT/'queue_status.json',status='running',stage=domain,tasks=200,replicas=count,
            queued=recipe['stages'][stage_index+1:],runtime_policy_sha256=recipe.get('runtime_policy_sha256'),
            batch_policy_sha256=recipe.get('batch_policy_sha256'),batch_size=recipe.get('batch_policy',{}).get('size',1),
            peptide_oracle_execution='cpu_batch' if recipe.get('peptide_oracle_policy',{}).get('enabled') else 'individual',
            cpu_admission='maximum_within_live_cpu_and_memory_limits')
        await_controllers(controllers(domain),services,tunnel,domain)
        if not all_complete(domain,recipe):raise RuntimeError('Incomplete evaluation budgets; next stage remains queued')
        atomic(ROOT/domain/'evaluation_complete.json',dict(status='complete',tasks=200,
            proposals_per_task=recipe[domain]['proposals'],recipe_sha256=digest(ROOT/'recipe.json'),
            adapter_sha256=recipe[domain]['adapter_sha256'],runtime_policy_sha256=recipe.get('runtime_policy_sha256'),
            batch_policy_sha256=recipe.get('batch_policy_sha256'),at=time.time()))
        stop(broker)
        for process in replicas:stop(process)
    stop(tunnel)
    status(ROOT/'queue_status.json',status='complete',stages=recipe['stages'],runtime_policy_sha256=recipe.get('runtime_policy_sha256'))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        status(ROOT/'queue_error.json',error=repr(exc))
        status(ROOT/'queue_status.json',status='error',error=repr(exc))
        raise
