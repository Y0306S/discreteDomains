"""Durable peptide ORPO -> one GRPO cycle, sharing with SemanticOPT evaluation."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
from peptide_prompt import atomic_json,file_sha

ROOT=Path(__file__).resolve().parents[1]
CPU='/root/data/cys/envs/bolt_clean_venv/bin/python'
GPU='/root/data/cys/baseline/envs/taskset_qwen35_fast_20261030/bin/python'
SEM=Path('/root/data/cys/baseline/campaigns/semanticopt_eval200_20261110')
ML=Path('/root/data/cys/baseline/campaigns/ml_orpo_fsbo120_20261109')

def load(path):return json.loads(Path(path).read_text())
def status(**values):
    value=dict(at=time.time(),pid=os.getpid(),**values)
    atomic_json(ROOT/'services/queue_status.json',value)
    orpo=load(ROOT/'orpo_status.json') if (ROOT/'orpo_status.json').exists() else {}
    atomic_json(ROOT/'status.json',dict(**value,ORPO_status=orpo.get('status','queued'),
        ORPO_step=orpo.get('step',0),ORPO_expected_steps=158,RL_cycles_requested=1,
        RL_completed=(ROOT/'cycles/round1/complete.json').exists(),
        semanticopt_concurrency=True,training_gpu_count=1))
def environment(gpu=''):
    return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4' if str(gpu) else '1',MKL_NUM_THREADS='4' if str(gpu) else '1',
        OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        DISABLE_VERSION_CHECK='1',WANDB_MODE='disabled',HF_HOME=str(ROOT/'hf_cache'),HF_DATASETS_CACHE=str(ROOT/'hf_cache/datasets'),
        TMPDIR=str(ROOT/'scratch'),TORCHINDUCTOR_CACHE_DIR=str(ROOT/'scratch/torchinductor'),
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PYTHONPATH=os.pathsep.join([str(ROOT/'code'),str(ROOT/'runtime_deps')]))

def process_exists(command):
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name)==os.getpid():continue
        try:
            args=[x.decode() for x in (path/'cmdline').read_bytes().split(b'\0') if x]
            if args==command:
                fields=(path/'stat').read_text().rsplit(') ',1)[1].split()
                if fields[0]!='Z':return int(path.name),fields[19]
        except (OSError,UnicodeError):pass
    return None

def alive(pid,start):
    try:
        fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split()
        return fields[0]!='Z' and fields[19]==start
    except OSError:return False

def prepared(path):
    try:return load(path).get('passed') is True
    except (OSError,ValueError):return False

def run(name,command,gpu='',gate=None):
    if gate and prepared(gate):return
    existing=process_exists(command)
    if existing:
        status(status='preparing',stage=name,adopted_pid=existing[0])
        while alive(*existing):time.sleep(10)
        if gate and prepared(gate):return
        raise RuntimeError('Adopted preparation exited without its passed gate: '+name)
    status(status='running' if str(gpu) else 'preparing',stage=name)
    with (ROOT/'logs'/f'{name}.log').open('ab') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=environment(gpu),start_new_session=True)
    try:
        code=child.wait()
    except BaseException:
        if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        raise
    if code:raise RuntimeError(name+' exited '+str(code)+'; inspect logs/'+name+'.log')
    if gate and not prepared(gate):raise RuntimeError('Missing passed gate: '+name)

def dependency_ready():
    marker=ML/'orpo_rl/evaluation_complete.json'
    if not marker.exists():return False
    d=load(marker)
    if d.get('status')!='complete' or d.get('tasks')!=120 or load(ML/'queue_status.json').get('status')!='complete':return False
    for task in load(ML/'tasks_120.json'):
        path=ML/'orpo_rl/state'/('/'.join(task[k] for k in ['task_type','dataset','algorithm'])+'.json')
        if load(path).get('status')!='complete':return False
    return True

def resource_handoff_ready():
    """User-authorized memory handoff can precede the last two ML task results."""
    path=ROOT/'services/resource_handoff.json'
    if not path.exists():return False
    handoff=load(path)
    release=ML/'orpo_rl/services/replica_memory_release.json'
    if not handoff.get('authorized') or handoff.get('training_gpu')!=0:return False
    if handoff['ML_release_sha256']!=file_sha(release):return False
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            args=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
            if str(ML/'services/qwen_inference_server.py') in args and 'replica' in args:
                config=load(args[args.index('--config')+1])
                if config['gpu']==0:return False
        except (OSError,UnicodeError,ValueError,IndexError):continue
    controller=load(SEM/'peptide/LLMOptimizer2.status.json')
    if time.time()-controller['at']>30:return False
    reserved=set(handoff['training_cpu_affinity'])
    return all(not reserved.intersection(w['cpus']) for w in controller.get('workers',[]))

def release_completed_ml_services():
    """Reclaim only the audited-finished ML pool, never active evaluation work."""
    if not dependency_ready():return
    record=ROOT/'services/completed_ml_pool_release.json'
    if record.exists():return
    try:
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open('http://127.0.0.1:19980/health',timeout=5) as response:health=json.load(response)
        if health.get('queued',0)!=0:return
    except OSError:health=dict(ready=False)
    victims=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            command=[x.decode() for x in (path/'cmdline').read_bytes().split(b'\0') if x]
            if ('--config' in command and str(ML/'services/qwen_inference_server.py') in command
                and Path(command[command.index('--config')+1]).parent==ML/'orpo_rl/services'):
                victims.append((int(path.name),command))
        except (OSError,UnicodeError,IndexError):pass
    # Recheck the durable completion proof immediately before releasing models.
    if not dependency_ready():return
    for pid,command in victims:
        try:
            current=[x.decode() for x in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if x]
            if current==command:os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:pass
    atomic_json(record,dict(at=time.time(),tasks_audited_complete=120,broker_queued=health.get('queued',0),
        terminated_only_finished_pool_pids=[pid for pid,_ in victims],SemanticOPT_services_untouched=True))

def wait_resources():
    while True:
        free=[int(v) for v in subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).split()]
        semantic=load(SEM/'queue_status.json');ready=dependency_ready();handoff=resource_handoff_ready()
        if ready:release_completed_ml_services()
        admitted=(ready or handoff) and semantic.get('status') in ('running','complete') and free[0]>=54*1024
        if admitted:
            filesystem=os.statvfs(ROOT)
            if filesystem.f_bavail*filesystem.f_frsize<40*2**30 or filesystem.f_favail<10000:
                raise RuntimeError('Insufficient shared checkpoint byte/inode headroom')
            status(status='resources_admitted',gpu=0,free_gpu_mib=free[0],semanticopt_stage=semantic.get('stage'),semanticopt_status=semantic.get('status'))
            return
        status(status='waiting_for_resources',dependency_complete=ready,resource_handoff_ready=handoff,gpu=0,free_gpu_mib=free[0],minimum_free_gpu_mib=54*1024,
               semanticopt_status=semantic.get('status'),semanticopt_stage=semantic.get('stage'),no_existing_job_stopped=True)
        time.sleep(20)

def freeze():
    settings=load(ROOT/'settings.json');orpo=load(ROOT/'orpo_status.json')
    if orpo['status']!='completed' or file_sha(ROOT/'models/orpo/adapter_model.safetensors')!=orpo['adapter_sha256']:raise ValueError('ORPO adapter corrupt/incomplete')
    settings.update(status='ready_to_launch',parent_adapter_sha256=orpo['adapter_sha256']);atomic_json(ROOT/'settings.json',settings)
    protocol=load(ROOT/'protocol.json');protocol['parent_sha256']=orpo['adapter_sha256'];atomic_json(ROOT/'protocol.json',protocol)
    checks={}
    for directory in ['code','assets']:
        for p in (ROOT/directory).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.npz'):
                checks[str(p.relative_to(ROOT))]=file_sha(p)
    for name in ['settings.json','protocol.json','orpo_config.json','rl_prompt_policy.json','data/orpo_train.jsonl','data/dataset_info.json']:
        checks[name]=file_sha(ROOT/name)
    atomic_json(ROOT/'checksums.json',checks)
    gates={}
    required_gates=[('pairs',ROOT/'reports/independent_audit.json'),('tests',ROOT/'evidence/offline_tests.json'),
        ('ORPO_tokens',ROOT/'orpo_preflight.json'),('GP_export',ROOT/'evidence/source_gp_export.json'),
        ('retrieval',ROOT/'evidence/retrieval_preflight.json'),('RL_tokens',ROOT/'evidence/rl_token_preflight.json'),
        ('CPU_oracle',ROOT/'evidence/cpu_oracle_check.json'),('exact_parent_and_backward',ROOT/'evidence/rl_parent_check.json')]
    if (ROOT/'assets/grpo_task_selection.json').exists():
        scope=load(ROOT/'evidence/grpo_scope_check.json')
        if scope.get('selection_sha256')!=file_sha(ROOT/'assets/grpo_task_selection.json') or scope.get('jobs_sha256')!=file_sha(ROOT/'assets/jobs.json'):
            raise ValueError('GRPO task/prompt scope validation changed')
        required_gates.append(('GRPO_scope',ROOT/'evidence/grpo_scope_check.json'))
    for name,path in required_gates:
        if not prepared(path):raise ValueError('Launch gate pending: '+name)
        gates[name]=dict(status='passed',evidence=str(path),sha256=file_sha(path))
    atomic_json(ROOT/'launch_gates.json',dict(checks=gates))

def main():
    lock=(ROOT/'services/queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic_json(ROOT/'services/queue_process.json',dict(pid=os.getpid(),started_at=time.time(),command=sys.argv))
    os.sched_setaffinity(0,{32,33,34,35})
    def stop(*_):raise KeyboardInterrupt('Training queue interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        audit=ROOT/'reports/independent_audit.json'
        if not prepared(audit) or load(audit)['data_sha256']!=file_sha(ROOT/'data/orpo_train.jsonl'):raise ValueError('Peptide pair audit missing/changed')
        run('orpo_preflight',[GPU,'-u',str(ROOT/'code/ml_orpo_train.py'),'prepare','--campaign',str(ROOT)],gate=ROOT/'orpo_preflight.json')
        run('training_tests',[CPU,'-m','unittest','test_pairs','test_rl_prompt_policy','test_peptide_training','test_selection_updates.OptimizerPath','test_selection_updates.SelectionIntegrity.test_matches_previous_ml_v14_random_replacement'])
        atomic_json(ROOT/'evidence/offline_tests.json',dict(passed=True,at=time.time()))
        run('cpu_oracle',[CPU,'-u',str(ROOT/'code/validate_training.py'),'oracle'],gate=ROOT/'evidence/cpu_oracle_check.json')
        gp_python=load(ROOT/'settings.json')['retrieval']['python']
        run('gp_export',[gp_python,'-u',str(ROOT/'code/export_source_gp.py')],gate=ROOT/'evidence/source_gp_export.json')
        run('retrieval',[CPU,'-u',str(ROOT/'code/prepare_retrieval.py')],gate=ROOT/'evidence/retrieval_preflight.json')
        run('rl_tokens',[GPU,'-u',str(ROOT/'code/validate_training.py'),'tokens'],gate=ROOT/'evidence/rl_token_preflight.json')
        wait_resources()
        run('orpo',[GPU,'-u',str(ROOT/'code/ml_orpo_train.py'),'train','--campaign',str(ROOT)],gpu=0)
        wait_resources()
        run('rl_parent',[GPU,'-u',str(ROOT/'code/validate_training.py'),'parent'],gpu=0,gate=ROOT/'evidence/rl_parent_check.json')
        freeze()
        run('rl_pipeline',[CPU,'-u',str(ROOT/'code/run_pipeline.py'),'--execute','--gpu','0','--workers','4','--port','19895','--min-free-gib','54'],gpu=0)
        checkpoint=ROOT/'cycles/round1/complete.json'
        if not checkpoint.exists():raise ValueError('RL cycle1 checkpoint missing')
        status(status='complete',ORPO=True,RL_cycles=1,cycle_checkpoint=str(checkpoint.parent))
    except BaseException as error:
        status(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'error',error=repr(error))
        atomic_json(ROOT/'services/queue_error.json',dict(error=repr(error),at=time.time()))
        raise

if __name__=='__main__':main()
