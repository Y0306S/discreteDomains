"""Durable queue: finish ML, hand off CPU slots, then admit two-GPU full SFT."""
import fcntl
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
from train import ROOT, load, atomic, sha, verify_identity

SEM = ROOT.parent/'semanticopt_eval200_20261110'
ML = ROOT.parent/'ml_orpo_fsbo120_20261109'
GPU = '/root/data/cys/baseline/envs/taskset_qwen35_fast_20261030/bin/python'
RUNTIME = ROOT.parent/'peptide_our_orpo_rl800_20261111/runtime_deps'
FIRST = 'cls/sf-police-incidents/extra_trees'
LAST = 'cls/volkert/catboost'

def publish(**values):
    atomic(ROOT/'status.json', dict(at=time.time(), pid=os.getpid(), **values))

def done(task):
    path = ML/'orpo_rl/state'/(task+'.json')
    return path.exists() and load(path).get('status') == 'complete'

def scheduling(remote_reserved=None, local_reserved=None):
    """Resource amendment only; retain scientific recipe and runtime policy hashes."""
    with (SEM/'sft_scheduling_amendment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = SEM/'scheduling.json'; previous = sha(path); value = load(path)
        if remote_reserved is not None:
            value['orpo_reserved_cpus']['LLMOptimizer4'] = remote_reserved
            value['reservation_purposes']['LLMOptimizer4'] = {'peptide_warmstart_SFT_preparation': remote_reserved}
        if local_reserved is not None:
            old = set(value['orpo_reserved_cpus']['LLMOptimizer2']) - {24,25,26,27,28,29}
            value['orpo_reserved_cpus']['LLMOptimizer2'] = sorted(old | set(local_reserved))
            value['reservation_purposes']['LLMOptimizer2']['peptide_warmstart_full_SFT'] = local_reserved
        value['authorized_by'] = 'User requested two GPU inference slots for peptide full SFT; CPU evaluation allocations return to SemanticOPT'
        value['released_ml_worker_destination'] = 'Two GPU inference slots reserved for full SFT on GPUs 2 and 3; remote CPU allocations return to SemanticOPT'
        atomic(path, value)
        identity = load(SEM/'identity.json')
        identity['files'][str(path)] = sha(path)
        identity['warmstart_sft_resource_amendment'] = dict(at=time.time(), previous_sha256=previous,
            scheduling_sha256=sha(path), campaign=str(ROOT), scientific_recipe_changed=False)
        atomic(SEM/'identity.json', identity)

def environment(gpu=''):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='3', MKL_NUM_THREADS='3',
        OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', DISABLE_VERSION_CHECK='1', WANDB_MODE='disabled',
        HF_HOME=str(ROOT/'hf_cache'), HF_DATASETS_CACHE=str(ROOT/'hf_cache/datasets'),
        TMPDIR=str(ROOT/'scratch'), TORCHINDUCTOR_CACHE_DIR=str(ROOT/'scratch/torchinductor'),
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', MAX_JOBS='2',
        TORCH_EXTENSIONS_DIR=str(ROOT/'scratch/torch_extensions'),
        PATH=str(ROOT/'runtime_deps/bin')+os.pathsep+os.environ.get('PATH',''),
        PYTHONPATH=os.pathsep.join([str(ROOT/'code'),str(ROOT/'runtime_deps'),str(RUNTIME)]))

def ml_model_on_gpu3():
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            args = [x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
            if str(ML/'services/qwen_inference_server.py') in args and 'replica' in args:
                config = load(args[args.index('--config')+1])
                if config['gpu'] == 3: return True
        except (OSError, ValueError, IndexError, UnicodeError): pass
    return False

def controller_clear():
    queue = load(SEM/'queue_status.json')
    if queue.get('status') == 'complete': return True
    path = SEM/queue.get('stage', 'peptide')/'LLMOptimizer2.status.json'
    if not path.exists(): return False
    status = load(path)
    if time.time()-status['at'] > 30: return False
    return all(not {24,25,26,27,28,29}.intersection(w['cpus']) for w in status.get('workers', []))

def remote_clear():
    queue = load(SEM/'queue_status.json')
    if queue.get('status') == 'complete': return True
    path = SEM/queue.get('stage', 'peptide')/'LLMOptimizer4.status.json'
    if not path.exists(): return False
    status = load(path)
    return time.time()-status['at'] < 30 and all(not {0,1,2}.intersection(w['cpus']) for w in status.get('workers', []))

def resources():
    free = [int(x) for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True).split()]
    cg = Path('/sys/fs/cgroup'); maximum = (cg/'memory.max').read_text().strip()
    ram = (int(maximum) if maximum != 'max' else 2**60)-int((cg/'memory.current').read_text())
    fs = os.statvfs(ROOT)
    return dict(gpu_free_gib={str(g):free[g]/1024 for g in (2,3)}, ram_free_gib=ram/2**30,
        disk_free_gib=fs.f_bavail*fs.f_frsize/2**30, free_inodes=fs.f_favail)

def run(command, name, env):
    with (ROOT/'logs'/(name+'.log')).open('ab') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
    atomic(ROOT/'services'/(name+'_process.json'), dict(pid=child.pid, command=command, at=time.time()))
    try:
        while child.poll() is None:
            publish(status='training' if name == 'sft' else 'preparing', stage=name, child_pid=child.pid)
            time.sleep(10)
    except BaseException:
        if child.poll() is None: os.killpg(child.pid, signal.SIGTERM)
        raise
    if child.returncode: raise RuntimeError(name+' failed; see logs/'+name+'.log')

def main():
    lock = (ROOT/'services/queue.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_identity()
    def stop(*_): raise KeyboardInterrupt('Queue interrupted')
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    # Never start a second trainer after a queue process loss.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            args = (proc/'cmdline').read_bytes().split(b'\0')
            if str(ROOT/'code/train.py').encode() in args and b'train' in args:
                raise RuntimeError('Existing SFT trainer found; monitor it before restarting queue')
        except OSError: pass
    held_local = False
    try:
        scheduling(remote_reserved=[0,1,2])
        while not done(FIRST) or not remote_clear():
            publish(status='queued', stage='waiting_for_first_ML_worker',
                sft_preparation_cpus=[0,1,2], other_worker_destination='SemanticOPT',
                first_ml_complete=done(FIRST), last_ml_complete=done(LAST))
            time.sleep(20)
        # Full tokenization uses the base tokenizer on the GPU host. Recheck
        # all shared inputs on the released remote worker before training.
        env = environment()
        remote_env = {k:env[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS','TOKENIZERS_PARALLELISM','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE',
            'DISABLE_VERSION_CHECK','WANDB_MODE','HF_HOME','HF_DATASETS_CACHE','TMPDIR','PYTHONPATH']}
        command = ['env']+[k+'='+v for k,v in remote_env.items()]+['taskset','-c','0,1,2','/usr/bin/python3','-u',str(ROOT/'code/remote_audit.py')]
        run(['ssh','-o','ConnectTimeout=15','LLMOptimizer4',shlex.join(command)], 'remote_preflight', env)
        if not load(ROOT/'evidence/preflight.json')['passed']: raise RuntimeError('Preflight missing')
        scheduling(remote_reserved=[])
        while True:
            r = resources()
            ready = done(FIRST) and done(LAST) and not ml_model_on_gpu3() and all(v >= 23 for v in r['gpu_free_gib'].values()) and r['ram_free_gib'] >= 240
            if ready and not held_local:
                scheduling(local_reserved=[24,25,26,27,28,29]); held_local = True
            if ready and held_local and controller_clear(): break
            publish(status='queued', stage='waiting_for_GPUs2_3_and_CPU_handoff',
                first_ml_complete=done(FIRST), last_ml_complete=done(LAST),
                ml_model_on_gpu3=ml_model_on_gpu3(), local_cpus_reserved=held_local,
                minimum_gpu_free_gib_per_rank=23, **r)
            time.sleep(20)
        if r['disk_free_gib'] < 600 or r['free_inodes'] < 5000:
            raise RuntimeError('Insufficient shared checkpoint storage')
        run(['taskset','-c','24-29',GPU,'-m','torch.distributed.run','--nproc_per_node=2','--master_addr=127.0.0.1','--master_port=19897',str(ROOT/'code/train.py'),'train'], 'sft', environment('2,3'))
        result = load(ROOT/'training_status.json')
        if result['status'] != 'completed' or result['step'] != 2500: raise RuntimeError('Missing completed SFT evidence')
        publish(status='completed', stage='finished', optimizer_steps=2500, epochs=5, model_dir=result['model_dir'], final_weights_evidence=result['final_weights_evidence'])
    except BaseException as error:
        publish(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'error', error=repr(error))
        raise
    finally:
        scheduling(remote_reserved=[], local_reserved=[])

if __name__ == '__main__': main()
