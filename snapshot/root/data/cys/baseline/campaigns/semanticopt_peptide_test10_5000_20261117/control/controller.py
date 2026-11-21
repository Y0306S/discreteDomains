"""Finite continuation job: two inference replicas and ten CPU workers."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
INFER = '/root/data/cys/baseline/envs/taskset_qwen35_fast_20261030/bin/python'
ORACLE = '/root/data/cys/envs/bolt_clean_venv/bin/python'


def write(name, data):
    path = ROOT/name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(at=time.time(), pid=os.getpid(), **data), indent=2)+'\n')
    tmp.replace(path)


def health(port):
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
                f'http://127.0.0.1:{port}/health', timeout=3) as response:
            return json.load(response)
    except Exception:
        return None


def start(command, label, env):
    log = (ROOT/'logs'/f'{label}.log').open('ab', buffering=0)
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
        stderr=subprocess.STDOUT, env=env, start_new_session=True)
    log.close()
    return process


class RetainedProcess:
    """Adopt a verified inference service after a controller-only restart."""
    def __init__(self, pid):
        self.pid = pid

    def poll(self):
        try:
            stat = Path(f'/proc/{self.pid}/stat').read_text().split()
            return 1 if stat[2] == 'Z' else None
        except FileNotFoundError:
            return 1

    def terminate(self):
        import signal
        os.kill(self.pid, signal.SIGTERM)

    def kill(self):
        import signal
        os.kill(self.pid, signal.SIGKILL)

    def wait(self, timeout):
        deadline = time.monotonic()+timeout
        while self.poll() is None:
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            time.sleep(.2)


def main():
    guard=(ROOT/'control/controller.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=json.loads((ROOT/'extension.json').read_text())
    stop=json.loads((ROOT/'stop_policy.json').read_text()) if (ROOT/'stop_policy.json').exists() else {}
    target=stop.get('proposals',cfg['final_budget'])
    assert 0 < target <= cfg['final_budget'] and target % cfg['batch_size'] == 0
    env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_QWEN_DEVICE_MAP='auto')
    subprocess.run([ORACLE,'-c',"import sys;sys.path.insert(0,sys.argv[1]);from common import verify;verify();print('Pinned files verified')",
        str(ROOT/'code')],env=env,check=True)
    retained=None
    if (ROOT/'processes.json').exists() and all((health(p) or {}).get('ready') for p in cfg['replica_ports']+[cfg['broker_port']]):
        retained=json.loads((ROOT/'processes.json').read_text())
        for pid in retained['replicas']+[retained['broker']]:
            argv=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            assert str(ROOT/'services/qwen_inference_server.py').encode() in argv
    if retained is None:
        gpu_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        memory={int(a.strip()):int(b.strip()) for a,b in (r.split(',') for r in gpu_rows.splitlines())}
        for pair in cfg['gpu_pairs']:
            for gpu in pair:
                if memory[gpu]>100:raise RuntimeError(f'Assigned GPU {gpu} is occupied: {memory[gpu]} MiB')
    processes=[]; workers=[]
    begun=time.time()
    try:
        for i,pair in enumerate(cfg['gpu_pairs']):
            process=RetainedProcess(retained['replicas'][i]) if retained else start([INFER,'-u',str(ROOT/'services/qwen_inference_server.py'),'replica',
                '--config',str(ROOT/f'peptide/services/replica{i:02}.json')],f'replica{i:02}',
                dict(env,CUDA_VISIBLE_DEVICES=','.join(map(str,pair))))
            processes.append(process)
        write('processes.json',dict(stage='loading_replicas',replicas=[p.pid for p in processes]))
        deadline=time.monotonic()+900
        while True:
            if any(p.poll() is not None for p in processes):raise RuntimeError('Inference replica exited; inspect logs')
            states=[health(port) for port in cfg['replica_ports']]
            write('status.json',dict(status='starting',stage='loading_replicas',
                replica_ready=[bool(s and s.get('ready')) for s in states]))
            if all(s and s.get('ready') for s in states):break
            if time.monotonic()>deadline:raise TimeoutError('Inference replicas failed to become ready')
            time.sleep(5)
        broker=RetainedProcess(retained['broker']) if retained else start([INFER,'-u',str(ROOT/'services/qwen_inference_server.py'),'broker',
            '--config',str(ROOT/'peptide/services/broker.json')],'broker',dict(env,CUDA_VISIBLE_DEVICES=''))
        processes.append(broker)
        deadline=time.monotonic()+90
        while not (health(cfg['broker_port']) or {}).get('ready'):
            if broker.poll() is not None:raise RuntimeError('Broker exited')
            if time.monotonic()>deadline:raise TimeoutError('Broker readiness timed out')
            time.sleep(2)
        write('evidence/replica_health.json',dict(health=health(cfg['broker_port'])))
        for task,cpu in zip(cfg['tasks'],cfg['worker_cpus']):
            process=start(['taskset','-c',str(cpu),ORACLE,'-u',str(ROOT/'code/evaluate.py'),
                'peptide',task],'worker_'+task,dict(env,CUDA_VISIBLE_DEVICES=''))
            workers.append((task,process))
        write('processes.json',dict(stage='evaluating',replicas=[p.pid for p in processes[:2]],
            broker=broker.pid,workers={t:p.pid for t,p in workers},cpu_affinities=cfg['worker_cpus']))
        while True:
            counts={};failed=[]
            for task,process in workers:
                path=ROOT/'peptide/runs'/task/'status.json'
                state=json.loads(path.read_text())
                counts[task]={k:state.get(k) for k in ['status','evaluations','best','random_fallback_evaluations']}
                if process.poll() is not None and (process.returncode or state.get('status')!='complete' or state.get('evaluations')!=target):
                    failed.append(dict(task=task,exit_code=process.returncode))
            if failed:raise RuntimeError('Worker failure: '+json.dumps(failed))
            if any(p.poll() is not None for p in processes):raise RuntimeError('Inference service exited')
            write('status.json',dict(status='running',stage='evaluation',tasks=counts,
                completed=sum(v['status']=='complete' and v['evaluations']==target for v in counts.values()),
                proposals_per_task=target,rounds_per_task=target//cfg['batch_size'],batch_size=cfg['batch_size'],
                evaluations=sum(v['evaluations'] for v in counts.values()),
                additional_evaluations=sum(v['evaluations']-1000 for v in counts.values()),
                elapsed_seconds=time.time()-begun))
            if all(p.poll() is not None for _,p in workers):break
            time.sleep(10)
        subprocess.run([sys.executable,str(ROOT/'control/report.py')],check=True)
        write('status.json',dict(status='complete',stage='reported',completed=10,
            evaluations=target*len(cfg['tasks']),additional_evaluations=(target-cfg['source_budget'])*len(cfg['tasks']),
            proposals_per_task=target,rounds_per_task=target//cfg['batch_size'],elapsed_seconds=time.time()-begun))
    except BaseException as exc:
        write('error.json',dict(error=repr(exc)))
        write('status.json',dict(status='failed',error=repr(exc)))
        raise
    finally:
        for process in [p for _,p in workers]+processes:
            if process.poll() is None:process.terminate()
        for process in [p for _,p in workers]+processes:
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.kill()


if __name__=='__main__':
    main()
