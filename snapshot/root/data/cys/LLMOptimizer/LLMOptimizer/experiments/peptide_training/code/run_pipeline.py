"""Inspect or explicitly execute four TaskSet rollout/training phases on one GPU."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import time

from rl_common import ROOT, parent_for, read, require_execution_enabled
from campaign_io import InfrastructureError, canonical, verify_campaign


def cycle_checkpoint(phase):
    """Retain a complete immutable adapter, tokenizer, optimizer and RNG snapshot."""
    import shutil
    from rl_common import digest
    settings = read('settings.json')
    cycle = phase['cycle']
    last = [p for p in settings['phases'] if p['cycle'] == cycle][-1]
    if phase['id'] != last['id']:
        return
    source = ROOT/'models'/phase['id']/'final'
    target = ROOT/'cycles'/f'round{cycle}'
    metadata = json.loads((source/'complete.json').read_text())
    for filename,key in [('adapter_model.safetensors','adapter_sha256'),('training_state.pt','training_state_sha256')]:
        if digest(source/filename) != metadata[key]:
            raise InfrastructureError('Cannot preserve corrupt cycle checkpoint')
    if not target.exists():
        temporary=target.with_name(target.name+'.tmp')
        shutil.copytree(source,temporary)
        os.rename(temporary,target)
    for filename in ['adapter_model.safetensors','training_state.pt','complete.json']:
        if digest(target/filename) != digest(source/filename):
            raise InfrastructureError('Retained cycle checkpoint differs from completed phase')
    print(canonical(dict(cycle=cycle,status='cycle_checkpoint_preserved',path=str(target))),flush=True)


def pipeline_status(**values):
    temporary=ROOT/'services/pipeline_status.tmp'
    temporary.write_text(canonical(dict(at=time.time(),**values)))
    temporary.replace(ROOT/'services/pipeline_status.json')


def gpu_capacity(gpu, required_gib):
    started=time.monotonic()
    while True:
        output = subprocess.check_output(['nvidia-smi', f'--id={gpu}',
            '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
        free = int(output.strip()) / 1024
        if free >= required_gib:
            return
        if time.monotonic()-started>1800:
            raise InfrastructureError(f'GPU {gpu} has {free:.1f} GiB free; need {required_gib} GiB. Existing jobs are left running.')
        print(canonical(dict(status='waiting_for_gpu_headroom',gpu=gpu,free_gib=free,required_gib=required_gib)),flush=True)
        time.sleep(30)


def execute(args):
    settings = read('settings.json')
    lock = (ROOT / 'services/pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    phases = settings['phases']
    if args.phase:
        phases = [p for p in phases if p['id'] == args.phase]
    for phase in phases:
        name = phase['id']
        pipeline_status(status='running',phase=name,cycle=phase['cycle'])
        if (ROOT / 'models' / name / 'final/complete.json').exists():
            # Trainer validates all completed-phase inputs before accepting it.
            subprocess.run([settings['training_python'], '-B', str(ROOT/'code/train_grpo.py'),
                            '--phase', name, '--execute'], check=True, env=environment(args.gpu))
            cycle_checkpoint(phase)
            continue
        parent_for(name)
        gpu_capacity(args.gpu, args.min_free_gib)
        token = ROOT / 'services' / (name + '.token')
        if not token.exists():
            descriptor = os.open(token, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, 'w') as stream:
                stream.write(secrets.token_hex(32))
        identity = ROOT / 'services' / (name + '.identity.json')
        identity.unlink(missing_ok=True)
        handles, workers = [], []
        server = None
        try:
            server_log = (ROOT/'logs'/f'{name}.server.log').open('a')
            handles.append(server_log)
            server = subprocess.Popen([settings['training_python'], '-u', '-B', str(ROOT/'code/serve_rollouts.py'),
                '--phase', name, '--port', str(args.port), '--token-file', str(token),
                '--identity-out', str(identity), '--max-pending', str(2*args.workers)],
                env=environment(args.gpu), stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
            started, heartbeat = time.monotonic(), 0
            while not identity.exists():
                if server.poll() is not None:
                    raise InfrastructureError('Rollout service failed to start; inspect ' + str(server_log.name))
                if time.monotonic()-started > 1800:
                    raise InfrastructureError('Rollout service startup exceeded 1800 seconds')
                if time.monotonic()-heartbeat > 30:
                    print(canonical(dict(phase=name, status='loading_rollout_model')), flush=True)
                    heartbeat = time.monotonic()
                time.sleep(1)
            for index in range(args.workers):
                log = (ROOT/'logs'/f'{name}.collector{index}.log').open('a')
                handles.append(log)
                workers.append(subprocess.Popen([settings['worker_python'], '-u', '-B', str(ROOT/'code/collect_rollouts.py'),
                    '--phase', name, '--execute', '--worker-index', str(index), '--worker-count', str(args.workers), '--endpoint', f'http://127.0.0.1:{args.port}',
                    '--token-file', str(token), '--identity-file', str(identity)], env=environment(''),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
            while any(worker.poll() is None for worker in workers):
                if server.poll() is not None or any(w.poll() not in (None,0) for w in workers):
                    raise InfrastructureError('Rollout server or collector exited; stop sibling processes')
                if time.monotonic()-heartbeat > 30:
                    print(canonical(dict(phase=name, status='collecting',
                        active_collectors=sum(w.poll() is None for w in workers))), flush=True)
                    heartbeat = time.monotonic()
                time.sleep(1)
            if any(w.returncode for w in workers):
                raise InfrastructureError('A rollout collector failed; inspect its log. No training phase was started.')
        finally:
            # Signals address only children created by this launcher.
            for process in workers + ([server] if server else []):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
            for handle in handles:
                handle.close()
        # load_groups is a hard completeness gate before any optimizer update.
        gpu_capacity(args.gpu, args.min_free_gib)
        subprocess.run([settings['training_python'], '-u', '-B', str(ROOT/'code/train_grpo.py'),
            '--phase', name, '--execute'], check=True, env=environment(args.gpu))
        print(canonical(dict(phase=name, status='complete')), flush=True)
        cycle_checkpoint(phase)
    pipeline_status(status='complete',cycles=settings['cycles'])


def environment(gpu):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HOME=str(ROOT/'hf_cache'), HF_DATASETS_CACHE=str(ROOT/'hf_cache/datasets'), TMPDIR=str(ROOT/'scratch'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', DISABLE_VERSION_CHECK='1', OMP_NUM_THREADS='1',
                MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
                PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                PYTHONPATH=os.pathsep.join([str(ROOT/'code'),str(ROOT/'runtime_deps')]))


def main():
    def terminate(signum, frame):
        raise KeyboardInterrupt('Pipeline termination requested')
    signal.signal(signal.SIGTERM, terminate)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--gpu', default='0')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--port', type=int, default=8874)
    p.add_argument('--min-free-gib', type=float, default=40)
    p.add_argument('--phase', choices=[p['id'] for p in read('settings.json')['phases']])
    a = p.parse_args()
    if a.workers < 1 or a.workers > 12:
        p.error('Use 1–12 CPU workers; default four avoids displacing current evaluations')
    if not a.execute:
        print(canonical(dict(configured=True, launched=False, phases=read('settings.json')['phases'],
            on_hold=read('settings.json')['status']!='ready_to_launch', frozen=(ROOT/'checksums.json').exists(),
            groups=len(read('assets/jobs.json')), min_samples=4*len(read('assets/jobs.json')),
            max_samples=10*len(read('assets/jobs.json')),
            training_group_size=4, max_selected_training_responses=4*len(read('assets/jobs.json')),
            gpu=a.gpu, cpu_workers=a.workers, min_free_gib=a.min_free_gib)))
    else:
        require_execution_enabled()
        verify_campaign(ROOT)
        try:
            execute(a)
        except BaseException as exc:
            pipeline_status(status='failed',error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    main()
