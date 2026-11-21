"""Prepare/install the post-BORA runtime; launch is an explicit separate action."""

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CODE = ROOT / 'code'
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / 'examples/benchmark'))
from ml_fixed_campaign import digest, load, save, claim

BASE = Path('/root/data/cys/baseline/campaigns')
CAMPAIGNS = {'tosfit': BASE/'ml_tosfit_132_20261005', 'optformer': BASE/'ml_optformer_132_20261007'}
BORA = BASE/'ml_bora_132_20261005'
CAPACITY = Path('/root/data/cys/logs/bora_capacity_20261011')
SHARED = Path('/root/data/cys/logs/shared_qwen_20261007')
HOLD = Path('/root/data/cys/logs/rl_v14_fsbo_queue_20261009/capacity_hold.json')
QUERY = Path('/root/data/cys/logs/query_test200_baselines_20261012/plan.json')
PYTHON = '/root/data/cys/baseline/envs/ml_paper_20261005/bin/python'
OPT_PYTHON = '/root/data/cys/baseline/envs/bl_optformer/bin/python'
HOSTS = {'LLMOptimizer2': 'cs-94bac-0693e-server', 'LLMOptimizer3': 'cs-0b2cd-18deb-server',
         'LLMOptimizer4': 'cs-fa498-8793f-server'}


def process(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] in ('Z', 'X'):
            return None
        return {'pid': int(pid), 'start_ticks': fields[19],
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except (OSError, IndexError):
        return None


def stop(record):
    if not record:
        return
    record = {key: record[key] for key in ('pid', 'start_ticks', 'boot_id')}
    if process(record['pid']) != record:
        return
    os.kill(record['pid'], signal.SIGTERM)
    until = time.monotonic() + 15
    while process(record['pid']) == record and time.monotonic() < until:
        time.sleep(.1)
    if process(record['pid']) == record:
        raise RuntimeError('Controller did not exit: ' + str(record['pid']))


def on_host(host, action):
    command = [PYTHON, str(Path(__file__)), action, '--host', host]
    if socket.gethostname() != HOSTS[host]:
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, shlex.join(command)]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    if result.returncode:
        raise RuntimeError(host + ': ' + result.stderr[-3000:] + result.stdout[-1000:])
    return json.loads(result.stdout)


def hold():
    if socket.gethostname() != HOSTS['LLMOptimizer2']:
        raise RuntimeError('Run hold on LLMOptimizer2')
    record = load(CAPACITY/'capacity_monitor_process.json')
    if record and process(record['pid']) == record:
        argv = Path(f'/proc/{record["pid"]}/cmdline').read_bytes().decode().split('\0')
        if str(CAPACITY/'deploy.py') not in argv or 'capacity_monitor' not in argv:
            raise RuntimeError('Unexpected BORA completion controller')
        stop(record)
    current = load(HOLD, {})
    current.update(block_orpo=True, deployment=str(ROOT),
                   reason='Capacity reserved for the eight-task ToSFiT and OptFormer plan')
    save(HOLD, current)
    save(ROOT/'hold.json', {'time': time.time(), 'stopped_completion_controller': record,
                           'production_workers_stopped': False})
    return {'orpo_held': True, 'bora_workers_unchanged': True}


def stop_old_queue(host):
    if host != 'LLMOptimizer3' or socket.gethostname() != HOSTS[host]:
        raise RuntimeError('Wrong host for the old OptFormer queue')
    state = load(CAMPAIGNS['optformer']/'queue_status.json', {})
    record = process(state.get('pid'))
    if record:
        argv = Path(f'/proc/{record["pid"]}/cmdline').read_bytes().decode().split('\0')
        if (str(CAMPAIGNS['optformer']) not in argv or
                not any(arg.endswith('/ml_deferred_campaign.py') for arg in argv)):
            raise RuntimeError('The old queue PID is not the expected deferred controller')
        stop(record)
    return {'stopped_old_queue': record}


def prepare():
    if any((ROOT/name).exists() for name in ('installed.json', 'install_intent.json')):
        raise RuntimeError('Installation has begun; do not replace its manifest')
    from ml_runtime_service import tosfit_identity
    allocation = load(Path('/root/data/cys/logs/post_bora_plan_20261012/allocation.json'))['eight_tosfit_model_instances']
    identity = tosfit_identity()
    model = Path('/root/data/cys/baseline/artifacts/optformer_ml_train80_100000')
    opt_identity = load(model/'meta.json')['checkpoint_sha256']
    services = []
    for host in ('LLMOptimizer2', 'LLMOptimizer3'):
        plan = allocation['hosts'][host]
        name = host + '.optformer'
        config = {'kind': 'optformer', 'host': host, 'identity': opt_identity,
                  'cpus': plan['optformer_service_cpus'], 'gpus': '2', 'port': 19620,
                  'model_dir': str(model), 'journal': str(ROOT/'journals'/(name+'.sqlite')),
                  'max_pending': 8, 'python': OPT_PYTHON}
        save(ROOT/'config'/(name+'.json'), config)
        services.append(name)
    for index, cpus in enumerate(allocation['hosts']['LLMOptimizer3']['tosfit_service_cpus']):
        name = 'LLMOptimizer3.tosfit%d' % index
        config = {'kind': 'tosfit', 'host': 'LLMOptimizer3', 'identity': identity,
                  'cpus': cpus, 'gpus': ','.join(map(str, allocation['hosts']['LLMOptimizer3']['tosfit_gpu_pairs'][index])),
                  'port': 19610+index, 'session_root': str(CAMPAIGNS['tosfit']/'sessions/tosfit'),
                  'checkpoint_lock': str(ROOT/'locks/LLMOptimizer3.checkpoint.lock'),
                  'max_pending': 2, 'python': PYTHON}
        save(ROOT/'config'/(name+'.json'), config)
        services.append(name)
        save(ROOT/'config'/('tosfit_client%d.json' % index),
             {'url': 'http://127.0.0.1:%d' % (19610+index), 'identity': identity, 'timeout_seconds': 7200})
    save(ROOT/'config/optformer_client.json', {'urls': ['http://127.0.0.1:19620', 'http://127.0.0.1:19621'],
        'identity': opt_identity, 'timeout_seconds': 1800})
    originals = {}
    estimates = {'adaboost': 8, 'gradient_boosting': 8, 'lightgbm': 12, 'xgboost': 12,
                 'catboost': 16, 'random_forest': 16, 'extra_trees': 16,
                 'cls/dionis/random_forest': 40, 'cls/sf-police-incidents/extra_trees': 40}
    for method, campaign in CAMPAIGNS.items():
        original = load(campaign/'protocol.json')
        originals[method] = digest(campaign/'protocol.json')
        protocol = copy.deepcopy(original)
        for key in ('resource_pool', 'scheduling_revision'):
            protocol.pop(key, None)
        protocol.update(code_root=str(CODE), runtime_deployment=str(ROOT), queue_enabled=False,
                        queue_disabled_reason='Prepared eight-instance runtime; launch approval pending',
                        execution_plan='Private task runtimes with durable RPC; production disabled',
                        cpu_policy='ToSFiT 3 evaluation cores; OptFormer 1 serial / 3 parallel', hosts={})
        hosts = ('LLMOptimizer2', 'LLMOptimizer4') if method == 'tosfit' else ('LLMOptimizer4',)
        for host in hosts:
            plan = allocation['hosts'][host]
            cpus = ([c for slot in (plan['tosfit_local_cpus'] or plan['tosfit_evaluator_cpus']) for c in slot]
                    if method == 'tosfit' else plan['optformer_cpu_pool'])
            spec = dict(hostname=HOSTS[host], cpus=cpus, scheduler_cpus=plan['controller_cpus'],
                cpu_quota=64 if host=='LLMOptimizer4' else 40, memory_gib=256 if host=='LLMOptimizer4' else 400,
                memory_guard='cgroup_working_set', memory_headroom_gib=32, task_memory_gib=estimates,
                memory_admission_registry=str(ROOT/'admission'/(host+'.json')),
                limits={method: 2 if method=='tosfit' else 12}, serial_worker_cores=1, parallel_worker_cores=3,
                gpu_slots=[], gpu_methods=[], launch_interval_seconds=60 if method=='tosfit' else 5,
                throttle_quiet_passes=3)
            if method == 'tosfit':
                spec['worker_cores'] = {'tosfit': 3}
                spec['fixed_slots'] = []
                slots = plan['tosfit_local_cpus'] or plan['tosfit_evaluator_cpus']
                for index, cpuset in enumerate(slots):
                    slot = {'cpus': cpuset, 'gpu': None, 'env': {}}
                    if host=='LLMOptimizer2':
                        slot['gpu'] = ','.join(map(str, plan['tosfit_gpu_pairs'][index]))
                        spec['gpu_slots'].append(slot['gpu'])
                        slot['env']['TOSFIT_CHECKPOINT_LOCK'] = str(ROOT/'locks/LLMOptimizer2.checkpoint.lock')
                    else:
                        slot['env']['TOSFIT_REMOTE_CONFIG'] = str(ROOT/'config'/('tosfit_client%d.json' % index))
                    spec['fixed_slots'].append(slot)
                if host=='LLMOptimizer2':
                    spec.update(gpu_methods=['tosfit'], local_tosfit=True)
            protocol['hosts'][host] = spec
        if method=='optformer':
            protocol.setdefault('worker_env', {})['OPTFORMER_REMOTE_CONFIG'] = str(ROOT/'config/optformer_client.json')
        save(ROOT/'config'/(method+'.protocol.json'), protocol)
    files = list(CODE.rglob('*.py')) + list((ROOT/'config').glob('*.json')) + [Path(__file__)]
    manifest = {'created_at': time.time(), 'status': 'prepared_not_launched',
                'original_protocols': originals, 'services': services, 'query_plan_sha256': digest(QUERY),
                'allocation': allocation, 'files': {str(p.relative_to(ROOT)): digest(p) for p in files}}
    save(ROOT/'manifest.json', manifest)
    return {'prepared': True, 'services': len(services), 'tosfit_target': 8, 'initial_tosfit': 4,
            'initial_optformer': 12}


def verify():
    manifest = load(ROOT/'manifest.json')
    if digest(QUERY) != manifest['query_plan_sha256']:
        raise RuntimeError('QueryOpt reservations changed; revise the allocation')
    for name, expected in manifest['files'].items():
        if digest(ROOT/name) != expected:
            raise RuntimeError('Staged file changed: ' + name)
    if not load(HOLD, {}).get('block_orpo'):
        raise RuntimeError('ORPO capacity hold is missing')
    return manifest


def installed_protocol(method, manifest):
    protocol = load(ROOT/'config'/(method+'.protocol.json'))
    protocol.setdefault('external_checksums', {}).update(
        {str(ROOT/name): expected for name, expected in manifest['files'].items()})
    return protocol


def write_protocol(campaign, protocol):
    save(campaign/'protocol.json', protocol)
    checks = load(campaign/'checksums.json')
    checks['protocol.json'] = digest(campaign/'protocol.json')
    save(campaign/'checksums.json', checks)
    prepared = load(campaign/'prepared.json')
    prepared['checksums_sha256'] = digest(campaign/'checksums.json')
    save(campaign/'prepared.json', prepared)


def install():
    lock = claim(ROOT/'install.lock')
    if lock is None:
        raise RuntimeError('Another installation is in progress')
    with lock:
        return _install()


def _install():
    manifest = verify()
    identity = digest(ROOT/'manifest.json')
    intent = load(ROOT/'install_intent.json')
    if intent and intent['manifest_sha256'] != identity:
        raise RuntimeError('Cannot resume installation with a different manifest')
    for method, campaign in CAMPAIGNS.items():
        original = digest(campaign/'protocol.json') == manifest['original_protocols'][method]
        expected = load(campaign/'protocol.json') == installed_protocol(method, manifest)
        if not original and not (intent and expected):
            raise RuntimeError('Campaign changed since preparation: ' + method)
        index = load(campaign/'task_index.json', {'tasks': []})
        if any(t.get('attempts', 0) or t.get('status')=='running' for t in index['tasks']):
            raise RuntimeError('Campaign already has work; inspect before installing: ' + method)
    hold()
    on_host('LLMOptimizer3', 'stop-old-queue')
    # Back up both campaigns before the first mutation; retries never replace backups.
    for method, campaign in CAMPAIGNS.items():
        backup = ROOT/'before'/method
        backup.mkdir(parents=True, exist_ok=True)
        for name in ('protocol.json', 'checksums.json', 'prepared.json'):
            if not (backup/name).exists():
                if intent:
                    raise RuntimeError('Interrupted installation lost its backup: ' + str(backup/name))
                temporary = backup/(name+'.tmp')
                shutil.copy2(campaign/name, temporary)
                os.replace(temporary, backup/name)
        if digest(backup/'protocol.json') != manifest['original_protocols'][method]:
            raise RuntimeError('Original protocol backup changed: ' + method)
    save(ROOT/'install_intent.json', {'manifest_sha256':identity, 'time':time.time()})
    for method, campaign in CAMPAIGNS.items():
        write_protocol(campaign, installed_protocol(method, manifest))
        save(campaign/'queue_status.json', {'status': 'prepared_disabled', 'runtime_deployment': str(ROOT),
            'updated_at': time.time(), 'reason': 'Launch is separate; ToSFiT target is eight tasks'})
    save(ROOT/'installed.json', {'time': time.time(), 'enabled': False, 'manifest_sha256': digest(ROOT/'manifest.json')})
    return {'installed': True, 'production_enabled': False}


def preflight(host):
    manifest = verify()
    if socket.gethostname() != HOSTS[host]:
        raise RuntimeError('Wrong machine')
    allocation = manifest['allocation']['hosts'][host]
    reserved = set(allocation['controller_cpus'])
    for key in ('tosfit_local_cpus', 'tosfit_evaluator_cpus', 'tosfit_service_cpus'):
        for group in allocation.get(key, []):
            reserved.update(group)
    for key in ('optformer_service_cpus', 'optformer_cpu_pool'):
        reserved.update(allocation.get(key, []))
    if not reserved <= os.sched_getaffinity(0):
        raise RuntimeError('Prepared CPUs are outside this host affinity')
    quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
    if quota != 'max' and int(quota)/int(period) < allocation['total_reserved_cpu_cores']:
        raise RuntimeError('CPU quota below the prepared allocation')
    memory_limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
    required_memory = 256 if host == 'LLMOptimizer4' else 400
    if memory_limit != 'max' and int(memory_limit)/2**30 < required_memory:
        raise RuntimeError('Memory limit below the prepared allocation')
    python = OPT_PYTHON if host != 'LLMOptimizer4' else PYTHON
    if not os.access(python, os.X_OK):
        raise RuntimeError('Missing runtime environment')
    if host == 'LLMOptimizer2':
        for remote in ('LLMOptimizer3', 'LLMOptimizer4'):
            subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                            remote, 'true'], check=True, timeout=15)
    cg = Path('/sys/fs/cgroup')
    stats = dict(line.split() for line in (cg/'memory.stat').read_text().splitlines())
    working = (int((cg/'memory.current').read_text()) - int(stats.get('inactive_file', 0))
               - int(stats.get('slab_reclaimable', 0))) / 2**30
    return {'host':host, 'passed':True, 'reserved_cpu_cores':allocation['total_reserved_cpu_cores'],
            'working_memory_gib':working, 'production_started':False}


def barrier(host):
    from ml_deferred_campaign import predecessor_readiness
    if socket.gethostname() != HOSTS[host]:
        raise RuntimeError('Wrong machine')
    reasons = predecessor_readiness(BORA, host)
    return {'host': host, 'ready': not reasons, 'reasons': reasons}


def drain(host):
    status = barrier(host)
    if not status['ready']:
        raise RuntimeError(str(status['reasons']))
    if host=='LLMOptimizer2':
        hold()
        for name in ('broker_process.json', 'replica_processes.json', 'relay_process.json'):
            records = load(CAPACITY/name, [])
            for record in records if isinstance(records, list) else [records]:
                stop(record)
        for record in load(CAPACITY/'relay_status.json', {}).get('children', {}).values():
            stop(record)
    if host=='LLMOptimizer3':
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            try:
                argv = (entry/'cmdline').read_bytes().decode().split('\0')
                if str(SHARED/'watch_services.py') in argv:
                    stop(process(int(entry.name)))
            except OSError:
                pass
        records = load(SHARED/'LLMOptimizer3/service_processes.json', [])
        if isinstance(records, dict):
            records = records.get('processes', list(records.values()))
        for record in records:
            if isinstance(record, dict) and 'pid' in record:
                stop(record)
    return {'host': host, 'bora_services_drained': True}


def resource_check(host):
    if socket.gethostname() != HOSTS[host]:
        raise RuntimeError('Wrong machine')
    allocation = load(ROOT/'manifest.json')['allocation']['hosts'][host]
    quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
    required = allocation['total_reserved_cpu_cores']
    if quota != 'max' and int(quota)/int(period) < required:
        raise RuntimeError('CPU quota below the prepared allocation')
    if host != 'LLMOptimizer4':
        owned_opt_gib = 0.
        own = load(ROOT/'processes'/(host+'.optformer.json'))
        if own and process(own['pid']) == own:
            applications = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                                    '--format=csv,noheader,nounits'], text=True)
            for row in applications.splitlines():
                pid, memory = row.split(',')
                if int(pid) == own['pid']:
                    owned_opt_gib += float(memory.strip())/1024
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total,memory.used',
                                          '--format=csv,noheader,nounits'], text=True)
        for line in output.splitlines():
            index, total, used = (float(v.strip()) for v in line.split(','))
            need = 45.46484375 + (max(0.,16-owned_opt_gib) if index==2 else 0) + 8
            if used/1024 + need > total/1024:
                raise RuntimeError('GPU%d cannot accommodate the projected full allocation' % index)
    space = os.statvfs(ROOT)
    if space.f_bavail * space.f_frsize / 2**30 < 320:
        raise RuntimeError('Insufficient shared space for eight atomic ToSFiT checkpoints')
    return {'host': host, 'resource_projection_passed': True, 'concurrent_peaks_verified': False}


def spawn(name, command, env):
    record_path = ROOT/'processes'/(name+'.json')
    old = load(record_path)
    if old and process(old['pid']) == old:
        return old
    path = ROOT/'logs'/(name+'.log')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    record = process(child.pid)
    if record is None:
        raise RuntimeError('Process failed at startup: ' + name)
    save(record_path, record)
    return record


def environment(gpus=''):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
        TF_NUM_INTEROP_THREADS='1', TF_NUM_INTRAOP_THREADS='1', XLA_PYTHON_CLIENT_PREALLOCATE='false',
        XLA_PYTHON_CLIENT_MEM_FRACTION='0.2', XLA_FLAGS='--xla_gpu_autotune_level=0',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        PYTHONPATH=f'{CODE}/examples/benchmark:{CODE}:/root/data/cys/baseline/repos/optformer-neurips22:'
                   '/root/data/cys/baseline/repos/t5x-2023-07')


def start_services(host):
    verify()
    if socket.gethostname() != HOSTS[host]:
        raise RuntimeError('Wrong machine')
    if host!='LLMOptimizer4':
        for name in load(ROOT/'manifest.json')['services']:
            config = load(ROOT/'config'/(name+'.json'))
            if config['host'] != host:
                continue
            spawn(name, [config['python'], str(CODE/'examples/benchmark/ml_runtime_service.py'),
                         '--config', str(ROOT/'config'/(name+'.json'))], environment(config['gpus']))
    if host=='LLMOptimizer2':
        controller = load(ROOT/'manifest.json')['allocation']['hosts'][host]['controller_cpus']
        command = ['taskset','-c',','.join(map(str,controller)),'ssh','-NT','-S','none',
                   '-o','ControlMaster=no','-o','BatchMode=yes','-o','ExitOnForwardFailure=yes',
                   '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3']
        from3 = list(command)
        for local, remote in [(p,p) for p in range(19610,19614)] + [(19621,19620)]:
            from3 += ['-L', f'127.0.0.1:{local}:127.0.0.1:{remote}']
        spawn('relay.from3', from3+['LLMOptimizer3'], environment())
        to4 = list(command)
        for port in [*range(19610,19614), 19620, 19621]:
            to4 += ['-R', f'127.0.0.1:{port}:127.0.0.1:{port}']
        spawn('relay.to4', to4+['LLMOptimizer4'], environment())
    return {'host': host, 'services_started': True}


def start_workers(host):
    from ml_fixed_campaign import start
    if socket.gethostname() != HOSTS[host]:
        raise RuntimeError('Wrong machine')
    for method, campaign in CAMPAIGNS.items():
        if host in load(campaign/'protocol.json')['hosts']:
            # Capture the scheduler's JSON in its log; this command itself returns one JSON record.
            command = [PYTHON, str(CODE/'examples/benchmark/ml_fixed_campaign.py'), 'run',
                       '--campaign', str(campaign), '--host', host]
            spawn(host+'.'+method+'.scheduler', command, environment())
    return {'host': host, 'workers_started': True}


def restart_workers(host):
    if socket.gethostname() != HOSTS[host]:
        raise RuntimeError('Wrong machine')
    for method in CAMPAIGNS:
        stop(load(ROOT/'processes'/(host+'.'+method+'.scheduler.json')))
    return start_workers(host)


def expansion_check(host):
    if socket.gethostname() != HOSTS[host]:
        raise RuntimeError('Wrong machine')
    if host=='LLMOptimizer4':
        return {'host': host, 'checked': True}
    evidence = load(ROOT/'expansion_evidence.json')['tasks']
    peaks = {0:0., 1:0.}
    for task in evidence:
        if task['model_host'] == host:
            for logical, physical in enumerate(task['gpus']):
                if physical in peaks:
                    peaks[physical] += task['gpu_peak_reserved_gib'].get(str(logical), 0.)
    increment_gib = max(45.46484375, *peaks.values())
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total,memory.used',
                                      '--format=csv,noheader,nounits'], text=True)
    for line in output.splitlines():
        index, total, used = (float(v.strip()) for v in line.split(','))
        increment = increment_gib if index in (2,3) else 0
        if used/1024 + increment + 8 > total/1024:
            raise RuntimeError('Insufficient GPU headroom for the second ToSFiT pair')
    return {'host': host, 'expansion_projection_passed': True, 'increment_gib':increment_gib}


def expand():
    verify()
    if not (ROOT/'launched.json').exists():
        raise RuntimeError('Start and validate the initial four tasks first')
    import pickle
    evidence = []
    for host in ('LLMOptimizer2', 'LLMOptimizer4'):
        state = load(CAMPAIGNS['tosfit']/'hosts'/(host+'.json'), {})
        if time.time()-state.get('updated_at',0) > 30:
            raise RuntimeError('ToSFiT scheduler heartbeat is stale')
        active = state.get('active', [])
        if len(active) < 2:
            raise RuntimeError('Need two active ToSFiT tasks on each evaluation host')
        for task in active:
            with open(Path(task['session'])/'checkpoint.pkl', 'rb') as stream:
                checkpoint = pickle.load(stream)
            extra = checkpoint['extra']
            if extra.get('version',0) < 2 or not extra.get('gpu_peak_reserved_gib'):
                raise RuntimeError('Initial tasks must complete at least two real gradient rounds')
            if task.get('remote_runtime_config'):
                index = Path(task['remote_runtime_config']).stem.removeprefix('tosfit_client')
                gpus = load(ROOT/'config'/('LLMOptimizer3.tosfit'+index+'.json'))['gpus']
                model_host = 'LLMOptimizer3'
            else:
                gpus, model_host = task['gpu'], host
            evidence.append({'host':host, 'session':task['session'], 'version':extra['version'],
                             'model_host':model_host, 'gpus':[int(g) for g in gpus.split(',')],
                             'gpu_peak_reserved_gib':extra['gpu_peak_reserved_gib']})
    save(ROOT/'expansion_evidence.json', {'time':time.time(), 'tasks':evidence})
    for host in HOSTS:
        on_host(host, 'expansion-check')
    campaign = CAMPAIGNS['tosfit']
    protocol = load(campaign/'protocol.json')
    for spec in protocol['hosts'].values():
        spec['limits']['tosfit'] = 4
    write_protocol(campaign, protocol)
    for host in ('LLMOptimizer2','LLMOptimizer4'):
        on_host(host, 'restart-workers')
    save(ROOT/'expanded.json', {'time':time.time(), 'tosfit_limit':8, 'evidence':evidence})
    return {'tosfit_limit':8, 'optformer_limit':12}


def launch():
    verify()
    if not (ROOT/'installed.json').exists():
        raise RuntimeError('Install the prepared deployment first')
    if (ROOT/'launched.json').exists():
        return load(ROOT/'launched.json')
    from ml_deferred_campaign import verify_campaign
    for campaign in CAMPAIGNS.values():
        verify_campaign(campaign)
    for host in HOSTS:
        on_host(host, 'preflight')
    for host in HOSTS:
        state = on_host(host, 'barrier')
        if not state['ready']:
            raise RuntimeError(str(state['reasons']))
    intent = load(ROOT/'launch_intent.json', {})
    if intent.get('phase') != 'workers':
        for host in HOSTS:
            on_host(host, 'drain')
            on_host(host, 'resource-check')
    for host in HOSTS:
        on_host(host, 'start-services')
    from mindware.components.optimizers.runtime_rpc import call
    # Health checks are issued from machine 4, where all six loopback endpoints are reachable.
    on_host('LLMOptimizer4', 'wait-services')
    for method, campaign in CAMPAIGNS.items():
        protocol = load(campaign/'protocol.json')
        protocol['queue_enabled'] = True
        protocol.pop('queue_disabled_reason', None)
        write_protocol(campaign, protocol)
    save(ROOT/'launch_intent.json', {'phase':'workers', 'time':time.time()})
    for host in HOSTS:
        on_host(host, 'start-workers')
    save(ROOT/'launched.json', {'time': time.time(), 'tosfit_limit': 4, 'optformer_limit': 12,
                              'target_tosfit': 8, 'stage': 'initial_concurrency_validation'})
    return {'launched': True, 'tosfit_initial': 4, 'tosfit_target': 8, 'optformer_initial': 12}


def wait_services():
    from mindware.components.optimizers.runtime_rpc import call
    expected = {p: load(ROOT/'config/tosfit_client0.json')['identity'] for p in range(19610,19614)}
    expected.update({p: load(ROOT/'config/optformer_client.json')['identity'] for p in (19620,19621)})
    until = time.monotonic()+150
    missing = list(expected)
    while time.monotonic() < until:
        missing = []
        for port, identity in expected.items():
            try:
                health = call(f'http://127.0.0.1:{port}/health', timeout=2, retries=0)
                if not health.get('ready') or health.get('identity') != identity:
                    missing.append(port)
            except RuntimeError:
                missing.append(port)
        if not missing:
            return {'all_services_ready': True}
        time.sleep(2)
    raise RuntimeError('Services failed readiness: '+str(missing))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['hold','prepare','verify','install','barrier','stop-old-queue',
        'drain','resource-check','start-services','start-workers','restart-workers','wait-services',
        'expansion-check','launch','expand','preflight'])
    parser.add_argument('--host', choices=HOSTS)
    args = parser.parse_args()
    actions = {'hold': hold, 'prepare': prepare, 'verify': lambda: {'verified': bool(verify())},
               'install': install, 'launch': launch, 'expand': expand, 'wait-services': wait_services}
    function = actions.get(args.action)
    result = function() if function else globals()[args.action.replace('-', '_')](args.host)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
