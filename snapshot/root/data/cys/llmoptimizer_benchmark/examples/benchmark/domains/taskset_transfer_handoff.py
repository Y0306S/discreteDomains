#!/usr/bin/env python3
"""Transfer the final backfill worker's four cores without interrupting its evaluator."""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

CAMPAIGN = Path('/root/data/logs/taskset_major_families_cpu_20260929')
HOST = 'LLMOptimizer4'
WORKER = 'LLMOptimizer4.w06'
TASK = 'word_rnn_language_model_family_seed94'
CPUS = [28, 29, 30, 31]
PYTHON = '/root/data/cys/baseline/envs/bl_taskset_lolbo/bin/python'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.%d.tmp' % os.getpid())
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def process(pid):
    path = Path('/proc') / str(pid)
    try:
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=int(pid), state=fields[0], ppid=int(fields[1]), start_ticks=fields[19],
                    exit_status=int(fields[49]), args=(path / 'cmdline').read_bytes().decode().split('\0')[:-1])
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(expected):
    current = process(expected['pid'])
    return current if current and current['start_ticks'] == expected['start_ticks'] else None


def pause_supervisor(parent, child):
    current, evaluator = same_process(parent), same_process(child)
    if not current or not evaluator or evaluator['ppid'] != parent['pid']:
        raise RuntimeError('Supervisor/evaluator identity changed before drain')
    os.kill(parent['pid'], signal.SIGSTOP)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        current, evaluator = same_process(parent), same_process(child)
        if current and current['state'] == 'T' and evaluator and evaluator['ppid'] == parent['pid']:
            return
        time.sleep(0.05)
    if same_process(parent):
        os.kill(parent['pid'], signal.SIGCONT)
    raise RuntimeError('Could not pause only the supervisor')


def exit_code(child):
    current = same_process(child)
    if current is None:
        raise RuntimeError('Evaluator disappeared before its exit status was observed')
    if current['state'] != 'Z':
        return None
    status = current['exit_status']
    return os.waitstatus_to_exitcode(status)


def retire_supervisor(parent):
    current = same_process(parent)
    if current is None or current['state'] == 'Z':
        return
    if current['state'] != 'T':
        raise RuntimeError('Expected the drained supervisor to remain paused')
    os.kill(parent['pid'], signal.SIGTERM)
    try:
        os.kill(parent['pid'], signal.SIGCONT)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        current = same_process(parent)
        if current is None or current['state'] == 'Z':
            return
        time.sleep(0.05)
    raise RuntimeError('Drained supervisor has not exited')


def check_host():
    plan = read(CAMPAIGN / 'distributed/plan.json')
    if socket.gethostname() != plan['hosts'][HOST]['hostname']:
        raise RuntimeError('CPU handoff must run on ' + HOST)
    return plan


def arm(training):
    plan = check_host()
    state_path = CAMPAIGN / 'distributed/workers' / (WORKER + '.json')
    saved_path = training / 'handoff.json'
    if saved_path.exists():
        saved = read(saved_path)
        if saved['status'] in ('arming', 'draining'):
            pause_supervisor(saved['parent'], saved['child'])
            saved.update(status='draining')
            write(saved_path, saved)
            return saved
        return saved
    spec = next(w for w in plan['hosts'][HOST]['workers'] if w['name'] == WORKER)
    state = read(state_path)
    if (spec['cpus'] != CPUS or state['cpus'] != CPUS or state['status'] != 'running'
            or state['job']['task_id'] != TASK or state['job']['side'] != 'train'):
        raise RuntimeError('The selected worker is no longer running the final backfill')
    parent, child = process(state['pid']), process(state['child_pid'])
    runner = str(CAMPAIGN / 'distributed/runner.py')
    if not parent or runner not in parent['args'] or WORKER not in parent['args']:
        raise RuntimeError('Supervisor command identity mismatch')
    if not child or TASK not in child['args'] or child['ppid'] != parent['pid']:
        raise RuntimeError('Backfill evaluator identity mismatch')
    saved = dict(status='arming', parent=parent, child=child, job=state['job'], cpus=CPUS,
                 worker=WORKER, original_worker_state=state, created_at=time.time())
    write(saved_path, saved)
    pause_supervisor(parent, child)
    saved.update(status='draining', armed_at=time.time())
    write(saved_path, saved)
    state.update(status='draining_for_transfer_training', supervisor_paused=True,
                 handoff=str(saved_path), updated_at=time.time())
    write(state_path, state)
    return {key: saved[key] for key in ('status', 'worker', 'cpus', 'armed_at')}


def cancel_drain(training):
    saved = read(training / 'handoff.json')
    if saved['status'] in ('arming', 'draining', 'backfill_failed'):
        if same_process(saved['parent']):
            os.kill(saved['parent']['pid'], signal.SIGCONT)
        saved.update(status='cancelled_drain', cancelled_at=time.time())
        write(training / 'handoff.json', saved)
    return dict(status=saved['status'])


def validate_ready(training):
    ready = read(training / 'launch_ready.json')
    if (not ready.get('component_checks_passed') or ready['cpus'] != CPUS
            or ready['methods'] != ['fsbo', 'rgpe'] or ready['full_fsbo_updates'] != 100000):
        raise RuntimeError('Training readiness contract is incomplete')
    for name, expected in ready['files'].items():
        actual = hashlib.sha256((training / 'code' / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError('Prepared training code changed: ' + name)
    return ready


def poll(training):
    check_host()
    validate_ready(training)
    saved = read(training / 'handoff.json')
    if saved['status'] == 'draining':
        code = exit_code(saved['child'])
        if code is None:
            return dict(status='backfill_running', supervisor_paused=True,
                        evaluator_pid=saved['child']['pid'], cpus=CPUS)
        saved.update(returncode=code, evaluator_exit_observed_at=time.time())
        if code != 0:
            saved.update(status='backfill_failed')
            write(training / 'handoff.json', saved)
            return dict(status='backfill_failed', returncode=code)
        path = Path(saved['job']['output_root']) / TASK / 'tpe/seed1/history.json'
        history = read(path)
        if len(history['observations']) != 100:
            raise RuntimeError('Evaluator exited without the full backfill budget')
        saved.update(status='backfill_finished')
        write(training / 'handoff.json', saved)
    if saved['status'] == 'backfill_finished':
        retire_supervisor(saved['parent'])
        nonce = uuid.uuid4().hex
        saved.update(status='launching_controller', nonce=nonce)
        write(training / 'handoff.json', saved)
        with (training / 'controller.log').open('a') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'execute',
                                      '--training', str(training), '--nonce', nonce],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        info = process(child.pid)
        saved.update(status='controller_launched', controller=info)
        write(training / 'handoff.json', saved)
    if saved['status'] == 'launching_controller':
        # A lost launch response cannot cause a second training launch.
        return dict(status='uncertain_controller_launch_requires_inspection', nonce=saved['nonce'])
    if saved['status'] == 'controller_launched':
        state_path = training / 'controller_state.json'
        state = read(state_path) if state_path.exists() else dict(status='starting')
        current = same_process(saved['controller'])
        if (not current or current['state'] == 'Z') and state['status'] not in ('complete', 'failed'):
            return dict(status='controller_unverified', detail=state)
        return dict(status=state['status'], detail=state)
    return dict(status=saved['status'])


def execute(training, nonce):
    plan = check_host()
    ready = validate_ready(training)
    saved = read(training / 'handoff.json')
    if saved.get('nonce') != nonce or saved.get('returncode') != 0:
        raise RuntimeError('Missing verified backfill exit or wrong launch nonce')
    locks = []
    children = {}
    try:
        lock_names = ['worker.' + WORKER, saved['job']['id']] + ['core.%s.%d' % (HOST, cpu) for cpu in CPUS]
        for name in lock_names:
            lock = (CAMPAIGN / 'locks' / (name + '.lock')).open('a')
            locks.append(lock)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.sched_setaffinity(0, CPUS)
        os.nice(10)
        write(training / 'controller_state.json', dict(status='finalizing_backfill', pid=os.getpid(), nonce=nonce))
        history = read(Path(saved['job']['output_root']) / TASK / 'tpe/seed1/history.json')
        observations = history['observations']
        if len(observations) != 100:
            raise RuntimeError('Final backfill history is incomplete')
        successful = sum(o['trial_state'] == 0 and math.isfinite(o['objectives'][0]) for o in observations)
        result = dict(job=saved['job'], host=HOST, worker=WORKER, returncode=0,
                      n_observations=100, n_successful=successful, n_gp_proposals=0,
                      status='complete' if successful == 100 else 'complete_with_failures',
                      finished_at=saved['evaluator_exit_observed_at'], supervisor_handoff=str(training / 'handoff.json'))
        write(CAMPAIGN / 'results' / (saved['job']['id'] + '.json'), result)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(CAMPAIGN / 'code'))
        # The controller owns the job lock. Its child receives that exact lock FD.
        command = [PYTHON, str(training / 'code/finalize_taskset_backfill.py'),
                   '--campaign', str(CAMPAIGN), '--job', saved['job']['id'], '--lock-fd', str(locks[1].fileno())]
        subprocess.run(command, env=env, check=True, pass_fds=(locks[1].fileno(),))
        machine = plan['hosts'][HOST]
        spec = next(w for w in machine['workers'] if w['name'] == WORKER)
        if spec['cpus'] != CPUS:
            raise RuntimeError('CPU allocation changed before handoff')
        # Keep the campaign's shared plan immutable while other jobs archive it.
        # The existing worker/core lock inodes enforce this allocation override.
        write(training / 'cpu_reservation.json', dict(host=HOST, replaces_worker=WORKER,
                                                     fsbo_cpus=CPUS[:2], rgpe_cpus=CPUS[2:],
                                                     lock_files=lock_names, status='reserved'))
        write(CAMPAIGN / 'distributed/workers' / (WORKER + '.json'),
              dict(saved['original_worker_state'], status='handed_off_to_transfer_training',
                   updated_at=time.time(), supervisor_paused=False, training=str(training)))
        write(training / 'controller_state.json', dict(status='exporting_full_corpus', pid=os.getpid(), nonce=nonce))
        subprocess.run([PYTHON, str(training / 'code/prepare_taskset_transfer.py'), '--output', str(training / 'corpus')],
                       env=env, check=True)
        for method, cpus, environment in [('fsbo', CPUS[:2], 'bl_fsbo_exact'), ('rgpe', CPUS[2:], 'bl_rgpe')]:
            output = training / method
            output.mkdir(exist_ok=True)
            if (output / 'launch.json').exists() or (output / 'status.json').exists():
                raise RuntimeError('Existing training launch requires inspection: ' + method)
            command = ['taskset', '-c', ','.join(map(str, cpus)),
                       '/root/data/cys/baseline/envs/%s/bin/python' % environment,
                       str(training / 'code/train_taskset_transfer.py'), method,
                       '--corpus', str(training / 'corpus'), '--output', str(output), '--steps', '100000', '--threads', '2']
            child_env = dict(env, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            held = locks[2:4] if method == 'fsbo' else locks[4:6]
            with (training / (method + '.log')).open('a') as log:
                child = subprocess.Popen(command, env=child_env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                         pass_fds=tuple(lock.fileno() for lock in held))
            children[method] = child
            write(output / 'launch.json', dict(pid=child.pid, cpus=cpus, command=command, nonce=nonce, started_at=time.time()))
        while True:
            codes = {method: child.poll() for method, child in children.items()}
            done = all(code is not None for code in codes.values())
            status = 'running'
            if done:
                good = all(code == 0 and read(training / method / 'status.json')['status'] == 'complete' for method, code in codes.items())
                status = 'complete' if good else 'failed'
            write(training / 'controller_state.json', dict(status=status, pid=os.getpid(), nonce=nonce,
                                                           returncodes=codes, updated_at=time.time()))
            if done:
                return
            time.sleep(5)
    except BaseException as exc:
        write(training / 'controller_state.json', dict(status='failed', pid=os.getpid(), nonce=nonce,
                                                       error=repr(exc), existing_children={k: v.pid for k, v in children.items()}))
        raise
    finally:
        for lock in locks:
            lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('arm', 'poll', 'cancel-drain', 'execute'))
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--nonce')
    args = parser.parse_args()
    args.training.mkdir(parents=True, exist_ok=True)
    if args.action == 'execute':
        execute(args.training, args.nonce)
    else:
        with (args.training / 'handoff.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = {'arm': arm, 'poll': poll, 'cancel-drain': cancel_drain}[args.action](args.training)
        print(json.dumps(result))
