"""Read-only five-minute QueryOpt monitoring with verified process identities."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


CAMPAIGN = Path(__file__).resolve().parent
LOOP = Path('/root/.codex/loops/query-train800-5m')
ML = Path('/root/data/cys/llmoptimizer_benchmark/benchmark_data')
TRAINING = Path('/root/blockdata/llmoptimizer_benchmark/benchmark_data/'
                'transfer_training_20260927/full_run1')
TASKSET = Path('/root/data/logs/taskset_major_families_cpu_20260929')
PYTHON = '/root/data/cys/envs/bolt_clean_venv/bin/python'


def iso(value=None):
    return datetime.fromtimestamp(time.time() if value is None else value, timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def process(pid, required):
    if pid is None:
        return dict(alive=False)
    try:
        proc = Path(f'/proc/{pid}')
        argv = proc.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
        stat = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()
        matches = all(any(word in arg for arg in argv) for word in required)
        return dict(pid=pid, alive=matches and stat[0] not in ('Z', 'X'),
                    identity_matches=matches, state=stat[0], birth=stat[19],
                    cpu_seconds=(int(stat[11]) + int(stat[12])) / os.sysconf('SC_CLK_TCK'),
                    cpus=sorted(os.sched_getaffinity(pid)))
    except (OSError, IndexError, ValueError):
        return dict(pid=pid, alive=False)


def host_probe(host):
    plan = read(CAMPAIGN / 'plan.json')
    expected_host = plan['hosts'].get(host)
    if expected_host is None:
        expected_host = read(TASKSET / 'distributed/plan.json')['hosts'][host]['hostname']
    if socket.gethostname() != expected_host:
        raise RuntimeError('SSH connected to the wrong host')
    workers = {}
    for spec in plan['workers']:
        if spec['host'] != host:
            continue
        state = read(CAMPAIGN / 'workers' / f'{spec["name"]}.json')
        database = read(CAMPAIGN / 'databases' / f'{spec["name"]}.json')
        workers[spec['name']] = dict(
            controller=process(state.get('pid'), ['query_train800_fleet.py', spec['name']]),
            optimizer=process(state.get('child_pid'), ['run_query_ceb_baselines.py', state.get('task', '')]),
            database=process(database.get('pid'), ['postgres', spec['pgdata']]))
    taskset_processes = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = proc.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
            if argv and 'python' in Path(argv[0]).name and any(str(TASKSET) in arg for arg in argv[1:]):
                taskset_processes.append(int(proc.name))
        except OSError:
            pass
    result = dict(host=host, hostname=socket.gethostname(), checked_at=iso(),
                  workers=workers, taskset_processes=taskset_processes)
    if host == 'LLMOptimizer2':
        launch = read(TRAINING / 'launch.json')['optformer']
        result['optformer'] = process(launch['trainer_pid'], ['optformer_t5x_train.py'])
    return result


def tail(path, size=32768):
    with Path(path).open('rb') as stream:
        stream.seek(max(0, stream.seek(0, 2) - size))
        return stream.read().decode(errors='replace')


def collect(previous):
    plan = read(CAMPAIGN / 'plan.json')
    hosts, issues = {}, []
    for host in ('LLMOptimizer2', 'LLMOptimizer3', 'LLMOptimizer4'):
        try:
            if host == 'LLMOptimizer2':
                hosts[host] = host_probe(host)
            else:
                command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
                           PYTHON, str(Path(__file__).resolve()), '--probe', host]
                result = subprocess.run(command, text=True, capture_output=True, timeout=30, check=True)
                hosts[host] = json.loads(result.stdout)
        except Exception as exc:
            hosts[host] = dict(unverified=True, error=str(exc))
            issues.append(f'{host}: liveness could not be verified')
    workers = []
    previous_workers = {w['name']: w for w in (previous or {}).get('workers', [])}
    for spec in plan['workers']:
        state = read(CAMPAIGN / 'workers' / f'{spec["name"]}.json')
        live = hosts[spec['host']].get('workers', {}).get(spec['name'])
        task = state.get('task')
        worker = dict(name=spec['name'], host=spec['host'], task=task,
                      saved_status=state['status'], live=live, error=state.get('error'))
        if hosts[spec['host']].get('unverified'):
            worker['status'] = 'unverified'
        elif state['status'] == 'complete' and not live['controller']['alive']:
            worker['status'] = 'complete'
        elif state['status'] == 'running' and live['controller']['alive'] and live['optimizer']['alive']:
            worker['status'] = 'running'
        else:
            worker['status'] = 'stopped'
            issues.append(f'{spec["name"]}: stopped at {task}')
        if live:
            for kind, cpus in [('controller', [spec['bo_cpu']]), ('optimizer', [spec['bo_cpu']]),
                               ('database', spec['db_cpus'])]:
                if live[kind]['alive'] and live[kind]['cpus'] != cpus:
                    issues.append(f'{spec["name"]}: unexpected {kind} CPU affinity')
            if not live['database']['alive']:
                issues.append(f'{spec["name"]}: PostgreSQL is not alive')
        directory = CAMPAIGN / 'outputs' / task / 'lolbo/seed0' if task else None
        worker.update(proposals=0, outcomes={}, pending=[])
        if directory and directory.exists():
            paths = sorted(directory.glob('proposal_[0-9][0-9][0-9][0-9].json'))
            counts = Counter()
            for path in paths:
                counts[read(path)['status']] += 1
            worker.update(proposals=len(paths), outcomes=dict(counts))
            if not paths and (directory / 'status.json').exists():
                summary = read(directory / 'status.json')
                worker['proposals'] = summary['proposals']
            for path in directory.glob('proposal_*.pending.json'):
                proposal = read(path)
                worker['pending'].append(dict(file=path.name, timeout=proposal['timeout'],
                                              age_seconds=round(time.time() - path.stat().st_mtime, 1)))
            if worker['status'] == 'stopped':
                log = CAMPAIGN / 'logs' / f'{spec["name"]}.{task}.run.log'
                if log.exists():
                    errors = [line for line in tail(log).splitlines()
                              if re.match(r'(psycopg\.errors\.|[A-Za-z.]+Error:)', line)]
                    if errors:
                        worker['failure_detail'] = errors[-1]
        older = previous_workers.get(spec['name'], {})
        if older.get('task') == task:
            worker['proposal_delta'] = worker['proposals'] - older.get('proposals', 0)
        worker['last_progress_at'] = (older.get('last_progress_at', iso())
                                      if worker.get('proposal_delta') == 0 else iso())
        quiet_seconds = time.time() - datetime.fromisoformat(worker['last_progress_at']).timestamp()
        expected_wait = max([900] + [float(p['timeout']) + 60 for p in worker['pending']])
        if worker['status'] == 'running' and quiet_seconds > expected_wait:
            issues.append(f'{spec["name"]}: no new saved proposal for {int(quiet_seconds)} seconds')
        workers.append(worker)
    claims = [read(p) for p in (CAMPAIGN / 'claims').glob('*.json')]
    complete = sum(c['status'] == 'complete' for c in claims)
    observed_proposals = {w['task']: w['proposals'] for w in workers if w['task']}
    total_proposals = complete * 1000
    for claim in claims:
        directory = CAMPAIGN / 'outputs' / claim['task'] / 'lolbo/seed0'
        if claim['status'] != 'complete':
            total_proposals += (observed_proposals[claim['task']] if claim['task'] in observed_proposals
                                else len(list(directory.glob('proposal_[0-9][0-9][0-9][0-9].json'))))
    ml = {}
    for folder in ('ml_test_20_fixed_ckpt32_retry5_20260929', 'ml_test_20_fixed_48h_20260925',
                   'ml_test_20_fixed_fsbo_rgpe_20260928'):
        summary = read(ML / folder / 'task_index.json')['summary']
        for method, counts in summary.items():
            if method == 'ckpt32' and 'retry5' not in folder:
                continue
            ml[method] = counts
    log = TRAINING / 'optformer_retry3.log'
    text = tail(log)
    steps = re.findall(r'\[(\d+)\] collection=train|Training: step (\d+)', text)
    latest_step = max(int(a or b) for a, b in steps) if steps else None
    optformer = dict(step=latest_step, total=100000, log_age_seconds=time.time() - log.stat().st_mtime,
                     live=hosts['LLMOptimizer2'].get('optformer'))
    if not optformer['live'] or not optformer['live']['alive']:
        issues.append('OptFormer trainer is not alive')
    elif optformer['log_age_seconds'] > 900:
        issues.append('OptFormer log has not advanced for over 15 minutes')
    taskset_count = sum(len(h.get('taskset_processes', [])) for h in hosts.values())
    storage = os.statvfs('/root/data')
    result = dict(checked_at=iso(), timezone='Asia/Shanghai', campaign=str(CAMPAIGN),
                  query=dict(tasks=800, completed=complete, running=sum(w['status'] == 'running' for w in workers),
                             stopped=sum(w['status'] == 'stopped' for w in workers),
                             unverified=sum(w['status'] == 'unverified' for w in workers),
                             unstarted=800 - len(claims), proposals=total_proposals, proposal_budget=800000),
                  workers=workers, hosts=hosts, ml=ml, optformer=optformer,
                  taskset=dict(visible_processes=taskset_count,
                               liveness_verified=all(not h.get('unverified') for h in hosts.values())),
                  storage=dict(free_bytes=storage.f_bavail * storage.f_frsize, free_inodes=storage.f_favail),
                  issues=issues)
    return result


def render(result):
    q = result['query']
    checked = datetime.fromisoformat(result['checked_at']).astimezone(ZoneInfo('Asia/Shanghai'))
    lines = [f'Checked: {checked.isoformat(timespec="seconds")}',
             f'QueryOpt: {q["completed"]}/800 complete; {q["running"]} running; '
             f'{q["stopped"]} stopped; {q["unverified"]} unverified; {q["proposals"]}/800000 proposals.', '',
             '| Worker | Task | Status | Proposals / 1000 | Change |',
             '| --- | --- | --- | ---: | ---: |']
    for w in result['workers']:
        lines.append(f'| {w["name"]} | {w["task"]} | {w["status"]} | '
                     f'{w["proposals"]} | {w.get("proposal_delta", "first check")} |')
    for w in result['workers']:
        if w.get('failure_detail'):
            lines.append(f'\n{w["name"]}: `{w["failure_detail"]}`\n')
    lines += ['', f'ckpt32: {result["ml"]["ckpt32"]["clean_total"]}/132 complete.',
              f'OptFormer: {result["optformer"]["step"]}/100000 training steps.',
              f'TaskSet processes: {result["taskset"]["visible_processes"]}; '
              f'verified={result["taskset"]["liveness_verified"]}.', '', 'Issues:']
    lines += ['- ' + issue for issue in result['issues']] or ['None detected.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(host_probe(args.probe)))
        return
    LOOP.mkdir(parents=True, exist_ok=True)
    os.sched_setaffinity(0, {22})
    os.nice(10)
    with (LOOP / 'monitor.lock').open('a') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        running = True

        def stop(signum, frame):
            nonlocal running
            running = False

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        interval = 300
        started = time.monotonic()
        due = started
        iteration = 0
        state = dict(loop_id='query-train800-5m', job='Monitor QueryOpt, ML completion, OptFormer and TaskSet liveness',
                     targets=[str(CAMPAIGN), str(ML), str(TRAINING), str(TASKSET)],
                     interval_seconds=interval, timezone='Asia/Shanghai',
                     mechanism='detached Python process; file reports, no automatic chat notifications',
                     pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                     started_at=iso(), status='running', iteration=0)
        write(LOOP / 'state.json', state)
        while running:
            state.update(last_start=iso())
            write(LOOP / 'state.json', state)
            try:
                previous = read(LOOP / 'latest.json') if (LOOP / 'latest.json').exists() else None
                result = collect(previous)
                write(LOOP / 'latest.json', result)
                report = LOOP / 'latest.md'
                temporary = report.with_suffix('.tmp')
                temporary.write_text(render(result))
                temporary.replace(report)
                with (LOOP / 'history.jsonl').open('a') as stream:
                    stream.write(json.dumps(result) + '\n')
                state['last_result'] = 'issues' if result['issues'] else 'ok'
                print(render(result), flush=True)
            except Exception as exc:
                state.update(last_result='check_failed', last_error=repr(exc))
                print(f'{iso()} Monitor failed: {exc!r}', flush=True)
            iteration += 1
            due = started + (int((time.monotonic() - started) // interval) + 1) * interval
            state.update(iteration=iteration, last_finish=iso(), next_due_unix=time.time() + due - time.monotonic())
            write(LOOP / 'state.json', state)
            if args.once:
                running = False
            while running and time.monotonic() < due:
                time.sleep(min(1, max(0, due - time.monotonic())))
        state.update(status='stopped', stopped_at=iso(), next_due_unix=None)
        write(LOOP / 'state.json', state)


if __name__ == '__main__':
    main()
