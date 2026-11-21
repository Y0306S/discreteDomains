"""A bounded production check, separate from the existing recovery watchdog."""
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'batch_code'))
from query_protocol import atomic


def snapshot():
    with urllib.request.urlopen('http://127.0.0.1:21811/health', timeout=30) as stream:
        h = json.load(stream)
    status = json.loads((ROOT / 'status.json').read_text())
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                                  '--format=csv,noheader,nounits'], text=True)
    return dict(at=time.time(), status=status, queued=h['queued'],
                replicas=[{k: r.get(k) for k in ['name', 'ready', 'busy', 'completed', 'error']}
                          for r in h['replica_health']], gpu=gpu)


def requests_between(db, start, finish):
    result = []
    for value, in db.execute('SELECT response FROM requests WHERE response IS NOT NULL AND created>=? AND created<=?',
                            (start - 600, finish)):
        response = json.loads(value)
        if start <= response['finished_at'] <= finish:
            result.append(response)
    return result


def summarize(requests, seconds):
    n = len(requests)
    return dict(requests=n, requests_per_hour=n * 3600 / seconds,
                mean_queue_seconds=sum(r.get('queue_wait_seconds', 0) for r in requests) / max(1, n),
                mean_gpu_admission_wait_seconds=sum(r.get('gpu_admission_wait_seconds', 0) for r in requests) / max(1, n),
                mean_input_tokens=sum(r['generation']['input_tokens'] for r in requests) / max(1, n),
                mean_output_tokens=sum(r['generation']['output_tokens'] for r in requests) / max(1, n))


def main():
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    assert duration > 0 and duration % 30 == 0
    resume = '--resume' in sys.argv
    history = json.loads((HERE / 'production_observations.json').read_text()) if resume else []
    origin = history[0]['at'] if history else time.time()
    start = time.monotonic() - (time.time() - origin)
    state = dict(job='Twelve-replica production throughput comparison', interval_seconds=30,
                 duration_seconds=duration, timezone='Asia/Shanghai', state='running',
                 mechanism='Bounded active Python process; existing recovery watchdog retained',
                 started_at=origin)
    atomic(HERE / 'monitor_state.json', state)
    for index in range(len(history), duration // 30 + 1):
        if index:
            time.sleep(max(0, start + index * 30 - time.monotonic()))
        sample = snapshot()
        history.append(sample)
        atomic(HERE / 'production_observations.json', history)
        state.update(last_finish=sample['at'], last_result=sample['status']['status'],
                     samples=len(history), next_due_at=state['started_at']+(index+1)*30 if index < duration // 30 else None)
        atomic(HERE / 'monitor_state.json', state)
        print(json.dumps(dict(sample=index, measurements=sample['status']['counted_measurements'],
              workers=sample['status'].get('worker_count'), queued=sample['queued'],
              busy=sum(r['busy'] for r in sample['replicas']))), flush=True)
        assert not sample['status']['paused'] and sample['status']['status'] == 'running'
        assert sample['status']['worker_count'] == 21
        assert len(sample['replicas']) == 12 and all(r['ready'] and not r['error'] for r in sample['replicas'])
    first, last = history[0], history[-1]
    elapsed = last['at'] - first['at']
    delta = last['status']['counted_measurements'] - first['status']['counted_measurements']
    before = json.loads((HERE / ('prior_30min_rate.json' if duration == 1800 else 'prior_10min_rate.json')).read_text())
    counters = {r['name']: r['completed'] for r in first['replicas']}
    with sqlite3.connect('file:' + str(ROOT / 'services/orpo.requests.sqlite') + '?mode=ro', uri=True, timeout=120) as db:
        new = summarize(requests_between(db, first['at'], last['at']), elapsed)
        old = summarize(requests_between(db, before['start'], before['end']), before['seconds'])
    rates = dict(at=time.time(), seconds=elapsed, new_evaluations=delta,
                 measurements_before=first['status']['counted_measurements'],
                 measurements_after=last['status']['counted_measurements'],
                 evaluations_per_hour=delta * 3600 / elapsed,
                 baseline=before, production_12=new, production_8=old,
                 per_replica_generations={r['name']: r['completed'] - counters[r['name']] for r in last['replicas']},
                 all_samples_healthy=True, short_window=True,
                 caveat='Sequential real-production windows; task, SQL, and prompt difficulty may differ.')
    rates['evaluation_speed_ratio'] = rates['evaluations_per_hour'] / before['evaluations_per_hour']
    rates['generation_speed_ratio'] = new['requests_per_hour'] / old['requests_per_hour'] if old['requests_per_hour'] else None
    report = HERE / f'production_rate_{duration // 60}min.json'
    atomic(report, rates)
    state.update(state='completed', finished_at=time.time(), next_due_at=None,
                 last_result=str(report))
    atomic(HERE / 'monitor_state.json', state)
    print('RESULT', json.dumps(rates), flush=True)


if __name__ == '__main__':
    main()
