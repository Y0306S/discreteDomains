"""Run four finite two-round pilots after the one-time source screen succeeds."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
TASKS = ['pep_104', 'pep_109', 'pep_111', 'pep_132']


def main():
    while not (ROOT/'prefilter/complete.json').exists():
        pid = json.loads((ROOT/'evidence/source_screen.process.json').read_text())['pid']
        os.kill(pid, 0)
        time.sleep(10)
    assert json.loads((ROOT/'prefilter/complete.json').read_text())['passed']
    from worker import check_pins
    check_pins()
    assert json.loads((ROOT/'evidence/validation.json').read_text())['passed']
    started = time.time()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    processes = []
    results = {}
    for i, task in enumerate(TASKS):
        progress = ROOT/'rgpe/runs'/task/'progress.json'
        if progress.exists():
            events = json.loads(progress.read_text())['events']
            if events >= 2:
                if events != 2:
                    raise RuntimeError('Task exceeded the finite pilot: '+task)
                results[task] = 0
                continue
        command = ['/root/data/cys/envs/bolt_clean_venv/bin/python', '-u', str(ROOT/'code/launch.py'),
                   '--endpoint', 'http://127.0.0.1:%d'%(22540+i),
                   '--sampling-mode', 'bootstrap', '--num-posterior-samples', '100', 'run-task',
                   '--campaign', str(ROOT/'rgpe'), '--task', task, '--cpu', str(104+i),
                   '--server', 'http://127.0.0.1:22560', '--stop-after-events', '2']
        with (ROOT/'dispatch'/(task+'.log')).open('ab') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        record = dict(pid=child.pid, command=command, started=time.time(),
                      start_ticks=Path('/proc/%d/stat'%child.pid).read_text().split()[21])
        (ROOT/'dispatch'/(task+'.process.json')).write_text(json.dumps(record, indent=2)+'\n')
        processes.append((task, child))
        print(json.dumps(dict(task=task, **record)), flush=True)
    for task, child in processes:
        results[task] = child.wait()
    record = dict(started=started, finished=time.time(), elapsed=time.time()-started,
                  exit_codes=results, passed=all(v==0 for v in results.values()))
    (ROOT/'evidence/pilot_processes.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
