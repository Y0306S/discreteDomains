"""Checkpoint and resume recorded remote workers for an inference-service repair."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

CAMPAIGN = Path('/root/data/logs/peptide_bo_train800_20261027_v5b')
RECORD = CAMPAIGN / 'workers/LLMOptimizer4.json'

def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)

def alive(row):
    try:
        command = Path(f'/proc/{row["pid"]}/cmdline').read_bytes().decode()
        return 'peptide_bo_v5.py' in command and str(CAMPAIGN) in command
    except OSError:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['stop', 'resume'])
    args = parser.parse_args()
    rows = json.loads(RECORD.read_text())
    if args.action == 'stop':
        for row in rows:
            if alive(row):
                os.kill(row['pid'], signal.SIGTERM)
        deadline = time.monotonic() + 180
        while any(alive(row) for row in rows):
            if time.monotonic() > deadline:
                raise RuntimeError('Workers have not yet finished checkpointing')
            time.sleep(2)
        write(CAMPAIGN / 'workers/remote_cache_handoff.json',
            dict(status='checkpointed', time=time.time(), previous_workers=rows))
        print(json.dumps(dict(checkpointed_workers=len(rows))))
        return
    if any(alive(row) for row in rows):
        raise RuntimeError('Resume requires the old managers to have exited')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    for row in rows:
        previous = row['pid']
        with Path(row['log']).open('ab') as log:
            log.write((json.dumps(dict(status='resuming_after_cache_fix', time=time.time())) + '\n').encode())
            log.flush()
            row['resume_log_offset'] = log.tell()
            process = subprocess.Popen(row['command'], env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                cwd='/root/data/logs/peptide_bo_release_20261027_v5')
        row.setdefault('previous_pids', []).append(previous)
        row.update(pid=process.pid, resumed_at=time.time(),
            resume_reason='GPU weight cache expanded to 64 with obsolete versions removed')
        write(RECORD, rows)
    print(json.dumps(dict(resumed_workers=len(rows))))

if __name__ == '__main__':
    main()
