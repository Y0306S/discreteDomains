"""Stop only a verified idle pre-amendment SemanticOPT worker."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parent.parent/'semanticopt_query_400_pilot800_20261123'


def ticks(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except FileNotFoundError:
        return None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--slot',type=int,required=True)
    slot=parser.parse_args().slot
    policy=json.loads((ROOT/'control/query_budget_800.json').read_text())
    old=next(w for w in policy['old_workers'] if w['slot']==slot)
    pid=old['pid'];expected=old['start_ticks']
    state=json.loads((ROOT/'query'/f'worker{slot:02}.status.json').read_text())
    marker=ROOT/'slot_pauses'/f'worker{slot:02}.PAUSE'
    if state['status']!='paused_slot' or not marker.exists():
        raise RuntimeError('Worker has not reached an idle handoff boundary')
    if ticks(pid) is not None:
        if ticks(pid)!=expected:
            raise RuntimeError('Worker PID identity changed')
        command=Path(f'/proc/{pid}/cmdline').read_bytes().split(bytes([0]))
        expected_tail=[b'slot',b'semanticopt',b'--slot',str(slot).encode()]
        if command[-5:-1] != expected_tail:
            raise RuntimeError(f'Unexpected worker command: {command!r}')
        children=Path(f'/proc/{pid}/task/{pid}/children').read_text().strip()
        if children:
            raise RuntimeError(f'Worker still has active children: {children}')
        os.kill(pid,signal.SIGTERM)
        for _ in range(100):
            if ticks(pid) is None:break
            time.sleep(.1)
        else:raise RuntimeError('Idle worker did not stop')
    print(json.dumps(dict(slot=slot,old_pid=pid,status='idle_worker_stopped')))


if __name__=='__main__':main()
