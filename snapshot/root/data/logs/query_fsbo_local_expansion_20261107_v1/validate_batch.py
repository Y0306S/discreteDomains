"""Validate durable bounded-batch evidence before admitting the four workers."""
from pathlib import Path
import hashlib
import json
import re
import sqlite3
import time
import zlib

root = Path(__file__).resolve().parent
campaign = Path(json.loads((root/'reservation.json').read_text())['campaign'])
worker = json.loads((root/'workers/w01/production.json').read_text())
assert worker['status'] == 'pilot_complete', worker
process = json.loads((root/'batch_process.json').read_text())['workers'][0]
proc = Path('/proc')/str(process['pid'])
if proc.exists():
    fields = (proc/'stat').read_text().split()
    assert fields[21] != process['process_start_ticks'] or fields[2] == 'Z', 'Pilot is still live'
    if fields[21] == process['process_start_ticks']:
        assert int(fields[51]) == 0, 'Pilot exited with an error'
log = Path(process['log']).read_text()
assert 'Traceback (most recent call last)' not in log
directory = campaign/'runs'/worker['task']
status = json.loads((directory/'status.json').read_text())
history = json.loads((directory/'history.json').read_text())
recipe_sha = hashlib.sha256((campaign/'recipe.json').read_bytes()).hexdigest()
assert status['status'] == 'paused' and status['error'] is None
assert status['recipe_sha256'] == recipe_sha == worker['recipe_sha256']
assert status['method'] == 'fsbo' and status['timeout_seconds'] == 50
assert status['initial_observations'] == 49
assert not status['update_e2e'] and not status['recenter']
assert status['e2e_updates'] == status['recenter_calls'] == status['unresolved_database_calls'] == 0
assert history['meta_info'] == status
assert len(history['observations']) == status['charged_observations'] == 49 + status['proposals']
baseline = json.loads((root/'batch_baseline.json').read_text())['tasks'].get(worker['task'])
assert baseline is None, 'Resumed task requires explicit preservation validation'
assert status['proposals'] == 10
checkpoint = directory/'checkpoint.pt'
assert checkpoint.stat().st_size > 0
assert checkpoint.stat().st_mtime >= process['started_at']
connection = sqlite3.connect((directory/'artifacts.sqlite3').as_uri()+'?mode=ro', uri=True)
assert connection.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
names = [row[0] for row in connection.execute('SELECT name FROM artifacts ORDER BY name')]
assert not any(name.endswith('.pending.json') for name in names)
results = [n for n in names if re.fullmatch(r'proposal_\d{4}\.json', n)]
assert results == [f'proposal_{i:04d}.json' for i in range(10)]
verified = 0
for name, payload, checksum in connection.execute('SELECT name,payload,sha256 FROM artifacts'):
    raw = zlib.decompress(payload)
    assert hashlib.sha256(raw).hexdigest() == checksum, name
    if name in results:
        row = json.loads(raw)
        index = int(name[9:13])
        assert row['proposal_index'] == index
        assert row['proposal']['timeout'] == 50
        assert history['observations'][49+index]['extra_info']['proposal_index'] == index
        assert history['observations'][49+index]['extra_info']['plan'] == row['plan']
        for attempt in row.get('database_attempt_history', []):
            assert attempt.get('finished_at') and attempt['status'] == 'returned'
    verified += 1
connection.close()
value = dict(passed=True, at=time.time(), task=worker['task'], recipe_sha256=recipe_sha,
             proposals=status['proposals'], initial_observations=49, unresolved_database_calls=0,
             verified_artifacts=verified, contiguous_indices=True, history_matches=True,
             fresh_task=True, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
             pilot_process=process, status=status)
(root/'batch_validation.json').write_text(json.dumps(value, indent=2)+'\n')
print(json.dumps({k:v for k,v in value.items() if k not in ('pilot_process','status')}, indent=2))
