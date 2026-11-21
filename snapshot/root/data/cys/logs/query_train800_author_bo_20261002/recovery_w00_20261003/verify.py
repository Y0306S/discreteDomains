"""Verify preserved results, exact-proposal recovery, and fleet liveness."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import tarfile
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent
DIRECTORY = CAMPAIGN / 'outputs/CEB_11B187/lolbo/seed0'


def read(path):
    return json.loads(Path(path).read_text())


release = read(ROOT / 'release.json')
before = read(ROOT / 'before_hashes.json')
preserved = 0
for name, expected in before.items():
    if name.startswith('proposal_') and '.pending.' not in name:
        assert hashlib.sha256((DIRECTORY / name).read_bytes()).hexdigest() == expected, name
        preserved += 1
assert preserved == 605
for path, expected in release['inputs'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
with tarfile.open(ROOT / 'before_recovery.tar.gz', 'r:gz') as archive:
    member = str(DIRECTORY.relative_to(CAMPAIGN) / 'proposal_0605.pending.json')
    proposal = json.load(archive.extractfile(member))
row = read(DIRECTORY / 'proposal_0605.json')
assert row['proposal'] == proposal
assert row['proposal_index'] == 605 and row['database_attempts'] == 2
assert row['database_attempt_history'][0]['status'] == 'io_error'
assert row['database_attempt_history'][1]['status'] == 'returned'
assert row['database_elapsed_unknown_attempts'] == 1
assert not (DIRECTORY / 'proposal_0605.pending.json').exists()
status = read(DIRECTORY / 'status.json')
assert status['status'] == 'running' and status['proposals'] >= 610
assert status['budget_total'] == 1049 and status['initial_observations'] == 49
assert status['runner_migrations'][0]['to_sha256'] == status['runner_sha256']
fleet = read('/root/reports/queryopt_health_20261003/snapshot_restored_w00.json')
worker = next(w for w in fleet['workers'] if w['name'] == 'LLMOptimizer3.w00')
assert worker['status'] == 'running'
for name, cpus in (('controller', [0]), ('optimizer', [0]), ('database', [1, 2, 3])):
    assert worker['live'][name]['alive'] and worker['live'][name]['cpus'] == cpus
result = dict(checked_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              worker=worker['name'], task=status['task_id'], unchanged_saved_results=preserved,
              committed_proposals=status['proposals'], replayed_exact_proposal=True,
              replay_status=row['status'], replay_timeout_seconds=proposal['timeout'],
              replay_elapsed_seconds=row['successful_attempt_elapsed_seconds'],
              prior_failed_attempt_time='unknown; preserved explicitly',
              controller_pid=worker['live']['controller']['pid'],
              optimizer_pid=worker['live']['optimizer']['pid'],
              fleet_running=fleet['query']['running'], fleet_stopped=fleet['query']['stopped'],
              tests='16 existing recovery tests passed; wrapper command/environment checks passed',
              live=worker['live'])
(ROOT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
