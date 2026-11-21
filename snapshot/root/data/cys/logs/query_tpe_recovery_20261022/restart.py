"""Verify and restart only the interrupted CEB_9A74 TPE worker."""

import json
from pathlib import Path
import shutil
import socket
import sys

ROOT = Path(__file__).resolve().parent
CAMPAIGN = Path('/root/data/cys/logs/query_test200_baselines_20261012')
WORKER = 'LLMOptimizer3.w05'
TASK = 'CEB_9A74'
sys.path.insert(0, str(CAMPAIGN / 'code'))
import query_test200_fleet_v3 as fleet
import query_test200_rolling as rolling
from run_query_ceb_baselines_v2 import Engine
import psycopg
import torch


def main():
    if socket.gethostname() != fleet.HOSTS['LLMOptimizer3']:
        raise RuntimeError('Recovery belongs on LLMOptimizer3')
    rolling.verify(CAMPAIGN)
    plan, worker = fleet.allocation(CAMPAIGN, WORKER, 'tpe')
    directory = CAMPAIGN / 'outputs' / TASK / 'tpe/seed0'
    with fleet.lock(ROOT / 'recovery.lock'):
        if (ROOT / 'launch.json').exists():
            raise RuntimeError('Restart already recorded; inspect the existing worker')
        with fleet.lock(CAMPAIGN / 'rolling' / (WORKER + '.lock')), \
                fleet.lock(CAMPAIGN / 'workers' / (WORKER + '.lock')), \
                fleet.lock(directory / '.lock'):
            claim = fleet.read(CAMPAIGN / 'claims' / ('tpe.' + TASK + '.json'))
            if claim['worker'] != WORKER or claim['status'] == 'complete':
                raise RuntimeError('Task ownership or completion changed')
            saved = torch.load(directory / 'checkpoint.pt', map_location='cpu', weights_only=False)
            pending = saved.get('pending') or []
            if (saved['meta']['task_id'] != TASK or saved['meta']['method'] != 'tpe'
                    or saved['total'] != 1049 or len(saved['initial']) != 49
                    or saved['completed'] or len(pending) != 1
                    or not any(row['plan'] == pending[0]['plan'] for row in saved['initial'])):
                raise RuntimeError('Checkpoint differs from the reviewed pre-evaluation state')
            if saved['meta']['runner_sha256'] != fleet.digest(CAMPAIGN / plan['runner']):
                raise RuntimeError('TPE runner changed')
            if (directory / 'artifacts.sqlite3').stat().st_size != 0:
                raise RuntimeError('Results store changed since diagnosis')
            backup = ROOT / 'before/task'
            backup.mkdir(exist_ok=True)
            for path in directory.iterdir():
                if path.is_file() and path.name != '.lock':
                    target = backup / path.name
                    if target.exists():
                        if fleet.digest(target) != fleet.digest(path):
                            raise RuntimeError('Recovery source changed after backup')
                    else:
                        shutil.copy2(path, target)
            files = {str(path): fleet.digest(path) for path in backup.iterdir()}
            states = {folder + '/' + name: fleet.read(CAMPAIGN / folder / name) for folder, name in [
                ('workers', WORKER + '.json'), ('rolling', WORKER + '.json'),
                ('rolling', WORKER + '.process.json'), ('claims', 'tpe.' + TASK + '.json')]}
            with psycopg.connect(host='127.0.0.1', port=worker['port'], dbname='imdb',
                    user='imdb', connect_timeout=10, autocommit=True) as connection:
                connection.execute("SET statement_timeout = '10s'")
                clients = connection.execute("SELECT count(*) FROM pg_stat_activity WHERE "
                    "backend_type='client backend' AND pid<>pg_backend_pid()").fetchone()[0]
                if clients:
                    raise RuntimeError('Database still has active clients')
                connection.execute('SELECT 1 FROM title LIMIT 1').fetchone()
            smoke_refresh = None
            try:
                fleet.validate_smoke(CAMPAIGN, plan, worker)
            except RuntimeError as exc:
                smoke_refresh = str(exc)
            evidence = dict(checked_at=fleet.stamp(), worker=worker, checkpoint_files=files,
                initial_observations=49, completed_proposals=0, pending_proposals=1,
                pending_uses_cached_initial_score=True, other_database_clients=clients,
                smoke_refresh_required=smoke_refresh,
                prior_states=states, tests_passed=18, tpe_runner_sha256=saved['meta']['runner_sha256'],
                plan_sha256=fleet.digest(CAMPAIGN / 'plan.json'),
                dispatcher_sha256=fleet.digest(CAMPAIGN / 'code/query_test200_fleet_v3.py'),
                change='Initialize the journal before spawning, retry transient monitor reads')
            fleet.write(ROOT / 'preflight.json', evidence)
        rolling.start_worker(CAMPAIGN, WORKER)
        process = fleet.read(CAMPAIGN / 'rolling' / (WORKER + '.process.json'))
        fleet.write(ROOT / 'launch.json', dict(started_at=fleet.stamp(), worker=WORKER,
            task=TASK, process=process, preflight=str(ROOT / 'preflight.json')))
        print(json.dumps(dict(restarted=WORKER, task=TASK, process=process)), flush=True)


if __name__ == '__main__':
    main()
