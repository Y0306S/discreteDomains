import collections
import datetime
import json
import pickle
import sqlite3
from pathlib import Path

ROOT = Path('/root/data/cys/baseline/campaigns/query_our_fsbo_vae80_test200_20261117')
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
sql = collections.defaultdict(collections.Counter)
generation = collections.defaultdict(collections.Counter)
reasons = collections.defaultdict(collections.Counter)
tasks = []
errors = []
for path in sorted((ROOT/'orpo_batch10/runs').glob('*/runtime.sqlite')):
    try:
        db = sqlite3.connect(f'file:{path}?mode=ro', uri=True, timeout=30)
        db.execute('BEGIN')
        proposals = {}
        for key, blob in db.execute("SELECT key,value FROM kv WHERE key LIKE 'proposal_batch/%'"):
            record = pickle.loads(blob)
            if record['status'] != 'complete':
                continue
            batch = record['result']
            proposals[int(key.split('/')[1])] = batch
            counts = collections.Counter(x['actual_arm'] for x in batch['items'])
            generation['llm']['accepted'] += counts['llm']
            generation['random']['accepted'] += counts['random']
            generation['random']['attempts'] += batch.get('random_draws', 0)
            for rejected in batch['rejected_generations']:
                arm = rejected.get('source', 'llm')
                reasons[arm][rejected['reason']] += 1
                if arm == 'llm':
                    generation['llm']['rejected'] += 1
                else:
                    generation['fsbo']['outer_rejected'] += 1
            diag = batch.get('fsbo_diagnostics')
            if diag:
                generation['fsbo']['accepted'] += diag['accepted']
                generation['fsbo']['rejected'] += len(diag['rejected_decodes'])
                for rejected in diag['rejected_decodes']:
                    reasons['fsbo_decode'][rejected['error']] += 1
        task_sql = collections.defaultdict(collections.Counter)
        pending = collections.Counter()
        for key, blob in db.execute("SELECT key,value FROM kv WHERE key LIKE 'measurement/%'"):
            parts = key.split('/')
            if len(parts) != 3:
                errors.append([path.parent.name, key, 'unexpected measurement key'])
                continue
            record = pickle.loads(blob)
            if record['status'] != 'complete':
                pending[record['status']] += 1
                continue
            batch = proposals[int(parts[1])]
            item = batch['items'][int(parts[2])]
            assert item['plan'] == record['argument']['plan']
            result = record['result']
            arm = item['actual_arm']
            for counter in (sql[arm], task_sql[arm]):
                counter['total'] += 1
                counter['feasible' if not result['censored'] else 'censored'] += 1
                counter['status:'+result['status']] += 1
        db.rollback()
        db.close()
        tasks.append(dict(task=path.parent.name, sql=dict(task_sql), pending=pending,
                          complete=(path.parent/'complete.json').exists()))
    except Exception as exc:
        errors.append([str(path), repr(exc)])

for arm, counter in sql.items():
    counter['feasibility_percent'] = 100*counter['feasible']/counter['total']
for arm in ('llm', 'fsbo'):
    counter = generation[arm]
    counter['attempts'] = counter['accepted'] + counter['rejected']
    counter['accepted_percent'] = 100*counter['accepted']/counter['attempts'] if counter['attempts'] else None
result = dict(started_utc=started, ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              campaign=str(ROOT), sql=dict(sql), generation=dict(generation),
              rejection_reasons=dict(reasons), task_count=len(tasks),
              complete_tasks=sum(t['complete'] for t in tasks), errors=errors,
              campaign_status=json.loads((ROOT/'status.json').read_text()), tasks=tasks)
Path(__file__).with_name('audit.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('tasks','campaign_status')}, indent=2))
print('campaign_status', json.dumps(result['campaign_status']))
