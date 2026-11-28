#!/usr/bin/env python3
"""Continue the pinned SemanticOPT query evaluator under an audited 800 cap."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent/'semanticopt_query_400_pilot800_20261123'
POLICY = ROOT/'control/query_budget_800.json'


def install(evaluate, policy):
    original_load = evaluate.load
    original_connect = evaluate.connect
    original_status = evaluate.status
    amendment_sha = hashlib.sha256(POLICY.read_bytes()).hexdigest()

    def amended_load(path):
        value = original_load(path)
        if Path(path).resolve() == (ROOT/'extension.json').resolve():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != policy['base_extension_sha256']:
                raise ValueError('Base extension changed')
            value = dict(value, default_budget=policy['target_budget'],
                         pilot_budget=policy['target_budget'])
        return value

    evaluate.load = amended_load

    def amended_connect(directory):
        db = original_connect(directory)
        key = 'operational_budget_amendment_800'
        old = evaluate.get(db,key)
        count = old['retained_records'] if old else db.execute('SELECT count(*) FROM records').fetchone()[0]
        digest = hashlib.sha256()
        rows = db.execute('SELECT n,value FROM records WHERE n<? ORDER BY n',(count,)).fetchall()
        if [n for n,_ in rows] != list(range(count)):
            raise ValueError('Saved measurement cursor is not contiguous')
        for n,value in rows:
            digest.update(str(n).encode()+bytes([0])+value.encode()+bytes([0]))
        prefix_sha = digest.hexdigest()
        if old and (old['prefix_sha256'] != prefix_sha or old['amendment_sha256'] != amendment_sha):
            raise ValueError('Retained measurements or amendment changed')
        if not old:
            with db:
                evaluate.put(db,key,dict(retained_records=count,prefix_sha256=prefix_sha,
                                         amendment_sha256=amendment_sha,target_budget=800))
        return db

    def amended_status(path, **values):
        original_status(path,operational_budget_amendment_sha256=amendment_sha,
                        effective_proposal_budget=800,**values)

    evaluate.connect = amended_connect
    evaluate.status = amended_status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task')
    parser.add_argument('--port',type=int,required=True)
    args = parser.parse_args()
    policy = json.loads(POLICY.read_text())
    extension = json.loads((ROOT/'extension.json').read_text())
    if (policy['target_budget'] != 800 or policy['batch_size'] != 10 or
            policy['task_count'] != len(extension['tasks']) or args.task not in extension['tasks']):
        raise ValueError('Invalid continuation policy or task')
    sys.path.insert(0,str(ROOT/'code'))
    import evaluate
    install(evaluate,policy)
    evaluate.run('query',args.task,args.port)


if __name__ == '__main__':
    main()
