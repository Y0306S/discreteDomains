"""Label retired machine-3 worker records without changing evaluation results."""

import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
QUERY=Path('/root/data/cys/logs/query_test200_baselines_20261012')
sys.path.insert(0,str(QUERY/'code'))
import query_test200_fleet_v3 as fleet

if not (ROOT/'LLMOptimizer3.released.json').exists():
    raise RuntimeError('Machine 3 has not been released')
for item in fleet.read(ROOT/'assignments.json'):
    name=item['before']['worker']
    assert name.startswith('LLMOptimizer3.')
    path=QUERY/'workers'/f'{name}.json'
    previous=ROOT/'retired'/f'{name}.worker_before.json'
    if not previous.exists():
        fleet.write(previous,fleet.read(path))
    old=fleet.read(previous)
    task=fleet.read(ROOT/'tasks'/f'{item["task"]}.json')
    fleet.write(path,dict(old,status='reassigned',pid=None,child_pid=None,error=None,
        saved_proposals=len(task['completed_sha256']),updated_at=fleet.stamp(),
        reassigned_to=item['after']['worker'],recovery_deployment=str(ROOT),
        note='LLMOptimizer3 reserved for user; durable task resumes on LLMOptimizer2'))
print(json.dumps({'retired_worker_records':9}))
