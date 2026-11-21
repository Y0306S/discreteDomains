#!/usr/bin/env python3
"""Check recorded source hashes and required migration entry points."""
import hashlib
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
REQUIRED=[
    '/root/data/logs/peptide_target_loss_gate_20round_max2_test10_20261119/code/evaluate.py',
    '/root/data/logs/peptide_target_loss_gate_20round_max2_test10_20261119/code/target_gate.py',
    '/root/data/logs/peptide_target_loss_gate_20round_max2_test10_20261119/code/retrain_schedule.py',
    '/root/data/cys/baseline/campaigns/query_our_fsbo_vae80_test200_20261117/batch_code/run.py',
    '/root/data/cys/baseline/campaigns/query_our_fsbo_vae80_test200_20261117/training/train.py',
    '/root/data/cys/llmoptimizer_benchmark/tools/query_fsbo_model.py',
    '/root/data/cys/BOLT/optimization/peptides/uniref_vae/transformer_vae_unbounded.py',
    '/root/data/cys/logs/query_train800_author_fixed_20261004/native/vae/model.py',
    '/root/data/cys/LLMOptimizer/LlamaFactory/src/llamafactory/model/loader.py',
    '/root/data/cys/baseline/repos/FSBO/fsbo_modules.py',
]


def main():
    inventory=json.loads((REPO/'migration/inventory/sources.json').read_text())
    failures=[]
    for path,record in inventory.items():
        snapshot=REPO/record['snapshot']
        if not snapshot.is_file() or hashlib.sha256(snapshot.read_bytes()).hexdigest()!=record['sha256']:
            failures.append(path)
    failures.extend(path for path in REQUIRED if path not in inventory)
    for env in json.loads((REPO/'environments/index.json').read_text()):
        if env.get('error') or not (REPO/'environments'/env['name']/'installed.json').exists():
            failures.append('environment: '+env['name'])
    if failures:
        raise SystemExit('Verification failed:\n'+'\n'.join(failures))
    print(f'PASS: {len(inventory)} source hashes, {len(REQUIRED)} required entry points, '
          '9 Python environment exports and PostgreSQL specification.')


if __name__=='__main__':
    main()
