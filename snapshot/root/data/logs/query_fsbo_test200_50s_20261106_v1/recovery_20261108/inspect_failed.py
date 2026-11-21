"""Verify a decode-only failure has no unmeasured database query before resuming."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, '/root/data/cys/llmoptimizer_benchmark/tools')
import query_fsbo_eval as evaluation
import torch
from run_artifact_store import JsonJournal

task = 'CEB_7A102'
run = ROOT/'runs'/task
evidence = ROOT/'recovery_20261108/CEB_7A102_before.json'
assert not evidence.exists(), 'Recovery already recorded; inspect it before repeating'
with (ROOT/'locks'/(task+'.lock')).open('a') as handle:
    fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
    recipe = evaluation.verify(ROOT)
    evaluation.peptide_turbo(recipe)
    saved = torch.load(run/'checkpoint.pt', weights_only=False, map_location='cpu')
    status = json.loads((run/'status.json').read_text())
    assert status['status']=='failed' and status['error']=='timed out'
    assert status['unresolved_database_calls']==0
    assert len(saved['completed'])==220 and saved['pending'] is None
    assert saved['total']==1049
    journal = JsonJournal(run)
    assert not journal.names('proposal_*.pending.json')
    committed = journal.names('proposal_*.json')
    committed = [n for n in committed if not n.endswith(('.pending.json','.database_attempts.json'))]
    assert len(committed)==220
    with sqlite3.connect(f'file:{run}/artifacts.sqlite3?mode=ro', uri=True) as connection:
        checks = dict(connection.execute('SELECT name,sha256 FROM artifacts'))
    log = (ROOT/'logs/w19.log').read_text()
    assert 'backend.decode' in log and 'query_vae_client.py' in log and 'TimeoutError: timed out' in log
    report = dict(passed=True, at=time.time(), task=task, status_before=status,
        checkpoint_sha256=evaluation.digest(run/'checkpoint.pt'),
        recipe_sha256=evaluation.digest(ROOT/'recipe.json'),
        committed_proposals=220, pending_database_proposals=0,
        all_artifact_hashes=checks, completed_checkpoint_rows=220,
        reason='VAE HTTP decoding timed out before the next decoded batch was saved or any SQL call began. Resume restores the previous committed checkpoint and RNG.',
        inference_timeout_seconds=1800, sql_timeout_seconds=50,
        physical_database_replay=False)
    evaluation.atomic_json(evidence, report)
    status.update(status='paused', error=None, recovery_evidence=str(evidence))
    evaluation.atomic_json(run/'status.json',status)
    print(json.dumps({k:v for k,v in report.items() if k not in ['status_before','all_artifact_hashes']},indent=2))
