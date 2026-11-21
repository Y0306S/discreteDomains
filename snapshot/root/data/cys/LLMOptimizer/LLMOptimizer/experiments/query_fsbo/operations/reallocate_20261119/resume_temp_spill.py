"""Audit the terminal spill read failure, then continue with all 20 LLMs."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OPS = Path(__file__).resolve().parent
sys.path.insert(0, str(OPS))
import eight_a16_transition as transition
from query_protocol import atomic
import recovery
import resources

assert (ROOT / 'PAUSE').exists()
assert not recovery.process_matches(transition.load(ROOT / 'services/batch_controller.process.json'))
probes = transition.probes()
assert all(not w['alive'] for p in probes for w in p['workers'])
folder = OPS / 'temp_spill_io' / str(time.time_ns())
folder.mkdir()
audit = recovery.audit_journals(folder, probes)
assert audit['audited_operations'] == 1, audit
record = transition.load(folder / 'journal_audit.json')['operations'][0]
assert '/CEB_2B291/' in record['journal'], record
error_path = ROOT / 'orpo_batch10/runs/CEB_2B291/error.json'
error = transition.load(error_path)
assert recovery.error_allowed(error['error'], error['traceback'])
error_path.rename(folder / 'error.json')
assert not list((ROOT / 'orpo_batch10/runs').glob('*/error.json'))
assert len(resources.config()['llm']) == 20
resources.health()
recovery.verify_pins()
atomic(OPS / 'temp_spill_recovery.json', dict(at=time.time(), audit=audit, incident=str(folder),
       disposition='One terminal unscored SELECT is replayable; retain the exact selected plan.'))
(ROOT / 'PAUSE').unlink()
(ROOT / 'operations/recovery/recovery_pause.json').unlink(missing_ok=True)
pid = resources.start([resources.PYTHON, '-u', str(ROOT / 'batch_code/run.py'), 'controller'], 'batch_controller')
atomic(OPS / 'eight_a16_transition.json', dict(phase='running_all_eight_a16', at=time.time(),
       controller_pid=pid, live_LLMs=20, spill_recovery=str(folder)))
print('Resumed controller', pid, 'with 20 LLMs; one audited SQL operation may be retried.', flush=True)
