import fcntl,json,shutil,sys,time
from pathlib import Path
R=Path('/root/data/cys/baseline/campaigns/query_our_fsbo_vae80_test200_20261117');O=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'operations/recovery'));import recovery
from query_protocol import atomic,digest
def main():
 lock=(R/'operations/recovery/tick.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert (R/'operations/recovery/STOP').exists()
 vals=[json.loads((O/f'{h}_validation.json').read_text()) for h in ['local','remote']]
 assert all(v['passed'] for v in vals) and sum(len(v['databases']) for v in vals)==24
 assert json.loads((O/'journal_backup.json').read_text())['total_committed_measurements']==20604
 assert json.loads((R/'operations/ram_migration_20261118/remote_cache.json').read_text())['stage']=='remote_cache_verified'
 assert json.loads((O/'state.json').read_text())['stage']=='local_migration_verified'
 pins=json.loads((O/'batch_code_pins.json').read_text())
 for p,h in pins.items():
  if p!=str(R/'resources.json'):assert digest(p)==h,'Unexpected code/model change: '+p
 probes=[recovery.probe('LLMOptimizer2'),recovery.host_call('LLMOptimizer4','probe')]
 folder=R/'operations/recovery/incidents'/f'ram_storage_migration_{time.time_ns()}';folder.mkdir(parents=True)
 audit=recovery.audit_journals(folder,probes)
 errors=[]
 for p in (R/'orpo_batch10/runs').glob('*/error.json'):
  e=json.loads(p.read_text());text=e.get('error','')
  assert p.parent.name in ['CEB_11A121','CEB_11A122','CEB_11A123']
  assert ('could not read block' in text) and any(s in text for s in ['DataCorrupted','QueryInfrastructureRetryExhausted'])
  dest=folder/'reviewed_storage_errors'/p.parent.name/'error.json';dest.parent.mkdir(parents=True)
  shutil.copy2(p,dest);errors.append(dict(path=str(p),error=e,archive=str(dest)))
 boundary=dict(at=time.time(),reason='User requested maximum practical RAM use to reduce storage errors',measurements_before=20604,
  local_ram_databases=6,remote_ram_databases=14,remote_lustre_databases_with_10GB_preloaded_cache=4,
  local_spill='RAM in each new data directory',remote_spill='Lustre remains: remote tmpfs only about 10GiB free',
  preserved_sources=json.loads((O/'state.json').read_text())['completed'],validation=vals,journal_audit=audit,reviewed_errors=errors,
  experiment_effect='Storage/cache changes affect runtime. SQL, models, proposals, budgets and saved scores are preserved; later timings must disclose this boundary.',
  durability='Original database copies retained on Lustre. Scores and proposal journals remain durable on Lustre and are backed up locally. RAM data and preloaded caches are volatile.')
 atomic(folder/'storage_remediation.json',boundary);atomic(R/'evidence/ram_storage_boundary_20261118.json',boundary);atomic(O/'storage_boundary.json',boundary)
 cfg=json.loads((R/'resources.json').read_text());cfg['ram_storage_migration']=dict(at=boundary['at'],evidence=str(R/'evidence/ram_storage_boundary_20261118.json'),local_base_and_spill='tmpfs',remote_disk_cache='10GB per disk-backed instance; tables, indexes and TOAST preloaded')
 atomic(R/'resources.json',cfg)
 pins[str(R/'resources.json')]=digest(R/'resources.json');atomic(R/'evidence/batch_code_pins.json',pins)
 recovery.verify_pins()
 for e in errors:Path(e['path']).unlink()
 pause=R/'PAUSE';st=pause.stat()
 atomic(R/'operations/recovery/recovery_pause.json',dict(at=time.time(),pause_inode=st.st_ino,pause_observed_at=st.st_mtime,reason='Reviewed RAM storage remediation; resume explicitly authorized',incident=str(folder)))
 (R/'operations/recovery/STOP').unlink()
 fcntl.flock(lock,fcntl.LOCK_UN)
 print('VALIDATED_AND_RELEASED_FOR_RECOVERY',flush=True)
 result=recovery.tick();atomic(O/'resume.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
