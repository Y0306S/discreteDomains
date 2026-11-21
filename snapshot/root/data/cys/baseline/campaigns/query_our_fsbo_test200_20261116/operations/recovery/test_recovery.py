import json,os,pickle,sqlite3,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import recovery as r

class DecisionTests(unittest.TestCase):
 def decide(self,**kw):
  args=dict(controller_alive=True,status=dict(status='running',at=100),paused=False,errors=[],services_ok=True,worker_count=21,expected_count=21,now=110)
  args.update(kw);return r.classify(**args)
 def test_healthy_does_not_restart(self):self.assertEqual(self.decide(),'healthy')
 def test_remote_worker_loss_recovers(self):self.assertEqual(self.decide(worker_count=20),'recover')
 def test_dead_environment_recovers(self):self.assertEqual(self.decide(controller_alive=False,services_ok=False,worker_count=0),'recover')
 def test_manual_pause_is_honored_even_after_controller_exits(self):self.assertEqual(self.decide(controller_alive=False,paused=True),'held_manual_pause')
 def test_recovery_owned_pause_resumes_after_failed_drain(self):self.assertEqual(self.decide(controller_alive=False,paused=True,managed_pause=True,worker_count=0),'recover')
 def test_single_deep_probe_timeout_does_not_cancel_active_workers(self):self.assertEqual(self.decide(services_ok=False),'degraded_health')
 def test_missing_worker_still_recovers_after_probe_timeout(self):self.assertEqual(self.decide(services_ok=False,worker_count=20),'recover')
 def test_unknown_errors_hold(self):self.assertEqual(self.decide(errors=[dict(error="ValueError('Wrong model identity')")]),'held_unknown_error')
 def test_known_transport_recovers(self):self.assertEqual(self.decide(paused=True,errors=[dict(error="RemoteDisconnected('closed')")]),'recover')
 def test_model_loading_gets_time(self):self.assertEqual(self.decide(status=dict(status='loading_inference',at=100),services_ok=False,worker_count=0),'loading')
 def test_completed_run_stays_complete(self):self.assertEqual(self.decide(status=dict(status='complete')),'complete')
 def test_pin_changes_prevent_recovery(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'evidence').mkdir();p=root/'pinned.py';p.write_text('changed')
   (root/'evidence/batch_code_pins.json').write_text(json.dumps({str(p):'old'}))
   with patch.object(r,'ROOT',root):
    with self.assertRaisesRegex(RuntimeError,'Pinned dependency changed'):r.verify_pins()

class AuditTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.folder=self.root/'incident';self.folder.mkdir()
  self.path=self.root/'warm30/runs/CEB_TEST/runtime.sqlite';self.path.parent.mkdir(parents=True)
  self.plan=[1,1,2,1,2];self.original=dict(status='started',argument=dict(plan=self.plan),at=1)
  with sqlite3.connect(self.path) as db:
   db.execute('CREATE TABLE kv(key TEXT PRIMARY KEY,value BLOB)')
   for k,v in [('warm_state',dict(rows=[dict(plan=self.plan)])),('measurement/1',self.original),('measurement/0',dict(status='complete',argument=dict(plan=[9]),result=dict(runtime=3)))]:db.execute('INSERT INTO kv VALUES (?,?)',(k,pickle.dumps(v)))
  self.probes=[dict(workers=[dict(alive=False)],databases=[dict(ready=True,active=[])],unrecorded_workers=[])]
 def tearDown(self):self.temp.cleanup()
 def test_terminal_unscored_select_keeps_plan_and_original_backup(self):
  with patch.object(r,'ROOT',self.root),patch('query_data.canonicalize',side_effect=lambda t,p,strict:p),patch('query_data.plan_key',side_effect=str):result=r.audit_journals(self.folder,self.probes)
  self.assertEqual(result['audited_operations'],1)
  with sqlite3.connect(self.path) as db:entry=pickle.loads(db.execute('SELECT value FROM kv WHERE key=?',('measurement/1',)).fetchone()[0]);saved=pickle.loads(db.execute('SELECT value FROM kv WHERE key=?',('measurement/0',)).fetchone()[0])
  self.assertEqual(entry['argument'],self.original['argument']);self.assertEqual(entry['status'],'safe_failed');self.assertNotIn('result',entry);self.assertEqual(saved['result']['runtime'],3)
  with sqlite3.connect(self.folder/'journals/warm30/CEB_TEST/runtime.sqlite') as db:old=pickle.loads(db.execute('SELECT value FROM kv WHERE key=?',('measurement/1',)).fetchone()[0])
  self.assertEqual(old,self.original)
 def test_active_sql_prevents_replay_authorization(self):
  self.probes[0]['databases'][0]['active']=[dict(pid=42)]
  with patch.object(r,'ROOT',self.root):
   with self.assertRaises(AssertionError):r.audit_journals(self.folder,self.probes)
 def test_live_worker_prevents_replay_authorization(self):
  self.probes[0]['workers'][0]['alive']=True
  with patch.object(r,'ROOT',self.root):
   with self.assertRaises(AssertionError):r.audit_journals(self.folder,self.probes)
 def test_unrecorded_worker_prevents_replay_authorization(self):
  self.probes[0]['unrecorded_workers']=[42]
  with patch.object(r,'ROOT',self.root):
   with self.assertRaises(AssertionError):r.audit_journals(self.folder,self.probes)

class DrainTests(unittest.TestCase):
 def test_worker_retouch_does_not_reset_owned_pause_age(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);p=root/'PAUSE';p.touch()
   (root/'recovery_pause.json').write_text(json.dumps(dict(pause_inode=p.stat().st_ino,pause_observed_at=0)))
   with patch.object(r,'ROOT',root),patch.object(r,'OPS',root):self.assertGreater(r.pause_age(),290)
 def test_replaced_pause_does_not_inherit_recovery_age(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);p=root/'PAUSE';p.touch()
   (root/'recovery_pause.json').write_text(json.dumps(dict(pause_inode=p.stat().st_ino+1,pause_observed_at=0)))
   with patch.object(r,'ROOT',root),patch.object(r,'OPS',root):self.assertLess(r.pause_age(),290)
 def test_generation_worker_can_only_stop_after_old_pause_and_idle_sql(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);p=root/'PAUSE';p.touch();os.utime(p,(0,0))
   state=dict(workers=[dict(alive=True,record=dict(pid=999,start_ticks='1'))],databases=[dict(ready=True,active=[])],unrecorded_workers=[])
   with patch.object(r,'ROOT',root),patch.object(r,'probe',return_value=state),patch.object(r,'record_alive',return_value=True),patch('recovery.os.kill') as kill:
    self.assertEqual(r.drain_generation('LLMOptimizer2')['stopped_generation_workers'],[999]);kill.assert_called_once()
 def test_active_sql_blocks_signals(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);p=root/'PAUSE';p.touch();os.utime(p,(0,0))
   state=dict(workers=[],databases=[dict(ready=True,active=[dict(pid=4)])],unrecorded_workers=[])
   with patch.object(r,'ROOT',root),patch.object(r,'probe',return_value=state),patch('recovery.os.kill') as kill:
    with self.assertRaises(AssertionError):r.drain_generation('LLMOptimizer2')
    kill.assert_not_called()
 def test_new_pause_blocks_signals(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'PAUSE').touch()
   with patch.object(r,'ROOT',root),patch('recovery.os.kill') as kill:
    with self.assertRaises(AssertionError):r.drain_generation('LLMOptimizer2')
    kill.assert_not_called()

if __name__=='__main__':unittest.main()
