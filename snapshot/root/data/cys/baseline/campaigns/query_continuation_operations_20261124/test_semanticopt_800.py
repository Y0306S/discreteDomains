import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import fleet
import semanticopt_800_task as wrapper
import semanticopt_800_handoff as handoff


class BudgetContinuationTest(unittest.TestCase):
    def test_handoff_stops_only_idle_verified_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for folder in ['control','query','slot_pauses']:(root/folder).mkdir()
            child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)',
                                    'slot','semanticopt','--slot','0'])
            try:
                record=dict(slot=0,pid=child.pid,start_ticks=handoff.ticks(child.pid))
                (root/'control/query_budget_800.json').write_text(json.dumps(dict(old_workers=[record])))
                state=root/'query/worker00.status.json'
                state.write_text('{"status":"running"}')
                (root/'slot_pauses/worker00.PAUSE').write_text('rollout')
                with patch.object(handoff,'ROOT',root),patch.object(sys,'argv',['handoff','--slot','0']):
                    with self.assertRaisesRegex(RuntimeError,'idle handoff'):
                        handoff.main()
                    self.assertIsNone(child.poll())
                    state.write_text('{"status":"paused_slot"}')
                    handoff.main()
                    child.wait(timeout=3)
                    self.assertEqual(child.returncode,-15)
            finally:
                if child.poll() is None:child.terminate();child.wait()

    def test_wrapper_retains_and_checks_measured_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);policy_path=root/'policy.json';policy_path.write_text('{}')
            extension=root/'extension.json';extension.write_text(json.dumps(dict(default_budget=400,pilot_budget=800)))
            db=sqlite3.connect(':memory:')
            db.execute('CREATE TABLE records(n INTEGER PRIMARY KEY,value TEXT)')
            db.execute('CREATE TABLE state(key TEXT PRIMARY KEY,value TEXT)')
            db.execute('INSERT INTO records VALUES(0,?)',('{"objective":1}',));db.commit()
            def get(db,key):
                r=db.execute('SELECT value FROM state WHERE key=?',(key,)).fetchone()
                return json.loads(r[0]) if r else None
            def put(db,key,value):db.execute('INSERT OR REPLACE INTO state VALUES(?,?)',(key,json.dumps(value)))
            status=[]
            evaluator=types.SimpleNamespace(load=lambda p:json.loads(Path(p).read_text()),
                connect=lambda p:db,get=get,put=put,status=lambda p,**kw:status.append(kw))
            policy=dict(target_budget=800,base_extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest())
            with patch.object(wrapper,'ROOT',root),patch.object(wrapper,'POLICY',policy_path):
                wrapper.install(evaluator,policy)
                self.assertEqual(evaluator.load(extension)['default_budget'],800)
                self.assertEqual(json.loads(extension.read_text())['default_budget'],400)
                evaluator.connect(root)
                audit=get(db,'operational_budget_amendment_800');self.assertEqual(audit['retained_records'],1)
                evaluator.connect(root)
                evaluator.status(root,status='running',budget=800)
                self.assertEqual(status[-1]['effective_proposal_budget'],800)
                db.execute('UPDATE records SET value=? WHERE n=0',('{"objective":2}',));db.commit()
                with self.assertRaisesRegex(ValueError,'Retained measurements'):
                    evaluator.connect(root)
            db.close()

    def test_amendment_expands_scheduling_and_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'control').mkdir()
            (root/'allocation.json').write_text('{"workers":[]}')
            ext=dict(tasks=['a','b'],pilot_tasks=['a'],source=str(root),source_budget=200,default_budget=400,pilot_budget=800)
            (root/'extension.json').write_text(json.dumps(ext))
            policy=dict(base_extension_sha256=hashlib.sha256((root/'extension.json').read_bytes()).hexdigest(),target_budget=800,batch_size=10,task_count=2)
            (root/'control/query_budget_800.json').write_text(json.dumps(policy))
            folder=root/'query/runs/b';folder.mkdir(parents=True)
            (folder/'status.json').write_text('{"status":"complete","evaluations":400}')
            with patch.dict(fleet.ROOTS,semanticopt=root):
                _,_,effective=fleet.configuration('semanticopt')
                self.assertEqual(effective['default_budget'],800)
                self.assertEqual(effective['additional_evaluations'],1200)
                self.assertFalse(fleet.completed('semanticopt','b',effective))
                command=fleet.task_command('semanticopt',{'optimizer_cpu':1,'db_port':123},'b')
                self.assertIn(str(fleet.OPS/'semanticopt_800_task.py'),command)
                (folder/'status.json').write_text('{"status":"complete","evaluations":800}')
                self.assertTrue(fleet.completed('semanticopt','b',effective))


if __name__=='__main__':unittest.main()
