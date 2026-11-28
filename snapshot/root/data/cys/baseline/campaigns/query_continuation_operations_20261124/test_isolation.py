import json,os,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import fleet

class IsolationTest(unittest.TestCase):
 def test_failed_task_then_two_completed_queries_use_different_processes(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);(root/'logs').mkdir()
   specs=[dict(slot=0,host='test',optimizer_cpu=0,db_port=1)]
   tasks=['bad','good1','good2']
   script=root/'fake.py'
   script.write_text('import os,sys,json\nfrom pathlib import Path\np=Path(sys.argv[1]);t=sys.argv[2]\n(p/"pids").open("a").write(str(os.getpid())+"\\n")\nif t=="bad":sys.exit(7)\n(p/"runs"/t/"complete.json").write_text("{}")\n')
   def cmd(kind,spec,task):return [sys.executable,str(script),str(root),task]
   with patch.dict(fleet.ROOTS,bolt=root),patch.object(fleet,'configuration',return_value=(specs,tasks,{})),patch.object(fleet,'database_check',return_value={}),patch.object(fleet,'task_command',side_effect=cmd):
    fleet.slot('bolt',0)
    self.assertTrue((root/'task_pauses/bad.json').exists())
    self.assertFalse((root/'PAUSE').exists())
    self.assertTrue((root/'runs/good1/complete.json').exists())
    self.assertTrue((root/'runs/good2/complete.json').exists())
    pids=(root/'pids').read_text().splitlines();self.assertEqual(len(set(pids)),3)
 def test_dead_slot_does_not_stop_healthy_slot(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder)
   failed=subprocess.Popen([sys.executable,'-c','raise SystemExit(9)']);failed.wait()
   healthy=subprocess.Popen([sys.executable,'-c','import time;time.sleep(5)'])
   try:
    with patch.dict(fleet.ROOTS,bolt=root):
     state=fleet.summarize('bolt',['a','b'],{},[({'slot':0,'host':'test'},failed),({'slot':1,'host':'test'},healthy)],time.time())
     self.assertEqual(state['status'],'degraded');self.assertEqual(state['live_worker_slots'],1)
     self.assertFalse((root/'PAUSE').exists());self.assertIsNone(healthy.poll())
   finally:healthy.terminate();healthy.wait()
 def test_bolt_command_uses_fresh_task_interpreter(self):
  cmd=fleet.task_command('bolt',{'optimizer_cpu':3},'CEB_test')
  self.assertEqual(cmd[-5:],[str(fleet.OPS/'fleet.py'),'task','bolt','--task','CEB_test'])
if __name__=='__main__':unittest.main()
