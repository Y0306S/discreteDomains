import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from method import *
from query_resource_recovery import QueryInfrastructureRetryExhausted

class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'run.sqlite';self.j=Journal(self.path,dict(task='train'))
  self.cfg=dict(seed=1,warm_start=dict(count=2,maximum_samples=5),optimization=dict(attempts=2,llm_generation_limit=2))
  self.pool=[observed([1],dict(runtime=3,censored=False,status='completed'),0,'initialization'),observed([2],dict(runtime=50,censored=True,status='timeout'),1,'initialization')]
 def tearDown(self):self.j.close();self.tmp.cleanup()
 def test_no_ambiguous_measurement_replay(self):
  self.j.put('m',dict(status='started',argument={'plan':[1]}))
  with self.assertRaises(UnresolvedMeasurement):self.j.once('m',{'plan':[1]},lambda:3,measurement=True)
 def test_safe_exhaustion_replays_same_proposal(self):
  def fail():raise QueryInfrastructureRetryExhausted([dict(error='connect failed')])
  with self.assertRaises(QueryInfrastructureRetryExhausted):self.j.once('m',1,fail,measurement=True)
  self.assertEqual(self.j.once('m',1,lambda:3,measurement=True),3)
  with self.assertRaises(ValueError):self.j.once('m',2,lambda:4,measurement=True)
 def test_durable_result_not_remeasured(self):
  self.assertEqual(self.j.once('m',1,lambda:7,measurement=True),7)
  self.j.close();self.j=Journal(self.path,dict(task='train'))
  self.assertEqual(self.j.once('m',1,lambda:1/0,measurement=True),7)
 def test_warm_duplicate_and_invalid_never_evaluated(self):
  replies=iter(['bad','1','1','2']);evaluated=[]
  def parse(text,task):
   if text=='bad':raise ValueError('bad')
   return [int(text)]
  def evaluate(p):evaluated.append(p);return dict(runtime=2,censored=False,status='completed')
  with patch('method.parse_action',parse):s=warm_start(self.j,'train',self.cfg,lambda r:dict(text=next(replies)),evaluate)
  self.assertEqual(evaluated,[[1],[2]]);self.assertEqual(s['attempts'],4)
  self.assertEqual(warm_start(self.j,'train',self.cfg,lambda r:1/0,lambda p:1/0)['rows'],s['rows'])
 def test_underfilled_no_random_fill(self):
  with patch('method.parse_action',side_effect=ValueError('invalid')):
   with self.assertRaises(IncompletePool):warm_start(self.j,'train',self.cfg,lambda r:dict(text='bad'),lambda p:1/0)
 def test_llm_exhaustion_uses_fsbo_and_timeout_no_incumbent(self):
  cfg=copy.deepcopy(self.cfg);cfg['optimization']['attempts']=1
  calls=[]
  with patch('method.parse_action',lambda text,task:json.loads(text)['plan'] if text.startswith('{') else [int(text)]),patch('method.build_prompt',return_value=([],None)):
   s=optimize(self.j,'train',cfg,self.pool,lambda r:dict(text='1'),lambda p:dict(runtime=50,censored=True,status='timeout'),lambda rows,seed:dict(plan=[3],diagnostics={'checked':True}))
  self.assertEqual(s['events'][0]['actual_arm'],'fsbo');self.assertEqual(best(s['rows']),3)
  self.assertEqual(s['events'][0]['best_runtime'],3);self.assertEqual(len(s['rows']),3)
 def test_no_fsbo_plan_consumes_attempt_without_measurement(self):
  cfg=copy.deepcopy(self.cfg);cfg['optimization']['attempts']=1
  with patch('method.parse_action',side_effect=ValueError('bad')),patch('method.build_prompt',return_value=([],None)):
   s=optimize(self.j,'train',cfg,self.pool,lambda r:dict(text='bad'),lambda p:1/0,lambda rows,seed:dict(plan=None,diagnostics={}))
  self.assertEqual(s['events'][0]['status'],'NO_VALID_PROPOSAL');self.assertEqual(len(s['rows']),2)
 def test_resume_optimizer_does_not_redo_first_attempt(self):
  with patch('method.parse_action',lambda text,task:json.loads(text)['plan'] if text.startswith('{') else [int(text)]),patch('method.build_prompt',return_value=([],None)):
   evaluate=lambda p:dict(runtime=1,censored=False,status='completed')
   s=optimize(self.j,'train',self.cfg,self.pool,lambda r:dict(text=str(r['attempt']+2)),evaluate,lambda *x:1/0,stop_after=1)
   self.assertEqual(len(s['events']),1)
   s=optimize(self.j,'train',self.cfg,self.pool,lambda r:dict(text=str(r['attempt']+2)),evaluate,lambda *x:1/0)
   self.assertEqual(len(s['events']),2);self.assertEqual(len(s['rows']),4)
 def test_plan_key_independent_order(self):self.assertEqual(plan_key([1,1,2,1,2,3,1,4,1,2]),plan_key([3,1,4,1,2,1,1,2,1,2]))
 def test_selector_warm_does_not_count_and_cooldown(self):
  b=selector()
  self.assertEqual(select_arm(b,[],1)[0],'llm');self.assertEqual(select_arm(b,[dict(selected_arm='llm')],10)[0],'fsbo')
 def test_changed_checkpoint_identity_rejected(self):
  with self.assertRaises(ValueError):Journal(self.path,dict(task='other'))

if __name__=='__main__':unittest.main()
