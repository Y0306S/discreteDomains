import copy,json,random,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import method,warm
from method import Journal,build_batch,optimize,SafePause
from single_method import observed,UnresolvedMeasurement,selector,select_arm
from query_data import definition,canonicalize,plan_key,build_prompt
from random_plans import query_proposal
ROOT=Path(__file__).resolve().parents[1]
TASK='CEB_10A19'
CFG=json.loads((ROOT/'batch_recipe.json').read_text());POLICY=json.loads((ROOT/'batch_policy.json').read_text())
def measurement(runtime=2,censored=False):return dict(runtime=runtime,censored=censored,status='timeout' if censored else 'completed')
def candidates(n):
 d,c,_,_=definition(TASK);rng=random.Random(867);out={}
 while len(out)<n:
  p=canonicalize(TASK,query_proposal(d,c,rng)['plan'],strict=True);out[plan_key(p)]=p
 return list(out.values())
class BatchTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.plans=candidates(120);cls.rows=[observed(p,measurement(i+1),i,'bolt') for i,p in enumerate(cls.plans[:30])]
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.j=Journal(Path(self.tmp.name)/'run.sqlite',dict(test=self.id()))
 def tearDown(self):self.j.close();self.tmp.cleanup()
 def test_constraints_and_strict_invalid(self):
  msgs,_=build_prompt(TASK,[],sft=True);x=json.loads(msgs[1]['content'])['current_task'];self.assertTrue(x['constraints']['preserve_SQL_and_predicates']);self.assertTrue(x['constraints']['no_implicit_codec_repairs']);self.assertTrue(x['query_text']);p=list(self.plans[0]);p[4]=9
  with self.assertRaises(ValueError):canonicalize(TASK,p,strict=True)
 def test_twenty_single_calls_dynamic_prompt_and_partial_fsbo(self):
  requests=[];calls=[]
  def gen(req):
   requests.append(req);i=len(requests)
   # First duplicate and one malformed, then6 unique, then duplicates until20.
   p=self.plans[0] if i<3 or i>8 else self.plans[27+i]
   return dict(text='garbage' if i==2 else json.dumps(dict(plan=p)))
  def fsbo(rows,seed,count,excluded):calls.append((len(rows),count,excluded));return dict(plans=self.plans[50:50+count],diagnostics={})
  b=build_batch(self.j,TASK,CFG,POLICY,self.rows,9,'llm',gen,fsbo)
  self.assertEqual(len(requests),20);self.assertEqual(calls[0][:2],(30,4));self.assertEqual(len(b['items']),10)
  self.assertEqual([x['actual_arm'] for x in b['items']].count('llm'),6)
  a=json.loads(requests[1]['messages'][1]['content']);self.assertEqual(len(a['unmeasured_avoidance']['recent_rejected']),1);self.assertNotIn('runtime',str(a['unmeasured_avoidance']['recent_rejected']))
  a=json.loads(requests[3]['messages'][1]['content']);self.assertEqual(len(a['unmeasured_avoidance']['accepted_plans']),1)
  self.assertEqual(self.rows,self.__class__.rows)
 def test_fsbo_batch10_and_random_only_missing(self):
  counts=[]
  def fsbo(rows,seed,count,excluded):counts.append(count);return dict(plans=self.plans[30:36],diagnostics={})
  b=build_batch(self.j,TASK,CFG,POLICY,self.rows,10,'fsbo',None,fsbo)
  self.assertEqual(counts,[10]);self.assertEqual([x['actual_arm'] for x in b['items']].count('fsbo'),6);self.assertEqual([x['actual_arm'] for x in b['items']].count('random'),4)
  self.assertEqual(len({plan_key(x['plan']) for x in b['items']}),10)
 def test_random_exhaustion_no_sql(self):
  policy=dict(POLICY,random_draw_limit=3)
  def fsbo(*args):return dict(plans=[],diagnostics={})
  with patch('method.query_proposal',return_value=dict(plan=self.plans[0])):
   with self.assertRaises(method.UnderfilledBatch):build_batch(self.j,TASK,CFG,policy,self.rows,10,'fsbo',None,fsbo)
 def test_retrieval_round4_and_frozen_batch_resume(self):
  requests=[];measured=[];retrieval=[];paused=[False]
  # A real validated training card; target task excluded.
  from retrieval import assets
  cards,_,_=assets();card=next(v for t,v in cards.items() if t!=TASK)
  def retrieve(*args):retrieval.append(1);return dict(cards=[card],ranking=dict(confidence='weak_or_undefined_DO_NOT_ASSUME_TRANSFER'))
  def generate(req):requests.append(req);return dict(text=json.dumps(dict(plan=self.plans[29+len(requests)])))
  def fsbo(*args):raise AssertionError('Full LLM batch should skip FSBO')
  def evaluate(p):
   self.assertEqual(len(requests),10);measured.append(plan_key(p))
   if len(measured)==3:paused[0]=True
   return measurement(.5 if len(measured)==2 else 50,len(measured)!=2)
  with self.assertRaises(SafePause):optimize(self.j,TASK,CFG,POLICY,self.rows,generate,evaluate,fsbo,retrieve,stop_after=1,paused=lambda:paused[0])
  self.assertEqual(len(measured),3);self.assertEqual(self.j.get('optimization_state'),None)
  paused[0]=False
  state=optimize(self.j,TASK,CFG,POLICY,self.rows,generate,evaluate,fsbo,retrieve,stop_after=1,paused=lambda:False)
  self.assertEqual(len(measured),10);self.assertEqual(len(set(measured)),10);self.assertEqual(len(requests),10);self.assertEqual(len(retrieval),1)
  self.assertEqual(state['events'][0]['round'],4);self.assertEqual(state['selector'].n['llm'],1);self.assertEqual(len(state['rows']),40)
  context=json.loads(requests[0]['messages'][1]['content']);self.assertEqual(context['scenario'],'retrieval');self.assertIn('similar_task_evidence',context)
 def test_schedule_global_rounds(self):
  b=selector();events=[]
  for r in range(4,81):
   arm,d=select_arm(b,events,r)
   if r<=8:self.assertEqual(arm,'llm')
   else:self.assertEqual(d['schedule_index'],r-9)
   if 10<=r<=20 and events[-1]['selected_arm']=='llm':self.assertEqual(arm,'fsbo')
   b.update(arm,-r);events.append(dict(selected_arm=arm))
  self.assertEqual(len(events),77);self.assertFalse(d['cooldown_active'])
 def test_legacy_reuse_and_fresh_random_completion_resumes(self):
  old=dict(rows=copy.deepcopy(self.rows[:21]),attempts=46,rejected=[],archive=[],legacy_sha256='test')
  calls=[];evaluated=[]
  def gen(req):calls.append(req);return dict(text=json.dumps(dict(plan=self.plans[0])))
  def ev(p):evaluated.append(p);return measurement()
  with patch('warm.legacy',return_value=old):state=warm.warm_start(self.j,TASK,CFG,gen,ev,pool_loader=lambda task:self.plans[30:60])
  self.assertEqual(len(calls),4);self.assertEqual(len(evaluated),9);self.assertEqual(len(state['rows']),30)
  self.assertEqual(sum(r['source_phase']=='warm_random_previous_BO' for r in state['rows']),9)
  warm.warm_start(self.j,TASK,CFG,gen,ev,pool_loader=lambda task:[]);self.assertEqual(len(evaluated),9)
 def test_new_bolt30_single_calls_and_no_old_runtime_import(self):
  calls=[];sql=[]
  def gen(req):calls.append(req);return dict(text=json.dumps(dict(plan=self.plans[len(calls)-1])))
  def ev(p):sql.append(p);return measurement()
  old=dict(rows=[],attempts=0,rejected=[],archive=[],legacy_sha256=None)
  with patch('warm.legacy',return_value=old):state=warm.warm_start(self.j,TASK,CFG,gen,ev,pool_loader=lambda task:[])
  self.assertEqual(len(state['rows']),30);self.assertEqual(len(calls),30);self.assertEqual(len(sql),30)
  self.assertTrue(all(r['provenance']['temperature']==.9 for r in state['rows']))
 def test_mixed_batch_credits_fsbo_and_labels_random(self):
  self.j.put('optimization_state',dict(rows=copy.deepcopy(self.rows),events=[dict(selected_arm='llm') for _ in range(5)],selector=selector()))
  def gen(req):return dict(text=json.dumps(dict(plan=self.plans[0])))
  def fsbo(*args):return dict(plans=self.plans[30:36],diagnostics={})
  state=optimize(self.j,TASK,CFG,POLICY,self.rows,gen,lambda p:measurement(),fsbo,stop_after=6)
  e=state['events'][-1];self.assertEqual(e['round'],9);self.assertEqual(state['selector'].n,dict(llm=0,fsbo=1));self.assertEqual(state['selector'].attempt_n,dict(llm=1,fsbo=1));self.assertFalse(e['arm_updates']['random']['credited_to_selector']);self.assertEqual(e['arm_updates']['random']['measurements'],4)
 def test_interrupted_sql_never_replayed(self):
  calls=[]
  def fail():calls.append(1);raise RuntimeError('ambiguous transport loss')
  with self.assertRaises(RuntimeError):self.j.once('sql',{},fail,measurement=True)
  with self.assertRaises(UnresolvedMeasurement):self.j.once('sql',{},fail,measurement=True)
  self.assertEqual(len(calls),1)
if __name__=='__main__':unittest.main()
