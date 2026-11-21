import unittest,json
from unittest.mock import patch
from query_data import *
from query_context import sample
from rl_common import select_training_members,stop_sampling
class QueryContract(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.task=(SOURCE/'train800.txt').read_text().split()[0];cls.h=json.loads((ROOT/'assets/histories'/f'{cls.task}.json').read_text());cls.valid=[r for r in cls.h['records'] if r['first_occurrence']]
 def test_invalid_plan_never_calls_oracle(self):
  result=sample('{"plan":[999,999,999,999,99]}',self.task,[],self.valid,[],lambda p:self.fail('Invalid plan evaluated'))
  self.assertEqual(result['evaluation_status'],'PARSE_FAILURE')
 def test_duplicate_never_calls_oracle(self):
  r=self.valid[0];result=sample(canonical(dict(plan=r['plan'])),self.task,[r],self.valid,[],lambda p:self.fail('Duplicate evaluated'))
  self.assertEqual(result['evaluation_status'],'DUPLICATE');self.assertEqual(result['reward']['reward'],-1.)
 def test_censored_bound_is_never_improvement(self):
  plan=self.valid[0]['plan'];result=sample(canonical(dict(plan=plan)),self.task,[],self.valid,[],lambda p:dict(objective=.0000001,censored=True,status='timeout'))
  self.assertEqual(result['evaluation_status'],'CENSORED');self.assertNotIn('loss',result);self.assertEqual(result['reward']['delta_best'],0.)
 def test_causal_history_and_top_ten_contract(self):
  rows=self.h['records'][:79];messages,ctx=build_prompt(self.task,rows);self.assertTrue(all(r['observation_index']<79 for k in ['good','bad','recent50'] for r in ctx[k]));self.assertEqual(ctx['current_best_runtime'],min(r['runtime'] for r in rows if r['first_occurrence']))
 def test_adaptive_four_then_late_improvement(self):
  rows=[dict(sample_i=i,evaluation_status='SUCCESS',completion='x',reward=dict(reward=float(i),delta_best=0.),loss=5.,prefix_best_loss=4.) for i in range(4)]
  self.assertFalse(stop_sampling(rows));rows.append(dict(sample_i=4,evaluation_status='SUCCESS',completion='y',reward=dict(reward=5.,delta_best=1.),loss=3.,prefix_best_loss=4.));self.assertTrue(stop_sampling(rows));pick=select_training_members(rows,'g',1);self.assertEqual(len(pick['train_members']),4);self.assertIn(4,pick['train_members']);self.assertEqual(pick,select_training_members(rows,'g',1))
if __name__=='__main__':unittest.main()
