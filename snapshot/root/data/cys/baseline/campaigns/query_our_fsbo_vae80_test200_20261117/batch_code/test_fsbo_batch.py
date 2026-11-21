import unittest,types
import numpy as np
import torch
from backends import FSBO
from query_data import plan_key
import test_batch
from test_batch import TASK,measurement
from single_method import observed
class FSBOTests(unittest.TestCase):
 def test_resume_encodes_large_history_in_bounded_ordered_batches(self):
  calls=[];fits=[]
  class FitReached(Exception):pass
  class VAE:
   def encode(self,plans):
    self_test.assertLessEqual(len(plans),512);calls.append(plans)
    return np.asarray([[float(p[0])]*64 for p in plans])
  class Model:
   def request(self,op,**values):fits.append((op,values));raise FitReached()
  self_test=self;f=FSBO.__new__(FSBO);f.q=types.SimpleNamespace(normalized=lambda x,c:x);f.cfg={};f.encodings={};f.vae=VAE();f.model=Model()
  rows=[dict(plan=[i],runtime=i+1,censored=False) for i in range(1025)]
  rows.append(dict(plan=[2000],runtime=9999,censored=True))
  with self.assertRaises(FitReached):f(rows,123)
  self.assertEqual([len(p) for p in calls],[512,512,1])
  self.assertEqual([p for chunk in calls for p in chunk],[r['plan'] for r in rows[:-1]])
  values=fits[0][1];self.assertEqual(values['x'],[[float(i)]*64 for i in range(1025)])
  np.testing.assert_array_equal(values['score'],-np.arange(1,1026))
  with self.assertRaises(FitReached):f(rows,123)
  self.assertEqual(len(calls),3)
 def test_one_fit_five_joint_acquisitions_keep_valid_decodes(self):
  test_batch.BatchTests.setUpClass();plans=test_batch.BatchTests.plans;fit=[];acq=[];decodes=[];encoded=[]
  rows=[observed(p,measurement(i+1,i==2),i,'test') for i,p in enumerate(plans[:3])]
  class Model:
   last_sample={}
   def request(self,*args,**kwargs):fit.append(kwargs);return dict(ok=True)
  class VAE:
   def encode(self,p):encoded.extend(p);return np.zeros((len(p),64))
   def decode(self,z,seed):
    i=len(decodes);decodes.append(1)
    # One novel and one decoder fallback/duplicate, retained across attempts.
    raw=[8,1,14,1,1,10,2,20,1,1,8,1,16,1,1,10,2,12,1,1,10,2,10,1,2,11,1,16,2,1,12,1,1,3,2,10,1,2,1,1,8,1,3,1,1,8,1,19,1,1,8,1,20,2,1,8,1,2,3,1,8,1,6,1,2]
    return [dict(plan=raw if i==0 else plans[30+i],fallback=False),dict(plan=plans[0],fallback=True)]
  def generate(state,bridge,x,y,needed,**kw):acq.append((state.batch_size,needed,len(x),kw['n_candidates']));return torch.zeros((needed,64))
  q=types.SimpleNamespace(normalized=lambda x,c:x,peptide_turbo=lambda c:types.SimpleNamespace(TurboState=lambda **kw:types.SimpleNamespace(**kw),generate_batch=generate),BridgeModel=lambda *a:None)
  f=FSBO.__new__(FSBO);f.q=q;f.task=TASK;f.cfg=dict(n_candidates=5000);f.limit=5;f.vae=VAE();f.model=Model();f.encodings={}
  out=f(rows,123,count=10,excluded=[plans[35]])
  self.assertEqual(len(fit),1);self.assertEqual(len(encoded),2);self.assertEqual(len(decodes),5);self.assertEqual([a[1] for a in acq],[10,9,8,7,6]);self.assertTrue(all(a[0]==10 and a[2]==2 and a[3]==5000 for a in acq));self.assertEqual(len(out['plans']),5)
  self.assertEqual(len({plan_key(p) for p in out['plans']}),5);self.assertTrue(out['diagnostics']['accepted_decodes'][0]['native_canonicalized'])
  from query_data import canonicalize
  for p in out['plans']:canonicalize(TASK,p,strict=True)
if __name__=='__main__':unittest.main()
