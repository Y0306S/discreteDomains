"""Regression: retrieval must not consume the retrained FSBO features."""
import unittest,types
from unittest.mock import patch
import numpy as np
import retrieval
class RepresentationTests(unittest.TestCase):
 def test_retrieval_requests_original_encoder_in_a_separate_cache(self):
  task='target';source='source';plan=[1,1,2,1,2];calls=[]
  class Pool:
   def request(self,op,**kw):calls.append((op,kw));return dict(z=np.ones((1,64)).tolist())
  fsbo=types.SimpleNamespace(vae=types.SimpleNamespace(pool=Pool()),encodings={retrieval.plan_key(plan):[99.]*64})
  card=dict(task=dict(constraints=dict(aliases=[dict(table='a')])))
  rows=[dict(first_occurrence=True,feasible=True,censored=False,observation_index=0,runtime=1.,plan=plan)]
  manifest=dict(lower=[0.]*64,upper=[1.]*64,vae_sha256='original')
  with patch.object(retrieval,'assets',return_value=({'source':card},{source:dict(sha256='gp')},manifest)),patch.object(retrieval,'task_description',return_value=card['task']),patch.object(retrieval,'source_gp',return_value={}),patch.object(retrieval,'prediction',side_effect=lambda model,x:np.asarray([float(x[0,0])])):
   result=retrieval.retrieve(task,rows,fsbo,[source])
  self.assertEqual(calls[0][0],'encode_retrieval_mean')
  self.assertEqual(fsbo.encodings[retrieval.plan_key(plan)],[99.]*64)
  self.assertEqual(fsbo.retrieval_encodings[retrieval.plan_key(plan)],[1.]*64)
  self.assertTrue(result['ranking']['representation_compatible_with_source_GPs'])
if __name__=='__main__':unittest.main()
