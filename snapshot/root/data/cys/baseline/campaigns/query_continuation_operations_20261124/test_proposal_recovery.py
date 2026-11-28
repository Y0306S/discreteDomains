import json,pickle,tempfile,types,unittest
from pathlib import Path
from bolt_proposal_recovery import collect,write_checkpoint

class ProposalRecoveryTest(unittest.TestCase):
 def test_partial_batch_continues_without_accepting_fallback_or_duplicates(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);(root/'runs/test').mkdir(parents=True)
   def atomic(p,d):p.write_text(json.dumps(d))
   runner=types.SimpleNamespace(ROOT=root,plan_key=tuple,canonicalize=lambda task,plan,**kw:plan,atomic=atomic)
   state={'rows':[{'plan':[0]}]};checkpoint=dict(round=66,next_retry=0,accepted=[],rejected=[],fit={'seed':1})
   path=root/'checkpoint.pickle'
   def initial(retry):
    self.assertEqual(retry,0)
    return [[i] for i in range(10)],[{'plan':[i]} for i in range(9)]+[{'plan':[99],'fallback':True}]
   with self.assertRaisesRegex(RuntimeError,'8/10'):
    collect(runner,'test',state,checkpoint,path,initial,1)
   saved=pickle.loads(path.read_bytes());self.assertEqual(saved['next_retry'],1)
   def resumed(retry):
    self.assertEqual(retry,1)
    return [[i] for i in range(10)],[{'plan':[8]},{'plan':[99],'fallback':True}]+[{'plan':[i]} for i in range(9,17)]
   result=collect(runner,'test',state,saved,path,resumed,2)
   plans=[tuple(x['plan']) for x in result['items']]
   self.assertEqual(len(plans),10);self.assertEqual(len(set(plans)),10)
   self.assertNotIn((0,),plans);self.assertNotIn((99,),plans)
   final=pickle.loads(path.read_bytes())
   def forbidden(_):self.fail('A complete proposal batch must not be regenerated')
   again=collect(runner,'test',state,final,path,forbidden,2)
   self.assertEqual(result,again)
 def test_decode_failure_preserves_previous_retry_checkpoint(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);(root/'runs/test').mkdir(parents=True)
   runner=types.SimpleNamespace(ROOT=root,plan_key=tuple,canonicalize=lambda task,plan,**kw:plan,atomic=lambda p,d:p.write_text(json.dumps(d)))
   path=root/'checkpoint.pickle';checkpoint=dict(round=66,next_retry=0,accepted=[],rejected=[],fit={})
   def sample(retry):
    if retry==1:raise ConnectionError('VAE unavailable')
    return [[0]]*10,[{'plan':[1]}]*10
   with self.assertRaises(ConnectionError):collect(runner,'test',{'rows':[]},checkpoint,path,sample,3)
   saved=pickle.loads(path.read_bytes());self.assertEqual(saved['next_retry'],1);self.assertEqual(len(saved['accepted']),1)

if __name__=='__main__':unittest.main()
