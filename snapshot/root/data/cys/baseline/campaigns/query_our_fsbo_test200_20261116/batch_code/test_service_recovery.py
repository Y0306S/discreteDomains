import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import resources
from backends import LLM
from qwen_inference_transport import SharedInferenceError

class RecoveryTests(unittest.TestCase):
 def test_pid_identity_reuse(self):
  ticks=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19]
  self.assertTrue(resources.record_alive(dict(pid=os.getpid(),start_ticks=ticks)))
  self.assertFalse(resources.record_alive(dict(pid=os.getpid(),start_ticks=str(int(ticks)-1))))
  self.assertFalse(resources.record_alive(dict(pid=os.getpid())))
 def test_every_service_identity_is_checked(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'services').mkdir()
   (root/'services/active_inference_stage.json').write_text(json.dumps(dict(role='warm_start',stage='warm30')))
   for name in ['a_resources','b_resources','warm_start_resources_broker','resources_tunnel']:
    (root/'services'/f'{name}.process.json').write_text(json.dumps(dict(pid=123,start_ticks='1',name=name)))
   with patch.object(resources,'ROOT',root),patch.object(resources,'config',return_value=dict(llm=[dict(name='a')],vae=[dict(name='b')])),patch.object(resources,'record_alive',side_effect=lambda r:r['name']!='a_resources'):
    with self.assertRaisesRegex(RuntimeError,'a_resources'):resources.health(deep=False)
 def test_transport_retry_preserves_request(self):
  obj=LLM.__new__(LLM);calls=[];body=dict(request_id='fixed')
  def http(url,value,timeout):
   calls.append((url,value,timeout))
   if len(calls)<3:raise ConnectionResetError('temporary disconnect')
   return dict(replayed=True)
  obj.http=http
  with patch('backends.time.sleep'):self.assertEqual(obj.request('url',body),dict(replayed=True))
  self.assertEqual(len(calls),3);self.assertTrue(all(c[1] is body for c in calls))
 def test_semantic_failure_is_not_retried(self):
  obj=LLM.__new__(LLM)
  with patch.object(obj,'http',create=True,side_effect=SharedInferenceError('identity mismatch')) as http:
   with self.assertRaises(SharedInferenceError):obj.request('url',{})
   self.assertEqual(http.call_count,1)
 def test_transport_exhaustion_is_bounded(self):
  obj=LLM.__new__(LLM)
  with patch.object(obj,'http',create=True,side_effect=ConnectionResetError()) as http,patch('backends.time.sleep'):
   with self.assertRaises(ConnectionResetError):obj.request('url',{})
   self.assertEqual(http.call_count,4)

if __name__=='__main__':unittest.main()
