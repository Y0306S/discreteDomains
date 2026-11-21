"""Admit only capacity-checked replicas without interrupting the durable broker."""
import argparse,hashlib,json,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'batch_code'))
from query_protocol import atomic
from resources import placement_ok
parser=argparse.ArgumentParser();parser.add_argument('--names',nargs='+');args=parser.parse_args()
planned=json.loads((ROOT/'operations/reallocate_20261119/resources.final_planned.json').read_text())
live=json.loads((ROOT/'resources.json').read_text())
assert live['workers']==planned['workers'] and live['vae']==planned['vae']
for spec in planned['llm'][12:]:
 if args.names and spec['name'] not in args.names:continue
 port=spec['port'];url=f'http://127.0.0.1:{port}'
 smoke=json.loads((ROOT/f'operations/reallocate_20261119/inference_smoke_{port}.json').read_text())
 assert smoke['passed'] and smoke['input_tokens']>=32000 and smoke['SQL_calls']==0
 health=json.load(urllib.request.urlopen(url+'/health',timeout=30))
 assert placement_ok(spec,health) and health['adapter_verifications'][0]['all_tensors_exact']
 if spec.get('context_fallback'):
  short=json.loads((ROOT/f'operations/reallocate_20261119/inference_smoke_{port}.short.json').read_text())
  assert short['passed'] and 14000<short['input_tokens']<=health['local_max_input_tokens']
  assert short['result']['replica']==health['name'] and not short['result'].get('context_forwarded_from')
  assert smoke['result']['context_forwarded_from']==health['name']
  assert smoke['result']['routing_input_tokens']==smoke['input_tokens']==smoke['result']['generation']['input_tokens']
 req=urllib.request.Request('http://127.0.0.1:21811/add_replica',data=json.dumps(dict(url=url)).encode(),headers={'Content-Type':'application/json'})
 print(json.load(urllib.request.urlopen(req,timeout=60)),flush=True)
 if not any(w['name']==spec['name'] for w in live['llm']):live['llm'].append(spec)
 atomic(ROOT/'resources.json',live)
urls=[f"http://127.0.0.1:{w['port']}" for w in live['llm']]
for role in ['orpo','warm_start']:
 p=ROOT/f'services/{role}.resources.broker.json';s=json.loads(p.read_text());s['replicas']=urls;atomic(p,s)
p=ROOT/'services/active_inference_stage.json';s=json.loads(p.read_text());s.update(replicas=len(urls),at=time.time());atomic(p,s)
live['llm_stage_policy']='Verified GPU placements; full context and unchanged model identities. See llm entries and capacity smoke evidence.'
live['capacity_reallocation_20261119'].update(llm_replicas=len(urls),verified_at=time.time(),llm_layout=dict(A800=12,A16=len(urls)-12))
atomic(ROOT/'resources.json',live)
p=ROOT/'evidence/batch_code_pins.json';pins=json.loads(p.read_text())
files=['resources.json','services/orpo.resources.broker.json','services/warm_start.resources.broker.json','batch_code/resources.py','batch_code/qwen_inference_server.py','operations/reallocate_20261119/inference_smoke.py','operations/reallocate_20261119/admit_verified.py','operations/reallocate_20261119/capacity_audit.py']
for name in files:
 f=ROOT/name;pins[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
for spec in live['llm']:
 for role in ['orpo','warm_start']:
  f=ROOT/f"services/{role}.{spec['name']}.resources.json"
  pins[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
atomic(p,pins)
assert all(Path(k).is_file() and hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in pins.items())
print('Pinned allocation:',len(live['workers']),'evaluators,',len(urls),'LLMs,',len(live['vae']),'VAEs',flush=True)
