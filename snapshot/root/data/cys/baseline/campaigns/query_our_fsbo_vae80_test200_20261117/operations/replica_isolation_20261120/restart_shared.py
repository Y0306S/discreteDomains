"""Restart only failed A16 pairs; forward intact long prompts to A800s."""
import concurrent.futures
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OPS=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'batch_code'))
from query_protocol import atomic,digest
import resources

names=['llm16','llm17']
pins_path=ROOT/'evidence/batch_code_pins.json'
pins=json.loads(pins_path.read_text())
for name in names:
    for role in ['orpo','warm_start']:
        path=ROOT/f'services/{role}.{name}.resources.json'
        before=OPS/'before'/path.relative_to(ROOT);before.parent.mkdir(parents=True,exist_ok=True)
        if not before.exists():shutil.copy2(path,before)
        cfg=json.loads(path.read_text())
        assert cfg['context_fallback_urls'] and cfg['fallback_max_input_tokens']==32768
        # Target GP fitting can grow to 5.3 GiB on these shared GPUs. Local
        # prefill at 16k was validated only before that peak. Keep the complete
        # original conversation and sampling budget through existing forwarding.
        cfg['max_input_tokens']=4096
        atomic(path,cfg)
        pins[str(path)]=digest(path)
atomic(pins_path,pins)

def restart(name):
    resources.stop_service(name+'_resources','LLMOptimizer3')
    deadline=time.monotonic()+60
    port=21944 if name=='llm16' else 21945
    while True:
        try:resources.json_get(f'http://127.0.0.1:{port}/health')
        except OSError:break
        assert time.monotonic()<deadline
        time.sleep(.5)
    pid=resources.start_service([resources.INFER,'-u',str(ROOT/'batch_code/qwen_inference_server.py'),
        'replica','--config',str(ROOT/f'services/orpo.{name}.resources.json')],name+'_resources','LLMOptimizer3')
    print('Restarted',name,'launcher',pid,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(restart,names))
atomic(OPS/'shared_pairs_restart.json',dict(at=time.time(),names=names,local_prefill_limit=4096,
    full_context_limit=32768,reason='share GPU memory with growing peptide target GPs'))
