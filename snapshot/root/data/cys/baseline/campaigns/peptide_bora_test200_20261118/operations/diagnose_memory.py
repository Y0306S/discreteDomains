"""Replay only the failed prompt prefill; no peptide oracle calls or task edits."""
import collections
import copy
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'services'))
from qwen_inference_server import Replica
from qwen_inference_transport import digest

out = Path((ROOT/'operations/resume_latest.txt').read_text().strip())
db = sqlite3.connect('file:'+str(out/'requests_before.sqlite')+'?mode=ro', uri=True)
row = db.execute('select body from requests where error like ? limit 1', ('%GPU allocation exhausted%',)).fetchone()
body = json.loads(row[0])
db.close()
body['settings']['max_new_tokens'] = 1
body['request_id'] = digest({k:v for k,v in body.items() if k != 'request_id'})
cfg = json.loads((ROOT/'services/replica00.json').read_text())
os.environ.update(HF_QWEN_DEVICE_MAP='auto', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
balanced = '--balanced' in sys.argv
result = dict(started_at=time.time(), purpose='failed prompt prefill, one diagnostic output token, no oracle calls',
              weight_memory_gib=cfg.get('weight_memory_gib'), tests=[])
try:
    replica = Replica(cfg)
    import torch
    sizes = collections.defaultdict(int)
    for p in replica.client._model.parameters():
        sizes[str(p.device)] += p.numel()*p.element_size()
    result['parameter_gib'] = {k:v/2**30 for k,v in sizes.items()}
    def token_count(messages):
        prompt = replica.client._tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        return len(replica.client._tokenizer(prompt)['input_ids'])
    result['prompt_tokens'] = token_count(body['messages'])
    print(json.dumps(result), flush=True)
    result['tests'].append(dict(name='previously_failed_prompt', response=replica.generate(body)))
    if balanced:
        longest = copy.deepcopy(body)
        longest['messages'] = [dict(role='user', content=' A'*27300)]
        # Stay within the original envelope while exercising almost its full length.
        n = token_count(longest['messages'])
        assert 27200 <= n <= cfg['max_input_tokens'], n
        longest['request_id'] = digest({k:v for k,v in longest.items() if k != 'request_id'})
        result['tests'].append(dict(name='maximum_context_prefill', input_tokens=n,
            response=replica.generate(longest)))
    result['passed'] = True
except BaseException:
    result['passed'] = False
    result['traceback'] = traceback.format_exc()
    print(result['traceback'], flush=True)
finally:
    import torch
    result['memory'] = {str(i):dict(allocated_gib=torch.cuda.memory_allocated(i)/2**30,
        reserved_gib=torch.cuda.memory_reserved(i)/2**30, peak_reserved_gib=torch.cuda.max_memory_reserved(i)/2**30)
        for i in range(torch.cuda.device_count())}
    result['finished_at'] = time.time()
    (out/('balanced_memory_validation.json' if balanced else 'memory_diagnosis.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
