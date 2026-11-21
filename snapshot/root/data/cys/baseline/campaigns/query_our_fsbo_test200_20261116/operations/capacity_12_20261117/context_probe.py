"""Probe the full configured input envelope using training-only messages."""
import concurrent.futures
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'batch_code'), str(ROOT / 'operations/capacity_20261117')]
from benchmark import bodies
from hf_qwen_client import render_s1b_prompt
from qwen_inference_transport import digest, http_json
from query_protocol import atomic
from transformers import AutoTokenizer

body = bodies()[0]
tokenizer = AutoTokenizer.from_pretrained(body['identity']['base'], trust_remote_code=True)
messages = body['messages']
original = messages[-1]['content']
target_tokens = int(os.environ.get('CAPACITY_INPUT_TOKENS', '32768'))
prefix = 'max_context' if target_tokens == 32768 else f'context_{target_tokens}'
low, high = 0, target_tokens
while low < high:
    mid = (low + high + 1) // 2
    messages[-1]['content'] = original + '\nSynthetic memory test only, ignore padding:\n' + 'padding ' * mid
    count = len(tokenizer.encode(render_s1b_prompt(tokenizer, messages)))
    if count <= target_tokens:
        low = mid
    else:
        high = mid - 1
messages[-1]['content'] = original + '\nSynthetic memory test only, ignore padding:\n' + 'padding ' * low
tokens = len(tokenizer.encode(render_s1b_prompt(tokenizer, messages)))
assert target_tokens - 10 <= tokens <= target_tokens, tokens
body['namespace'] = dict(capacity_probe=True, training_only=True, maximum_context=True)
body['request_id'] = digest({k: v for k, v in body.items() if k != 'request_id'})
atomic(HERE / f'{prefix}_request.json', body)
print('PROBING', tokens, 'input tokens', flush=True)

def run(port):
    response = http_json(f'http://127.0.0.1:{port}/generate', body, timeout=600)
    assert response['identity'] == body['identity'] and response['request_id'] == body['request_id']
    result = dict(port=port, input_tokens=response['generation']['input_tokens'],
                  output_tokens=response['generation']['output_tokens'],
                  seconds=response['finished_at'] - response['started_at'],
                  peak_reserved_gib=response['peak_reserved_gib'],
                  gpu_admission_wait_seconds=response.get('gpu_admission_wait_seconds'), at=time.time())
    atomic(HERE / f'{prefix}_{port}.json', result)
    print(json.dumps(result), flush=True)
    return result

ports = [int(port) for port in sys.argv[1:]] or [21812]
with concurrent.futures.ThreadPoolExecutor(max_workers=len(ports)) as pool:
    results = list(pool.map(run, ports))
atomic(HERE / f'{prefix}_results.json', dict(passed=True, training_only=True, SQL_calls=0, results=results))
