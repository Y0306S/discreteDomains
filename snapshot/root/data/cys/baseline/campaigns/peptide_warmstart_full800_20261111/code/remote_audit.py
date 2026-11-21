"""Recheck shared SFT inputs on the released remote ML CPU worker."""
import json
import time
from train import ROOT, load, sha, atomic

identity = load(ROOT/'identity.json')
checked = 0
for name, expected in identity['files'].items():
    if name.startswith('/root/data/'):
        if sha(name) != expected:
            raise ValueError('Shared SFT input changed: '+name)
        checked += 1
gate = load(ROOT/'evidence/preflight.json')
if not gate['passed'] or gate['config_sha256'] != sha(ROOT/'sft_config.json'):
    raise ValueError('Local full-token preflight missing or changed')
data = load(ROOT/'evidence/data_audit.json')
if not data['passed'] or data['examples'] != 8000 or data['train_test_overlap'] != 0:
    raise ValueError('Local complete provenance audit missing')
atomic(ROOT/'evidence/remote_handoff_audit.json', dict(passed=True, at=time.time(),
    host='LLMOptimizer4', shared_inputs_verified=checked, examples=8000,
    local_token_preflight_sha256=sha(ROOT/'evidence/preflight.json'),
    local_data_audit_sha256=sha(ROOT/'evidence/data_audit.json'),
    note='Original report and base-model tokenizer live on LLMOptimizer2; full provenance and tokenization were checked there.'))
