"""Dedicated BF16 rollout service retaining the exact sampled token sequence."""
import argparse
import json
from pathlib import Path

from rl_common import ROOT, parent_for, read, require_execution_enabled
from campaign_io import verify_campaign
from mindware.domains.dlhpo.taskset_llm.llm_client import QwenLoRAClient
from mindware.domains.dlhpo.taskset_llm.spec import render_nothink
from mindware.domains.dlhpo.taskset_llm.inference import make_server, model_identity, fingerprint


class TokenQwenClient(QwenLoRAClient):
    def _load(self):
        with self._lock:
            if self._model is not None:
                return
            from transformers import AutoTokenizer
            from train_grpo import load_model
            self._tokenizer = AutoTokenizer.from_pretrained(self.adapter_path, trust_remote_code=True)
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model, self.adapter_verification = load_model(Path(self.adapter_path), False, read('settings.json'))

    def identity(self):
        return dict(super().identity(), rollout_tokens='exact_v1')

    def generate(self, messages, *, n=1, temperature=None, seed=None):
        import torch
        self._load()
        inputs = self._tokenizer(render_nothink(messages), return_tensors='pt',
                                 add_special_tokens=False).to('cuda:0')
        if inputs['input_ids'].shape[-1] + self.max_new_tokens > read('settings.json')['max_seq_len']:
            raise RuntimeError('Untruncated prompt plus generation exceeds validated context limit')
        devices = sorted({p.device.index for p in self._model.parameters() if p.is_cuda})
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(int(seed))
            output = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                num_return_sequences=n, eos_token_id=self._stop_ids(),
                pad_token_id=self._tokenizer.pad_token_id, do_sample=True,
                temperature=self.temperature if temperature is None else temperature,
                top_p=self.top_p, top_k=self.top_k)
        start = inputs['input_ids'].shape[-1]
        return [dict(completion=self._tokenizer.decode(row[start:], skip_special_tokens=True).strip(),
                     prompt_ids=inputs['input_ids'][0].tolist(), answer_ids=row[start:].tolist()) for row in output]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--token-file', type=Path, required=True)
    p.add_argument('--identity-out', type=Path, required=True)
    p.add_argument('--max-pending', type=int, default=8)
    a = p.parse_args()
    require_execution_enabled()
    verify_campaign(ROOT)
    settings = read('settings.json')
    client = TokenQwenClient(str(parent_for(a.phase)), base_path=settings['base_model'],
        temperature=settings['temperature'], top_p=settings['top_p'], top_k=settings['top_k'],
        max_new_tokens=settings['max_new_tokens'])
    client._load()
    identity = model_identity(client)
    server = make_server(('127.0.0.1', a.port), client, identity,
                         a.token_file.read_text().strip(), max_pending=a.max_pending)
    a.identity_out.write_text(json.dumps(dict(identity=identity, identity_sha256=fingerprint(identity))))
    print(json.dumps(dict(ready=True, phase=a.phase, port=a.port)), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
