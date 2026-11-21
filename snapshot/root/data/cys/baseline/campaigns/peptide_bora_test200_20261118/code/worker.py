"""Resumable native-feedback BORA, with one durable oracle request at a time."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from common import ROOT, atomic, config, read, status, valid, verify

sys.path[:0] = [str(ROOT/'vendor'), str(ROOT/'services')]
from bora.policy import Policy
from backend import Backend, bo_candidates
from assistant import Assistant


class Worker:
    def __init__(self, task, directory=None, cfg=None, backend=None, client=None, tokenizer=None):
        import torch
        self.cfg = cfg or verify()
        self.task = task
        self.directory = Path(directory or ROOT/'runs'/task)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.guard = open(self.directory/'worker.lock', 'a')
        fcntl.flock(self.guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.backend = backend or Backend(self.cfg)
        random.seed(self.cfg['seed'])
        torch.manual_seed(self.cfg['seed'])
        self.rng = np.random.RandomState(self.cfg['seed'])
        self.reference = read(ROOT/'initialization.json')[task]['reference']
        self.inference_error = None
        self.client, self.tokenizer = client, tokenizer
        self.selection = []
        self.started = time.monotonic()
        self.base_elapsed = 0
        if (self.directory/'checkpoint.pkl').exists():
            with open(self.directory/'checkpoint.pkl', 'rb') as stream:
                self.state = pickle.load(stream)
            if self.state['identity'] != self.cfg['identity']:
                raise RuntimeError('Campaign identity changed; refusing checkpoint adoption')
            self.rng.set_state(self.state['rng'])
            torch.set_rng_state(self.state['torch_rng'])
            random.setstate(self.state['python_rng'])
            self.base_elapsed = self.state['elapsed_seconds']
        else:
            pool = read(ROOT/'initialization.json')[task]
            sequences = pool['sequences']
            if len(sequences) != 100 or len(set(sequences)) != 100 or not all(valid(s, self.reference) for s in sequences):
                raise ValueError('Invalid shared initialization')
            x = self.backend.encode(sequences)
            low = x.min(axis=0)
            scale = np.maximum(x.max(axis=0)-low, 1e-6)
            self.state = dict(version=1, identity=self.cfg['identity'],
                rows=[dict(sequence=s, mic=-float(y), source='shared_initialization', evaluation=0) for s,y in zip(sequences, pool['scores'])],
                x=2*(x-low)/scale-1, low=low, scale=scale, policy=Policy(dim=256),
                initialized=False, ready=False, queue=[], rewards=[], assistant=None, inference=None,
                pending=None, kernel=None, timing={}, physical_oracle_calls=0, acquisition_events=0,
                monitoring=self.rng.uniform(-1, 1, size=(5000, 256)))
        for row, vector in zip(self.state['rows'], self.state['x']):
            self.backend.cache[row['sequence']] = self.state['low']+(vector+1)/2*self.state['scale']
        self.gp = None
        self.assistant = None
        if self.state['ready']:
            self.fit(restore=True)

    def record(self, value):
        with open(self.directory/'decisions.jsonl', 'a') as stream:
            stream.write(json.dumps(dict(at=time.time(), **value), allow_nan=False)+'\n')

    def prompt_tokens(self, messages):
        self.ensure_client()
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        return len(self.tokenizer(prompt)['input_ids'])

    def ensure_client(self):
        if self.client is None:
            from hf_qwen_client import HuggingFaceQwenClient
            from transformers import AutoTokenizer
            os.environ['ML_QWEN_POOL_CONFIG'] = str(ROOT/'services/client.json')
            os.environ['ML_QWEN_REQUEST_NAMESPACE'] = str(self.directory)
            os.environ['ML_RUN_ID'] = ROOT.name
            self.client = HuggingFaceQwenClient(self.cfg['model_identity']['base'],
                adapter_path=None, temperature=self.cfg['temperature'], max_new_tokens=self.cfg['max_new_tokens'],
                top_p=self.cfg['top_p'], top_k=self.cfg['top_k'], enable_thinking=False)
            self.client.restore_inference_state(self.state['inference'])
            self.client._ensure_loaded()
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg['model_identity']['base'], trust_remote_code=True)

    def complete(self, messages):
        self.ensure_client()
        tokens = self.prompt_tokens(messages)
        if tokens > self.cfg['max_input_tokens']:
            raise RuntimeError(f'BORA context limit exceeded ({tokens}); no oracle evaluation started')
        begun = time.monotonic()
        text = self.client.complete_messages(messages, use_s1b_template=False)
        self.timed('llm', time.monotonic()-begun)
        self.record(dict(stage='official_assistant_chat', messages=messages, raw_text=text,
            prompt_tokens=tokens, inference=self.client.last_generation))
        return text

    def timed(self, key, seconds):
        t = self.state['timing']
        t[key] = t.get(key, 0)+seconds

    def save(self):
        import torch
        self.state.update(rng=self.rng.get_state(), torch_rng=torch.get_rng_state(),
            python_rng=random.getstate(), elapsed_seconds=self.base_elapsed+time.monotonic()-self.started)
        if self.assistant is not None:
            self.state['assistant'] = self.assistant.snapshot()
        if self.client is not None:
            self.state['inference'] = self.client.inference_state()
        atomic(self.directory/'checkpoint.pkl', self.state, binary=True)

    def progress(self, phase='running'):
        n = len(self.state['rows'])-100
        status(self.directory/'status.json', status=phase, task=self.task,
            evaluations=n, budget=self.cfg['fresh_budget'], report_checkpoint=n//50,
            best_mic=min(r['mic'] for r in self.state['rows']),
            pending=self.state['pending']['status'] if self.state['pending'] else None,
            timing=self.state['timing'], elapsed_seconds=self.base_elapsed+time.monotonic()-self.started,
            acquisition_events=self.state['acquisition_events'],
            physical_oracle_calls=self.state['physical_oracle_calls'])

    def fit(self, restore=False):
        begun = time.monotonic()
        nfit = self.state.get('gp_fit_count', len(self.state['rows'])) if restore else len(self.state['rows'])
        y = -np.asarray([r['mic'] for r in self.state['rows'][:nfit]])
        self.gp = GaussianProcessRegressor(kernel=self.state['kernel'] if restore else Matern(nu=2.5),
            alpha=self.cfg['gp']['alpha'], normalize_y=True,
            n_restarts_optimizer=0 if restore else self.cfg['gp']['restarts'],
            optimizer=None if restore else 'fmin_l_bfgs_b', random_state=self.rng)
        self.gp.fit(self.state['x'][:nfit], y)
        self.state['gp_fit_count'] = nfit
        self.state['kernel'] = copy.deepcopy(self.gp.kernel_)
        self.timed('gp_restore' if restore else 'gp_fit', time.monotonic()-begun)
        if restore:
            return
        uncertainty = float(np.mean(np.concatenate([self.gp.predict(a, return_std=True)[1]
            for a in np.array_split(self.state['monitoring'], 20)])))
        iteration = len(self.state['rows'])
        if not self.state['initialized']:
            self.state['policy'].initialize(iteration, float(max(y)), uncertainty)
            self.state['initialized'] = True
        else:
            self.state['policy'].update(self.state['rewards'], float(max(y)), iteration, uncertainty)
        self.state['ready'] = True
        self.save()

    def get_assistant(self):
        if self.assistant is None:
            self.ensure_client()
            self.assistant = Assistant(self)
            if self.state['assistant'] is not None:
                self.assistant.restore(self.state['assistant'])
        return self.assistant

    def propose(self):
        observed = [r['sequence'] for r in self.state['rows']]
        rewards = -np.asarray([r['mic'] for r in self.state['rows']])
        action = self.state['policy'].get_action().name
        candidates = []
        if action == 'ASSISTANT_SUGGESTS':
            data = pd.DataFrame([dict(sequence=r['sequence'], score=-r['mic'], iteration=i+1)
                for i,r in enumerate(self.state['rows'])])
            candidates = self.get_assistant().suggest(data, observed)
        if action in ['BO', 'ASSISTANT_SELECTS'] or not candidates:
            count = 5 if action == 'ASSISTANT_SELECTS' else 1
            pool, timing = bo_candidates(self.backend, self.gp, self.state['x'], rewards,
                self.state['low'], self.state['scale'], self.reference, observed, self.rng, self.cfg, count)
            self.timed('acquisition', timing['seconds'])
            self.record(dict(stage='EI_acquisition', **timing))
            if action == 'ASSISTANT_SELECTS':
                data = pd.DataFrame([dict(sequence=r['sequence'], score=-r['mic'], iteration=i+1)
                    for i,r in enumerate(self.state['rows'])])
                candidates = self.get_assistant().select(data, pool)
            if not candidates:
                if action != 'BO':
                    self.state['policy'].skip_next_update()
                    action = 'BO_after_rejected_intervention'
                candidates = pool[:1]
        unique = list(dict.fromkeys(s for s in candidates if s not in observed and valid(s, self.reference)))
        if not unique:
            raise RuntimeError('No novel feasible BORA proposal; no oracle evaluation started')
        remaining = self.cfg['fresh_budget']-(len(self.state['rows'])-100)
        self.state['queue'] = [dict(sequence=s, source=action) for s in unique[:remaining]]
        self.state['rewards'] = []
        self.state['acquisition_events'] += 1
        self.save()

    def run(self, stop_after=None):
        target = min(self.cfg['fresh_budget'], stop_after or self.cfg['fresh_budget'])
        if self.state['pending'] and self.state['pending']['status'] == 'in_flight':
            raise RuntimeError('Unresolved oracle call requires audit; automatic replay is disabled')
        while len(self.state['rows'])-100 < target:
            self.progress()
            if not self.state['ready']:
                self.fit()
            if self.state['pending'] is None:
                if not self.state['queue']:
                    self.propose()
                proposal = self.state['queue'].pop(0)
                begun = time.monotonic()
                x = 2*(self.backend.encode([proposal['sequence']])[0]-self.state['low'])/self.state['scale']-1
                self.timed('encoding', time.monotonic()-begun)
                self.state['pending'] = dict(**proposal, status='selected', x=x)
                self.save()
            pending = self.state['pending']
            if pending['status'] == 'selected':
                if pending['sequence'] in {r['sequence'] for r in self.state['rows']}:
                    raise RuntimeError('Duplicate pending oracle request')
                pending['status'] = 'in_flight'
                self.state['physical_oracle_calls'] += 1
                self.save()
                begun = time.monotonic()
                pending['mic'] = float(self.backend.score([pending['sequence']])[0])
                self.timed('oracle', time.monotonic()-begun)
                pending['status'] = 'scored'
                self.save()
            if pending['status'] != 'scored' or not np.isfinite(pending['mic']):
                raise RuntimeError('Missing durable finite oracle result')
            n = len(self.state['rows'])-99
            self.state['rows'].append(dict(sequence=pending['sequence'], mic=pending['mic'], source=pending['source'],
                evaluation=n, elapsed_seconds=self.base_elapsed+time.monotonic()-self.started))
            self.state['x'] = np.vstack([self.state['x'], pending['x']])
            self.state['rewards'].append(-pending['mic'])
            self.state['pending'] = None
            self.state['ready'] = bool(self.state['queue'])
            self.save()
            self.progress()
        complete = len(self.state['rows'])-100 == self.cfg['fresh_budget']
        if complete:
            atomic(self.directory/'observations.json', self.state['rows'])
            status(self.directory/'result.json', status='complete', task=self.task,
                evaluations=self.cfg['fresh_budget'], initial_evaluations=100,
                best_mic=min(r['mic'] for r in self.state['rows']),
                elapsed_seconds=self.base_elapsed+time.monotonic()-self.started,
                timing=self.state['timing'], identity=self.cfg['identity'])
        self.progress('complete' if complete else 'validation_checkpoint')
        return complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--validation', action='store_true')
    args = parser.parse_args()
    cfg = verify()
    if args.task not in cfg['tasks']:
        raise ValueError('Task is not in the pinned 200-task test manifest')
    directory = ROOT/('validation/live' if args.validation else 'runs')/args.task
    try:
        worker = Worker(args.task, directory, cfg)
        worker.run(args.stop_after)
    except BaseException:
        status(directory/'error.json', status='failed', error=traceback.format_exc())
        if 'worker' in locals():
            worker.progress('failed')
        raise


if __name__ == '__main__':
    main()
