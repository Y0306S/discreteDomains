"""Bounded, immutable prompt snapshots, including Qwen3.5 recurrent state.

Each generation receives its own copy. Generated tokens must never contaminate
the snapshot used to sample a fresh answer from the same prompt.
"""
from collections import OrderedDict
import copy
import gc
import time


def tensor_bytes(value):
    import torch
    seen = set()
    def visit(obj):
        if id(obj) in seen:
            return 0
        seen.add(id(obj))
        if isinstance(obj, torch.Tensor):
            return obj.numel() * obj.element_size()
        if isinstance(obj, dict):
            return sum(visit(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(visit(v) for v in obj)
        if hasattr(obj, '__dict__'):
            return visit(vars(obj))
        return 0
    return visit(value)


class PromptKVCache:
    def __init__(self, max_bytes=512 * 2**20, min_tokens=256):
        self.max_bytes = max_bytes
        self.min_tokens = min_tokens
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0

    def clear(self):
        self.entries.clear()
        self.bytes = 0

    def _remember(self, key, cache):
        size = tensor_bytes(cache)
        if size > self.max_bytes:
            return False
        while self.entries and self.bytes + size > self.max_bytes:
            _, (_, old_size) = self.entries.popitem(last=False)
            self.bytes -= old_size
        self.entries[key] = (cache, size)
        self.bytes += size
        return True

    def _generate(self, model, key, kwargs, stats):
        import torch
        count = kwargs['input_ids'].shape[-1]
        if key in self.entries:
            cache, _ = self.entries[key]
            self.entries.move_to_end(key)
            working = copy.deepcopy(cache)
            self.hits += 1
            stats.update(hit=True, reused_tokens=count - 1)
        else:
            self.misses += 1
            # Leave the last prompt token unprocessed: generate() obtains its
            # logits normally and preserves its original full input/output IDs.
            inputs = {name: value[:, :-1] for name, value in kwargs.items()
                      if name in ('input_ids', 'attention_mask', 'token_type_ids')}
            prefill_started = time.monotonic()
            with torch.no_grad():
                output = model(**inputs, use_cache=True, logits_to_keep=1,
                               return_dict=True)
            if kwargs['input_ids'].is_cuda:
                torch.cuda.synchronize(kwargs['input_ids'].device)
            stats['prefill_seconds'] = time.monotonic() - prefill_started
            cache = output.past_key_values
            del output
            if cache is None or cache.get_seq_length() != count - 1:
                raise RuntimeError('Prompt cache did not preserve the full prefix')
            if self._remember(key, cache):
                working = copy.deepcopy(cache)
            else:
                working = cache
            del cache
        with torch.no_grad():
            return model.generate(**dict(kwargs, past_key_values=working))

    def generate(self, model, key, kwargs):
        import torch
        started = time.monotonic()
        stats = dict(enabled=True, hit=False, reused_tokens=0, memory_fallback=False)
        count = kwargs['input_ids'].shape[-1]
        if count < max(2, self.min_tokens) or self.max_bytes <= 0:
            with torch.no_grad():
                output = model.generate(**kwargs)
        else:
            # Cache retention is optional. If it consumes too much headroom,
            # evict it and retry the exact request from its original RNG state.
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            try:
                output = self._generate(model, key, kwargs, stats)
            except torch.cuda.OutOfMemoryError:
                self.clear()
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
                stats.update(hit=False, reused_tokens=0, memory_fallback=True)
            # Leave the exception scope before retrying so the traceback cannot
            # retain the failed working cache and its CUDA allocations.
            if stats['memory_fallback']:
                gc.collect()
                torch.cuda.empty_cache()
                with torch.no_grad():
                    output = model.generate(**kwargs)
        stats.update(retained_bytes=self.bytes, hits=self.hits, misses=self.misses,
                     generation_seconds=time.monotonic() - started)
        return output, stats
