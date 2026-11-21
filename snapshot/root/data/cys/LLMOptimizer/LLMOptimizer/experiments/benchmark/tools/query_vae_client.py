"""Client pool for query VAE services, including SSH-forwarded remote endpoints."""
import hashlib
import json
from pathlib import Path
import threading
import urllib.request


class QueryVAEPool:
    def __init__(self, manifest, timeout=300):
        self.manifest = json.loads(Path(manifest).read_text())
        if self.manifest['domain'] != 'ceb_query_runtime':
            raise ValueError('Expected a query VAE pool')
        self.endpoints = [w['url'] for w in self.manifest['workers']]
        self.token = Path(self.manifest['token_file']).read_text().strip()
        self.timeout = timeout
        self.lock = threading.Lock()
        self.counter = 0

    def request(self, op, seed=123, affinity=None, **values):
        if affinity is None:
            with self.lock:
                index = self.counter % len(self.endpoints)
                self.counter += 1
        else:
            index = int.from_bytes(hashlib.sha256(affinity.encode()).digest()[:4], 'big') % len(self.endpoints)
        payload = json.dumps(dict(values, op=op, seed=seed), allow_nan=False).encode()
        request = urllib.request.Request(self.endpoints[index] + '/inference', data=payload,
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.token})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def encode_mean(self, plans, task=None, seed=123):
        return self.request('encode_mean', plans=plans, task=task, seed=seed)['z']

    def decode(self, vectors, task, fallback, seed):
        return self.request('decode', z=vectors, task=task, fallback=fallback,
                            seed=seed, affinity=task)['rows']

    def health(self):
        result = []
        for endpoint in self.endpoints:
            with urllib.request.urlopen(endpoint + '/health', timeout=self.timeout) as response:
                result.append(json.load(response))
        return result


class QueryVAEBackend:
    """Task-bound encode/decode adapter for future query FSBO/RGPE runners.

    Encodings are posterior means, as used to train the query transfer weights.
    Callers retain responsibility for normalization, fitting and acquisition.
    """
    def __init__(self, manifest, task, fallback, seed=123, timeout=300):
        self.pool = QueryVAEPool(manifest, timeout=timeout)
        self.task, self.fallback, self.seed = task, list(fallback), int(seed)
        self.decode_index = 0

    def encode(self, plans):
        return self.pool.encode_mean(plans, task=self.task)

    def decode(self, vectors, seed=None):
        if seed is None:
            seed = self.seed + self.decode_index
            self.decode_index += 1
        return self.pool.decode(vectors, self.task, self.fallback, seed)
