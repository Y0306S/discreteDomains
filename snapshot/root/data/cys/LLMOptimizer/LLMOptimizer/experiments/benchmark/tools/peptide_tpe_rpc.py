"""Persistent TPE inference connections without changing pinned request/RNG semantics."""

import hashlib
import http.client
import time
import urllib.parse

from mindware.domains.protein.peptide_rpc import RemoteObjective, dumps, loads


class PersistentRemoteObjective(RemoteObjective):
    def __init__(self, objective, url, identity, timing=None):
        self.connection = None
        self.timing = timing or (lambda *args: None)
        super().__init__(objective, url, identity)

    def request(self, value):
        started = time.monotonic()
        value = dict(value, identity=self.identity)
        data = dumps(value)
        self.timing('rpc_serialization', time.monotonic() - started)
        request_id = hashlib.sha256(data).hexdigest()
        address = urllib.parse.urlsplit(self.url)
        if address.scheme != 'http' or not address.hostname:
            raise ValueError('Persistent peptide RPC requires an HTTP inference URL')
        error = None
        for attempt in range(3):
            started = time.monotonic()
            try:
                if self.connection is None:
                    self.connection = http.client.HTTPConnection(address.hostname, address.port, timeout=600)
                self.connection.request('POST', (address.path.rstrip('/') or '') + '/rpc', body=data,
                    headers={'Content-Type': 'application/octet-stream', 'X-Request-ID': request_id})
                response = self.connection.getresponse()
                payload = response.read()
                self.timing('rpc_' + value['op'], time.monotonic() - started)
                for header, label in [('X-Queue-Seconds', 'server_queue_' + value['op']),
                                      ('X-Execution-Seconds', 'server_execution_' + value['op'])]:
                    if response.getheader(header) is not None:
                        self.timing(label, float(response.getheader(header)))
                if response.status != 200:
                    raise RuntimeError('Inference HTTP status: %d' % response.status)
                result = loads(payload)
                if result.get('error'):
                    raise RuntimeError('GPU service: ' + result['error'])
                return result
            except (OSError, http.client.HTTPException) as exc:
                error = exc
                if self.connection is not None:
                    self.connection.close()
                self.connection = None
                time.sleep(attempt + 1)
        raise RuntimeError('GPU service unavailable after retries') from error

    def close(self):
        try:
            super().close()
        finally:
            if self.connection is not None:
                self.connection.close()
                self.connection = None
