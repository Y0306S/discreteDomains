"""Exercise the real HTTP broker with disposable CPU-only replica servers."""
import concurrent.futures
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from qwen_inference_transport import SharedInferenceError, digest, encoded, http_json


class IsolationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.identity = {'model': 'test-pinned-model'}
        self.ready = [True, True]
        self.calls = [[], []]
        self.oom = False
        self.semantic = False
        self.servers = []
        owner = self
        for index in range(2):
            class Handler(BaseHTTPRequestHandler):
                slot = index
                def log_message(self, *args):
                    pass
                def reply(self, code, payload):
                    data = encoded(payload)
                    self.send_response(code)
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                def do_GET(self):
                    self.reply(200, dict(ready=owner.ready[self.slot],
                        identities=[owner.identity], adapter_verifications=[]))
                def do_POST(self):
                    body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    owner.calls[self.slot].append(body)
                    if owner.oom and self.slot == 0:
                        owner.ready[:] = [False, True]
                        self.reply(503, dict(error='GPU allocation exhausted; replica requires inspection/restart'))
                    elif owner.semantic:
                        self.reply(503, dict(error='Request digest mismatch'))
                    else:
                        self.reply(200, dict(request_id=body['request_id'], identity=body['identity'],
                            text='saved-result-'+body['task'], cuda_rng='test', started_at=time.time()))
            server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            server.daemon_threads = True
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.servers.append(server)
        self.process = None

    def tearDown(self):
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
        for server in self.servers:
            server.shutdown()
            server.server_close()
        self.temp.cleanup()

    def body(self, task='task'):
        body = dict(identity=self.identity, task=task, messages=[{'role':'user','content':'keep this prompt'}],
                    rng={'seed':123}, settings={'temperature':0.7})
        body['request_id'] = digest(body)
        return body

    def start(self, replicas=None):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        self.url = 'http://127.0.0.1:'+str(port)
        config = dict(name='test', port=port, cpus=[min(os.sched_getaffinity(0))],
            journal=str(self.root/'journal.sqlite'), identities=[self.identity], task_affinity=False,
            replicas=replicas or ['http://127.0.0.1:'+str(s.server_port) for s in self.servers])
        path = self.root/'broker.json'
        path.write_text(json.dumps(config))
        self.process = subprocess.Popen([sys.executable, str(Path(__file__).with_name('qwen_inference_server.py')),
            'broker', '--config', str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic()+10
        while time.monotonic()<deadline:
            try:
                if http_json(self.url+'/health', timeout=1).get('ready'):
                    return
            except (OSError, SharedInferenceError):
                pass
            self.assertIsNone(self.process.poll())
            time.sleep(.05)
        self.fail('Broker did not start')

    def test_inflight_oom_reroutes_exact_request_and_replays_once(self):
        self.ready[:] = [True, False]
        self.oom = True
        self.start()
        body = self.body()
        result = http_json(self.url+'/generate', body, timeout=10)
        replay = http_json(self.url+'/generate', body, timeout=5)
        self.assertEqual(self.calls, [[body], [body]])
        self.assertEqual(result['text'], replay['text'])
        self.assertTrue(replay['replayed'])
        with sqlite3.connect(self.root/'journal.sqlite') as db:
            self.assertEqual(db.execute('SELECT count(*) FROM replica_failovers').fetchone()[0], 1)
            self.assertIsNone(db.execute('SELECT error FROM requests').fetchone()[0])

    def test_unreachable_replica_does_not_block_other_tasks(self):
        self.servers[0].shutdown()
        self.servers[0].server_close()
        self.start()
        bodies = [self.body(str(i)) for i in range(8)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda b:http_json(self.url+'/generate', b, timeout=10), bodies))
        self.assertEqual({r['request_id'] for r in results}, {b['request_id'] for b in bodies})
        self.assertEqual(len(self.calls[1]), 8)
        health = http_json(self.url+'/health')
        self.assertTrue(health['ready'])
        self.assertEqual(health['healthy_replicas'], 1)
        self.assertTrue(health['degraded'])

    def test_previously_saved_oom_is_recoverable_without_losing_success(self):
        body = self.body()
        with sqlite3.connect(self.root/'journal.sqlite') as db:
            db.execute('CREATE TABLE requests (id TEXT PRIMARY KEY, body TEXT NOT NULL, response TEXT, error TEXT, created REAL)')
            db.execute('INSERT INTO requests VALUES (?,?,NULL,?,?)',
                (body['request_id'],encoded(body).decode(),'GPU allocation exhausted; replica requires inspection/restart',time.time()))
        self.start()
        self.assertEqual(http_json(self.url+'/generate',body,timeout=10)['request_id'], body['request_id'])
        with sqlite3.connect(self.root/'journal.sqlite') as db:
            self.assertEqual(db.execute('SELECT count(*) FROM request_recoveries').fetchone()[0], 1)

    def test_semantic_error_stays_fatal_and_is_not_retried(self):
        self.semantic = True
        self.start()
        body = self.body()
        for _ in range(2):
            with self.assertRaisesRegex(SharedInferenceError,'Request digest mismatch'):
                http_json(self.url+'/generate',body,timeout=5)
        self.assertEqual(sum(map(len,self.calls)), 1)


if __name__ == '__main__':
    unittest.main()
