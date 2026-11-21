import argparse
import hashlib
import json
import os
import pickle
import threading
import time
import traceback
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from fast_math import Math


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, payload, kind='application/octet-stream', status=200):
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        s = self.server
        if self.path != '/health':
            self.reply(b'not found', status=404)
            return
        with s.stats_lock:
            status = dict(s.status, pending=s.pending, active_operation=s.active_operation,
                          calls=s.calls, errors=s.errors, operations=dict(s.operations),
                          queue_seconds_total=s.queue_total, compute_seconds_total=s.compute_total)
        self.reply(json.dumps(dict(status, at=time.time(), pid=os.getpid(),
                                   gpu=os.environ['CUDA_VISIBLE_DEVICES'], **s.math.run({'op':'health'}))).encode(),
                   'application/json')

    def do_POST(self):
        s = self.server
        if self.path != '/rpc' or self.headers.get('Authorization') != 'Bearer '+s.token:
            self.reply(b'forbidden',status=403)
            return
        size = int(self.headers.get('Content-Length','0'))
        if not 0 < size < 512*2**20:
            self.reply(b'invalid request size',status=413)
            return
        req = pickle.loads(self.rfile.read(size))
        queued = time.monotonic()
        with s.stats_lock:
            s.pending += 1
        with s.compute_lock:
            queue_seconds = time.monotonic()-queued
            started = time.monotonic()
            with s.stats_lock:
                s.pending -= 1
                s.active_operation = req.get('op')
            try:
                result = s.math.run(req)
                reply = dict(result=result, queue_seconds=queue_seconds,
                             compute_seconds=time.monotonic()-started)
                with s.stats_lock:
                    s.calls += 1
                    s.operations[req['op']] += 1
            except Exception:
                reply = dict(error=traceback.format_exc(),queue_seconds=queue_seconds,
                             compute_seconds=time.monotonic()-started)
                with s.stats_lock:
                    s.errors += 1
            finally:
                with s.stats_lock:
                    s.active_operation = None
                    s.queue_total += queue_seconds
                    s.compute_total += time.monotonic()-started
        try:
            self.reply(pickle.dumps(reply,protocol=4))
        except (BrokenPipeError,ConnectionResetError):
            pass


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--cpu',type=int,required=True)
    p.add_argument('--token-file',required=True)
    a=p.parse_args()
    os.sched_setaffinity(0,{a.cpu})
    s=ThreadingHTTPServer(('127.0.0.1',a.port),Handler)
    s.daemon_threads=True
    s.token=Path(a.token_file).read_text().strip()
    s.compute_lock=threading.Lock()
    s.stats_lock=threading.Lock()
    s.math=Math()
    s.pending=s.calls=s.errors=0
    s.active_operation=None
    s.operations=Counter()
    s.queue_total=s.compute_total=0.
    s.status={'ready':True,'code_sha256':{str(q):hashlib.sha256(q.read_bytes()).hexdigest()
              for q in [Path(__file__),Path(__file__).with_name('fast_math.py')]}}
    print(json.dumps(dict(port=a.port,**s.math.run({'op':'health'}))),flush=True)
    s.serve_forever()


if __name__=='__main__':
    main()
