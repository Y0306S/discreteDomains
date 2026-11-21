"""Bounded loopback RPC check across machine 4 -> machine 2 -> machine 3."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mindware.components.optimizers.runtime_rpc import call, server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('check','serve','probe'))
    args = parser.parse_args()
    port = 19680
    if args.mode == 'serve':
        class Echo:
            def health(self):
                return {'identity':'bounded-network-check', 'ready':True}
            def handle(self, request):
                threading.Timer(.5, httpd.shutdown).start()
                return request
        httpd = server(Echo(), port)
        timer = threading.Timer(45, httpd.shutdown)
        timer.start()
        try:
            httpd.serve_forever()
        finally:
            timer.cancel()
            httpd.server_close()
        return
    if args.mode == 'probe':
        for _ in range(15):
            try:
                health = call('http://127.0.0.1:%d/health'%port, timeout=2, retries=0)
                if health['identity'] != 'bounded-network-check':
                    raise ValueError('Unexpected diagnostic port owner')
                result = call('http://127.0.0.1:%d/rpc'%port, {'configuration':{'n_estimators':100}},
                              timeout=2, retries=0)
                assert result == {'configuration':{'n_estimators':100}}
                print(json.dumps({'passed':True, 'route':'LLMOptimizer4 -> LLMOptimizer2 -> LLMOptimizer3',
                                  'gpu_loads':0, 'ml_evaluations':0}))
                return
            except RuntimeError:
                time.sleep(1)
        raise RuntimeError('SSH RPC route did not become ready')
    ssh = ['ssh','-S','none','-o','ControlMaster=no','-o','BatchMode=yes',
           '-o','ConnectTimeout=10','-o','ExitOnForwardFailure=yes']
    script = [sys.executable, str(Path(__file__))]
    children = []
    try:
        children.append(subprocess.Popen(ssh+['LLMOptimizer3', shlex.join(['taskset','-c','38',*script,'serve'])]))
        children.append(subprocess.Popen(ssh+['-NT','-L',f'127.0.0.1:{port}:127.0.0.1:{port}','LLMOptimizer3']))
        children.append(subprocess.Popen(ssh+['-NT','-R',f'127.0.0.1:{port}:127.0.0.1:{port}','LLMOptimizer4']))
        subprocess.run(ssh+['LLMOptimizer4',shlex.join([*script,'probe'])],check=True,timeout=40)
        children[0].wait(timeout=5)
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == '__main__':
    main()
