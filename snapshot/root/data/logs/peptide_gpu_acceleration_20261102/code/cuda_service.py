import argparse,json,os,pickle,threading,time,traceback
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from cuda_math import Math
class Server(ThreadingHTTPServer):
 daemon_threads=True
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def reply(self,payload,kind='application/octet-stream',status=200):
  self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
 def do_GET(self):
  if self.path!='/health':self.reply(b'not found',status=404);return
  self.reply(json.dumps(dict(pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],at=time.time(),calls=self.server.calls,errors=self.server.errors,busy=self.server.busy,**self.server.health)).encode(),'application/json')
 def do_POST(self):
  s=self.server
  if self.path!='/rpc' or self.headers.get('Authorization')!='Bearer '+s.token:self.reply(b'forbidden',status=403);return
  size=int(self.headers.get('Content-Length','0'))
  if not 0<size<512*2**20:self.reply(b'bad size',status=413);return
  req=pickle.loads(self.rfile.read(size));queued=time.monotonic()
  with s.lock:
   s.busy=True
   try:reply=dict(result=s.math.run(req),queue_seconds=time.monotonic()-queued);s.calls+=1
   except Exception:reply=dict(error=traceback.format_exc());s.errors+=1
   finally:s.busy=False
  self.reply(pickle.dumps(reply,protocol=4))
def main():
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--cpu',type=int,required=True);p.add_argument('--token-file',required=True);a=p.parse_args()
 os.sched_setaffinity(0,{a.cpu});s=Server(('127.0.0.1',a.port),Handler);s.token=Path(a.token_file).read_text().strip();s.math=Math();s.health=s.math.run({'op':'health'});s.calls=0;s.errors=0;s.busy=False;s.lock=threading.Lock()
 print(json.dumps(dict(status='ready',port=a.port,**s.health)),flush=True);s.serve_forever()
if __name__=='__main__':main()
