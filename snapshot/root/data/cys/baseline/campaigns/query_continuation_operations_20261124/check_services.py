import concurrent.futures,json,sys,urllib.request
host=sys.argv[1];ports=[23090,*range(23300,23308)]
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def probe(port):
 try:
  with op.open(f'http://127.0.0.1:{port}/health',timeout=30) as r:d=json.load(r)
  return dict(port=port,ok=d.get('ready',d.get('status')=='ready'),status=d.get('status'),replicas=d.get('healthy_replicas'))
 except Exception as e:return dict(port=port,ok=False,error=repr(e))
with concurrent.futures.ThreadPoolExecutor(max_workers=9) as pool:results=list(pool.map(probe,ports))
print(json.dumps(results,indent=2));sys.exit(0 if all(x['ok'] for x in results) else 1)
