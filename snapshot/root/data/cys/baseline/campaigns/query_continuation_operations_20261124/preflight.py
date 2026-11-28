import concurrent.futures,json,os,socket,sys
from pathlib import Path
import fleet
host=sys.argv[1]
def check(pair):
 kind,s=pair
 try:return dict(kind=kind,slot=s['slot'],host=host,ok=True,**fleet.database_check(s))
 except Exception as e:return dict(kind=kind,slot=s['slot'],host=host,ok=False,error=repr(e))
items=[(k,s) for k in fleet.ROOTS for s in fleet.configuration(k)[0] if s['host']==host]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(check,items))
fleet.atomic(fleet.OPS/f'preflight_{host}.json',dict(results=results))
print(json.dumps(results,indent=2))
sys.exit(0 if all(r['ok'] for r in results) else 1)
