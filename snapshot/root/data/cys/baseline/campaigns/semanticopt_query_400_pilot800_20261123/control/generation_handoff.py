"""Freeze generation coordinators, verify no SQL ambiguity, then retire them for service handoff."""
import json,os,signal,sys,time,sqlite3
from pathlib import Path
ROOT=Path('/root/data/cys/baseline/campaigns/semanticopt_query_400_pilot800_20261123')
host=sys.argv[1];action=sys.argv[2]
def targets():
 rows=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   args=[a.decode() for a in (p/'cmdline').read_bytes().split(b'\0') if a]
   if str(ROOT/'code/evaluate.py') in args or (str(ROOT/'control/run.py') in args and 'slot' in args):rows.append((int(p.name),args))
  except (OSError,UnicodeError):pass
 return rows
if action=='freeze':
 rows=targets()
 for pid,_ in rows:os.kill(pid,signal.SIGSTOP)
 (ROOT/'evidence'/f'{host}.generation_frozen.json').write_text(json.dumps(dict(at=time.time(),processes=[dict(pid=p,argv=a) for p,a in rows]),indent=2))
 print(host,'frozen',len(rows),flush=True)
elif action=='audit':
 import psycopg
 specs=[w for w in json.loads((ROOT/'allocation.json').read_text())['workers'] if w['host']==host]
 for w in specs:
  with psycopg.connect(host='127.0.0.1',port=w['db_port'],dbname='imdb',user='imdb',password='imdb',autocommit=True) as conn:
   active=conn.execute("SELECT pid FROM pg_stat_activity WHERE datname='imdb' AND state='active' AND pid<>pg_backend_pid()").fetchall();assert not active,(host,w['slot'],active)
 rows=[]
 for p in (ROOT/'query/runs').glob('*/observations.sqlite'):
  d=sqlite3.connect(f'file:{p}?mode=ro',uri=True);r=d.execute("SELECT value FROM state WHERE key='pending'").fetchone();pending=json.loads(r[0]) if r else None;assert not pending or pending['status']!='in_flight',(str(p),pending)
  rows.append(dict(task=p.parent.name,records=d.execute('SELECT count(*) FROM records').fetchone()[0],pending=pending));d.close()
 (ROOT/'evidence'/f'{host}.generation_handoff_audit.json').write_text(json.dumps(dict(passed=True,at=time.time(),tasks=rows),indent=2));print(host,'no active SQL or ambiguous journal entries')
elif action=='retire':
 assert (ROOT/'evidence'/f'{host}.generation_handoff_audit.json').exists()
 for p in json.loads((ROOT/'evidence'/f'{host}.generation_frozen.json').read_text())['processes']:
  try:os.kill(p['pid'],signal.SIGTERM);os.kill(p['pid'],signal.SIGCONT)
  except ProcessLookupError:pass
 print(host,'generation coordinators retired')
