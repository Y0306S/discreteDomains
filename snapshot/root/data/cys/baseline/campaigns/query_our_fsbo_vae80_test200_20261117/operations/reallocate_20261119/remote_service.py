"""Identity-recorded remote service launcher used by resource recovery."""
import argparse,json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('action',choices=['start','stop']);p.add_argument('--name',required=True)
parts=sys.argv[1:];sep=parts.index('--') if '--' in parts else len(parts)
a=p.parse_args(parts[:sep]);a.command=parts[sep+1:]
record=ROOT/'services'/f'{a.name}.remote.json'
if a.action=='stop':
    if record.exists():
        r=json.loads(record.read_text());proc=Path('/proc')/str(r['pid'])
        if proc.exists():
            fields=(proc/'stat').read_text().rsplit(') ',1)[1].split();args=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
            if fields[19]==str(r['start_ticks']) and args==r['command']:os.kill(r['pid'],signal.SIGTERM)
    sys.exit(0)
command=a.command
if command and command[0]=='--':command=command[1:]
assert command
r=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19],command=command,at=time.time())
record.write_text(json.dumps(r,indent=2))
os.execvpe(command[0],command,dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
