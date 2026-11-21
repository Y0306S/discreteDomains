import json,time,subprocess,os
from pathlib import Path
p=Path('/root/data/cys/tmp/query_vae_control_20261121/original/fsbo/status.json')
start=time.monotonic()
while json.loads(p.read_text()).get('status')!='complete':
 if time.monotonic()-start>900:raise RuntimeError('Control training did not complete in15minutes')
 time.sleep(5)
raise SystemExit(subprocess.call(['/root/data/cys/baseline/envs/bl_fsbo_exact/bin/python','-u','/root/data/cys/tmp/query_vae_control_20261121/target_probe.py']))
