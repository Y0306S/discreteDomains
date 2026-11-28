import json,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent
for _ in range(60):
 states=[json.loads((p/f'status_{i}.json').read_text()) for i in range(4)]
 if all(s['status']=='complete' for s in states):break
 time.sleep(3)
else:raise TimeoutError('Decoder shards not complete within3minutes')
for name in ['validate_raw.py','aggregate.py','write_report.py']:
 subprocess.run(['/root/data/cys/envs/bolt_clean_venv/bin/python',str(p/name)],check=True)
