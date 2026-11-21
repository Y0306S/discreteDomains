import hashlib,json,pickle,sqlite3,sys,time
from pathlib import Path
import numpy as np
root=Path('/root/reports/peptide_rgpe_sobol_recovery_20261102')
c=Path('/root/data/logs/peptide_frozen_local_recenter_test200_20261101_v1/rgpe')
recipe=json.loads((c/'recipe.json').read_text())
sys.path.insert(0,recipe['cpu_upstream'])
(root/'before').mkdir(exist_ok=True)
rows=[]
for log in sorted((c/'dispatch/logs').glob('*.log')):
 if 'Supported range of dimensionality for SobolEngine is [1, 1111]' not in log.read_text(errors='replace'):continue
 task=log.name.split('.')[0]
 src=sqlite3.connect('file:'+str(c/'runs'/task/'runtime.sqlite')+'?mode=ro',uri=True)
 dest=sqlite3.connect(str(root/'before'/f'{task}.sqlite'))
 src.backup(dest);dest.close()
 vals={k:pickle.loads(v) for k,v in src.execute('select key,value from meta')};src.close()
 s=vals['state'];assert vals.get('pending') is None
 assert s['identity']['recipe_sha256']==hashlib.sha256((c/'recipe.json').read_bytes()).hexdigest()
 assert len(s['rows'])==len(s['features'])==len(s['constraint_z'])
 rows.append(dict(task=task,rows=len(s['rows']),progress=s['progress'],identity=s['identity'],
                  history_sha256=hashlib.sha256((c/'runs'/task/'history.json').read_bytes()).hexdigest()))
 if task=='pep_238':np.savez(root/'saved_coordinates.npz',features=s['features'])
(root/'before.json').write_text(json.dumps({'at':time.time(),'tasks':rows},indent=2)+'\n')
print(json.dumps(rows,indent=2))
