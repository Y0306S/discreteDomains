"""Pin automatic refresh of stale smoke evidence, preserving its failed pilot."""

from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
QUERY=Path('/root/data/cys/logs/query_test200_baselines_20261012')
sys.path.insert(0,str(QUERY/'code'))
import query_test200_fleet_v3 as fleet

manifest=fleet.read(ROOT/'manifest.json')
source=ROOT/'manage.py'
record=ROOT/'before_smoke_refresh_manifest.json'
if record.exists():
    raise RuntimeError('Smoke refresh has already been pinned')
for path,sha in manifest['sources'].items():
    if Path(path)!=source and fleet.digest(Path(path))!=sha:
        raise RuntimeError('Unexpected source change: '+path)
fleet.write(record,manifest)
manifest['sources'][str(source)]=fleet.digest(source)
manifest['smoke_refresh']={'time':fleet.stamp(),
    'reason':'Fourteen original smoke records precede the October22 supervisor monitoring fix',
    'action':'Rerun the same 20-proposal pause/reload gate; numerical runner unchanged'}
fleet.write(ROOT/'manifest.json',manifest)
state=ROOT/'state/LLMOptimizer4.random00.json'
old=fleet.read(state)
if old['status']!='failed' or old['error']!='Smoke verification is stale or belongs to different settings':
    raise RuntimeError('Unexpected pilot failure')
fleet.write(ROOT/'pilot_stale_smoke.json',old)
fleet.write(state,{'status':'smoke_pending','time':fleet.stamp(),
                  'review':'Refresh the stale source-bound smoke gate before production'})
