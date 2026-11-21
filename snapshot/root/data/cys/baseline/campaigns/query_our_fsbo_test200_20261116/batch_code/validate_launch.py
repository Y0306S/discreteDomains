"""Check resources and immutable dependencies, then pin the revised implementation."""
import json,time
from pathlib import Path
import psycopg
from query_protocol import digest,atomic
ROOT=Path(__file__).resolve().parents[1]
def main():
 old=json.loads((ROOT/'evidence/code_pins.json').read_text());checked={}
 for i,(p,sha) in enumerate(old.items()):
  actual=digest(p)
  if actual!=sha:raise ValueError('Historical pinned dependency changed '+p)
  checked[p]=actual
  if i%50==0:print('Dependencies checked',i+1,'/',len(old),flush=True)
 pg=[]
 for host in ['LLMOptimizer2','LLMOptimizer4']:
  record=json.loads((ROOT/'evidence'/f'resources.{host}.json').read_text())
  if not record['passed']:raise ValueError('Host not ready '+host)
  pg.extend(dict(row,host=host) for row in record['postgres'])
 if len(pg)!=21:raise ValueError('Expected 21 private databases')
 from resources import health
 health()
 extras=list((ROOT/'batch_code').glob('*.py'))+[ROOT/'batch_recipe.json',ROOT/'batch_policy.json',ROOT/'services/replica3.batch10.json',ROOT/'fsbo_recipe.json',ROOT/'services/vae_pool.json',ROOT/'services/warm_start.resources.broker.json',ROOT/'services/orpo.resources.broker.json',ROOT/'services/bolt.resources.json',ROOT/'services/orpo.resources.json',ROOT/'resources.json']+[ROOT/'services'/f'replica{i}.json' for i in range(3)]
 extras += list((ROOT/'services').glob('*.resources.json'))
 train=Path('/root/data/cys/baseline/campaigns/query_our_orpo_rl800_20261112')
 extras += [train/'assets/source_cards.json',train/'assets/source_gp_inventory.json',Path('/root/data/logs/query_transfer_train800_20261106_v1/corpus/manifest.json'),Path('/root/data/logs/query_fsbo_test200_50s_20261106_v1/tasks.json')]
 checked.update((str(p),digest(p)) for p in extras)
 atomic(ROOT/'evidence/batch_code_pins.json',checked);atomic(ROOT/'evidence/batch_launch_readiness.json',dict(passed=True,old_dependencies=len(old),total_pins=len(checked),postgres=pg,at=time.time()));print('READY pins',len(checked),'postgres',len(pg),flush=True)
if __name__=='__main__':main()
