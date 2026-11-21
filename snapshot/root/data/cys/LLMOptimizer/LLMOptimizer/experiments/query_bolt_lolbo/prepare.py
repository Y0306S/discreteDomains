"""Freeze a new LOLBO recipe from the audited BOLT30 Query run."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/root/data/cys/baseline/campaigns/query_our_fsbo_vae80_test200_20261117')
TOOLS = Path('/root/data/cys/llmoptimizer_benchmark/tools')
LOLBO = Path('/root/data/cys/baseline/campaigns/taskset_bolt_semanticopt_test200_20261031/code/mindware/components/optimizers/lolbo_turbo_port.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


source_recipe = json.loads((SOURCE / 'batch_recipe.json').read_text())
pool = SOURCE / 'pools/batch10_frozen_pool.json'
pool_audit = json.loads((SOURCE / 'evidence/bolt30_reuse_audit.json').read_text())
vae_status = json.loads((SOURCE / 'training/vae/status.json').read_text())
vae_checkpoint = SOURCE / 'training/vae/best.ckpt'
if not pool_audit.get('passed') or pool_audit.get('measurements') != 6000:
    raise SystemExit('BOLT30 pool audit failed')
if len(source_recipe['tasks']) != 200 or not vae_status.get('checkpoint_sha256'):
    raise SystemExit('Missing frozen task split or Query VAE')

recipe = dict(
    name='query_bolt_lolbo_test200_20261117',
    tasks=source_recipe['tasks'],
    seed=source_recipe['seed'],
    rounds=80,
    batch_size=10,
    warm_start_rows=30,
    timeout_seconds=source_recipe['timeout_seconds'],
    pool_path=str(pool),
    pool_sha256=sha(pool),
    vae_pool=str(SOURCE / 'services/vae_pool.json'),
    vae_checkpoint=str(vae_checkpoint),
    vae_sha256=sha(vae_checkpoint),
    vae_expected_sha256=vae_status['checkpoint_sha256'],
    algorithm='LOLBO DKL GP with BoTorch Thompson sampling and adaptive TuRBO',
    representation='frozen query VAE posterior-mean 64D latent; unbounded LOLBO trust region',
    initial_fit_epochs=20,
    update_fit_epochs=2,
    learning_rate=0.01,
    n_candidates=2000,
    workers=[24, 25, 26, 27, 28],
    pinned_files={
        str(pool): sha(pool),
        str(vae_checkpoint): sha(vae_checkpoint),
        str(LOLBO): sha(LOLBO),
        str(TOOLS / 'query_vae_client.py'): sha(TOOLS / 'query_vae_client.py'),
        str(TOOLS / 'query_vae_service.py'): sha(TOOLS / 'query_vae_service.py'),
        str(SOURCE / 'batch_code/query_data.py'): sha(SOURCE / 'batch_code/query_data.py'),
    },
)
target = ROOT / 'recipe.json'
if target.exists() and json.loads(target.read_text()) != recipe:
    raise SystemExit('Refusing to replace a different frozen recipe')
target.write_text(json.dumps(recipe, indent=2) + '\n')
print(json.dumps({'recipe': str(target), 'tasks': len(recipe['tasks']),
    'rounds': recipe['rounds'], 'batch_size': recipe['batch_size'],
    'new_evaluations': len(recipe['tasks']) * recipe['rounds'] * recipe['batch_size'],
    'pool_sha256': recipe['pool_sha256'], 'vae_sha256': recipe['vae_sha256']}, indent=2))
