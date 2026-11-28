"""Create the isolated recipe and export only the original initialization data."""
import copy
import hashlib
import json
from pathlib import Path
import pickle
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent/'peptide_rgpe_screen200_test200_20261119'
sys.path.insert(0, '/root/data/cys/llmoptimizer_benchmark/tools')
import numpy as np
import peptide_local_bo as runner
from source_screen import atomic, digest


def main():
    old = runner.verify(OLD/'rgpe')
    recipe = copy.deepcopy(old)
    tasks = ['pep_104', 'pep_109', 'pep_111', 'pep_132']
    recipe.update(purpose='component_check', allowed_tasks=tasks,
        local_variant='rgpe-top80-bootstrap100-screen1500-EI200-TS50',
        campaign_variant='fixed_init100_top80_bootstrap100_test4',
        rgpe_rank_samples=100, rgpe_sampling_mode='bootstrap',
        source_screen_count=80, source_screen_observations=100,
        source_screen_policy='one_time_mean_only_ordered_pair_ranking_loss',
        pilot_acquisition_rounds=2, baseline_recipe=str(OLD/'rgpe/recipe.json'),
        baseline_recipe_sha256=digest(OLD/'rgpe/recipe.json'))
    # Preserve the 4,000-proposal optimization horizon for weight dilution;
    # stop-after-events limits this pilot to exactly two batches.
    atomic(ROOT/'rgpe/recipe.json', recipe)
    sys.path.insert(0, recipe['cpu_upstream'])
    for task in tasks:
        path = OLD/'rgpe/runs'/task/'runtime.sqlite'
        with sqlite3.connect('file:'+str(path)+'?mode=ro', uri=True) as db:
            state = pickle.loads(db.execute('SELECT value FROM meta WHERE key="state"').fetchone()[0])
        rows = state['rows'][:100]
        reference, sequences, scores = runner.bo.load_pool(recipe, task)
        if (len(rows) != 100 or any(r['phase']!='initialization' for r in rows)
                or [r['sequence'] for r in rows] != sequences
                or not np.array_equal([r['score'] for r in rows], scores)):
            raise ValueError('Baseline initialization differs from the pinned pool: '+task)
        x = runner.transfer.normalized(state['features'][:100], recipe)
        directory = ROOT/'prefilter'/task
        directory.mkdir(exist_ok=True)
        np.savez(str(directory/'initialization.npz'), x=x,
            features=state['features'][:100], mic=-np.asarray(scores),
            sequences=np.asarray(sequences), scores=np.asarray(scores))
    atomic(ROOT/'evidence/preparation.json', dict(passed=True, tasks=tasks,
        baseline_recipe_sha256=digest(OLD/'rgpe/recipe.json'),
        initialization_observations_per_task=100, fresh_oracle_calls=0,
        unchanged_proposal_horizon=recipe['proposals'], pilot_rounds=2))


if __name__ == '__main__':
    main()
