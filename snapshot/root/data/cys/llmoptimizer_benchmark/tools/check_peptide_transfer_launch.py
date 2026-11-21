#!/usr/bin/env python3
"""Bind real wrapper checks to a production recipe; do not launch evaluations."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

import numpy as np
import peptide_transfer_eval as adapter
from peptide_transfer_eval import bo, torch
from check_peptide_transfer_eval import archived_state, production_pool_check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--component', required=True, type=Path)
    parser.add_argument('--reports', required=True, type=Path)
    parser.add_argument('--cpu', type=int, default=18)
    parser.add_argument('--server', default='http://127.0.0.1:18871')
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    torch.set_num_threads(1)
    results = {}
    for method in ('fsbo', 'rgpe'):
        directory = args.campaign / method
        if not (directory / 'recipe.json').exists():
            results[method] = dict(ready=False, reason='Source training is incomplete', launched=False)
            continue
        recipe = adapter.verify(directory)
        component = adapter.verify(args.component / method)
        sys.path.insert(0, recipe['cpu_upstream'])
        report_path = args.reports / (method + '_report.json')
        report = json.loads(report_path.read_text())
        if (not report['passed'] or report['method'] != method
                or report['recipe_sha256'] != bo.digest(args.component / method / 'recipe.json')
                or component['code'] != recipe['code']
                or component['corpus_sha256'] != recipe['corpus_sha256']
                or recipe['purpose'] != 'production' or component['purpose'] != 'component_check'):
            raise ValueError('Component checks do not match the production implementation')
        if (component['vae'] != recipe['vae'] or component['prepared_pool'] != recipe['prepared_pool']
                or component['lower'] != recipe['lower'] or component['upper'] != recipe['upper']
                or component['fsbo_weights_sha256'] != recipe['fsbo_weights_sha256']):
            raise ValueError('Source, representation or initialization changed')
        if method == 'rgpe' and (report['source_models'] != 800 or not report['production_scale_passed']):
            results[method] = dict(ready=False, reason='Run complete-bank component/performance checks', launched=False)
            continue
        state = archived_state(args.component / method, report['task'])
        state = copy.deepcopy(state)
        state['identity'] = dict(format=adapter.FORMAT,
            recipe_sha256=bo.digest(directory / 'recipe.json'), task=report['task'])
        model = adapter.ModelClient(directory / 'recipe.json', recipe)
        try:
            adapter.check_memory(recipe, len(state['rows']))
            fit = model.request('fit', x=adapter.normalized(state['features'], recipe),
                score=np.asarray([r['score'] for r in state['rows']]), seed=recipe['seed'] + 2)
            state['model_snapshot'] = model.request('snapshot')
        finally:
            model.close()
        pool = production_pool_check(directory, report['task'], recipe, state, args.server)
        production = dict(report, recipe_sha256=state['identity']['recipe_sha256'],
            component_report=str(report_path), component_report_sha256=bo.digest(report_path),
            component_recipe_sha256=bo.digest(args.component / method / 'recipe.json'),
            production_target_fit=fit, production_pool_check=pool, launched=False,
            production_scale_passed=True,
            gate_checker_sha256=bo.digest(Path(__file__)),
            full_5000_evaluation_campaign_tested=False)
        path = directory / 'preflight_report.json'
        bo.atomic_json(path, production)
        bo.atomic_json(directory / 'preflight_passed.json', dict(recipe_sha256=production['recipe_sha256'],
            report=str(path), report_sha256=bo.digest(path)))
        adapter.preflight(directory)
        results[method] = dict(ready=True, report=str(path), launched=False)
    bo.atomic_json(args.campaign / 'readiness.json', results)
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
