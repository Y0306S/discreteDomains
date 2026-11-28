#!/usr/bin/env python3
"""Extract final peptide model parameters from completed local, trusted runs.

Run with the deployed bolt_clean_venv interpreter. Never use this loader for
untrusted checkpoints. Full scientific resume state stays in campaign backups.
"""
import argparse
import json
from pathlib import Path
import sys

from model_catalog import digest

VERSIONS = {
    'online-original': 'peptide_apexgo_recipe_fsbo_test10_20261118',
    'online-rollback': 'peptide_target_loss_gate_20round_max2_test10_20261119',
    'online-feasible-search': 'peptide_fsbo_feasible_search_20261122',
}
KEYS = ('vae', 'fsbo_prior', 'fsbo_low', 'fsbo_scale', 'constraint', 'inducing_count')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/root/data/discreteDomains-final-models/online'))
    args = parser.parse_args()
    sys.path.insert(0, '/root/data/cys/BOLT/optimization/peptides')
    import torch
    torch.set_num_threads(1)
    all_records = []
    for version, campaign in VERSIONS.items():
        root = Path('/root/data/logs')/campaign/'evaluation/adapted_online'
        paths = sorted(root.glob('*/checkpoint.pt'))
        if len(paths) != 10:
            raise RuntimeError(f'Expected ten final checkpoints: {root}')
        records = []
        for path in paths:
            # Completion is recorded in result.json; status.json is a progress
            # record and does not consistently contain a terminal status field.
            status = json.loads((path.parent/'result.json').read_text())
            if status.get('status') != 'complete' or not (path.parent/'result.json').exists():
                raise RuntimeError(f'Not a completed run: {path}')
            state = torch.load(path, map_location='cpu', weights_only=False)
            if state['identity'].get('smoke') or state['identity']['task'] != path.parent.name:
                raise RuntimeError(f'Unexpected checkpoint identity: {path}')
            parameters = {key: state[key] for key in KEYS}
            parameters['identity'] = state['identity']
            folder = args.output/version/path.parent.name
            folder.mkdir(parents=True, exist_ok=True)
            output = folder/'final-model.pt'
            temporary = output.with_suffix('.tmp')
            torch.save(parameters, temporary)
            temporary.replace(output)
            # Validate that export reloads and every tensor/scalar equals source.
            restored = torch.load(output, map_location='cpu', weights_only=False)
            for key in KEYS:
                expected, actual = parameters[key], restored[key]
                if isinstance(expected, dict):
                    assert expected.keys() == actual.keys()
                    assert all(torch.equal(expected[k], actual[k]) for k in expected)
                elif isinstance(expected, torch.Tensor):
                    assert torch.equal(expected, actual)
                else:
                    assert expected == actual
            record = dict(task=path.parent.name, original_checkpoint=str(path),
                          original_sha256=digest(path), export=str(output), sha256=digest(output),
                          bytes=output.stat().st_size, identity=state['identity'],
                          status={k: status[k] for k in ['status', 'rounds', 'proposals', 'fresh_evaluations'] if k in status},
                          contents=list(KEYS),
                          excluded='History, RNG, trust-region state, adaptation events and optimizer/resume state.')
            (folder/'provenance.json').write_text(json.dumps(record, indent=2)+'\n')
            records.append(record)
            print(f'{version}/{path.parent.name}: verified final parameter export', flush=True)
        (args.output/version/'index.json').write_text(json.dumps(records, indent=2)+'\n')
        all_records.extend(records)
    (args.output/'index.json').write_text(json.dumps(all_records, indent=2)+'\n')


if __name__ == '__main__':
    main()
