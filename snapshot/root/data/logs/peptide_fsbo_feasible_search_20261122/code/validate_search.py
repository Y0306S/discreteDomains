"""Real VAE + FSBO search smoke check, with no MIC oracle constructed or called."""
import argparse
import time

from common import *
import numpy as np
import torch
from Levenshtein import distance
import fsbo_search_policy as policy
from evaluate import FSBO, MaxPosteriorSampling
from lolbo.info_transformer_vae_objective import ApexConstrainedDiverseObjective
from lolbo.utils.bo_utils.turbo import TurboState


def main(count):
    started = time.monotonic()
    cfg = prepare()
    setup(seed=cfg['seed'])
    torch.cuda.set_per_process_memory_fraction(.06)
    task = cfg['tasks'][0]
    pool = read('/root/reports/peptide_train800_test200_audit_20261027/test200_revalidated_initialization.json')[task]
    reference = pool['reference']
    vae, data = load_vae(ROOT/'vae/weights.pt')
    # Only bind the production decoding method. Deliberately omit oracle setup.
    decoder = object.__new__(ApexConstrainedDiverseObjective)
    decoder.vae, decoder.dataobj = vae, data
    features = encode(vae, data, pool['sequences'])
    center = torch.as_tensor(features[int(np.argmax(pool['scores']))])
    fsbo = FSBO(ROOT/'fsbo/weights', read(ROOT/'corpus/manifest.json'), max_updates=16)
    fit = fsbo.fit(features, pool['scores'], cfg['seed'])
    def constraints(sequences):
        return np.array([[.75-(1-distance(s, reference)/len(reference))] for s in sequences])
    calls = []
    def decode(x):
        calls.append(len(x))
        return decoder.vae_decode(x)
    region = TurboState(dim=256, batch_size=50)
    candidates, sequences, feasible, search = policy.adaptive_pool(
        center, region, count, decode, constraints,
        lower=cfg['tr_feasibility_lower'], upper=cfg['tr_feasibility_upper'],
        max_adjustments=cfg['tr_max_adjustments'], decode_batch_size=cfg['decode_batch_size'])
    before_select = len(calls)
    selected, decoded, selection = policy.select_batch(
        candidates.cuda(), sequences, feasible, 50,
        lambda pool, n: policy.thompson_indices(fsbo, pool, n, MaxPosteriorSampling))
    checks = dict(
        full_pool_decoded=sum(calls) == count*len(search['probes']),
        bounded_decoder_batches=max(calls) <= cfg['decode_batch_size'],
        at_most_three_adjustments=search['adjustments'] <= 3,
        length_within_limits=cfg['tr_length_min'] <= region.length <= cfg['tr_length_max'],
        exact_batch50=len(selected) == len(decoded) == 50,
        feasible_priority=int((constraints(decoded) <= 0).sum()) == min(50, int(feasible.sum())),
        retained_decodes=len(calls) == before_select,
        valid_peptides=all(s and len(s) <= 30 and s.count('C') <= 1 and
                           not set(s)-set('ACDEFGHIKLMNPQRSTVWY') for s in decoded),
        no_oracle_constructed=not hasattr(decoder, 'objective_function'))
    value = dict(passed=all(checks.values()), checks=checks, task=task,
        candidates=count, search=search, selection=selection, target_fit=fit,
        gpu=torch.cuda.get_device_name(), gpu_peak_bytes=torch.cuda.max_memory_allocated(),
        seconds=time.monotonic()-started, oracle_calls=0,
        scope='real retained offline VAE/FSBO; 16-update target fit for integration only; no online training',
        implementation_sha256={p.name: sha(p) for p in sorted((ROOT/'code').glob('*.py'))})
    write(ROOT/'validation'/f'real_search_{count}.json', value)
    print(json.dumps(value, indent=2))
    assert value['passed'], checks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', type=int, default=5000)
    main(parser.parse_args().candidates)
