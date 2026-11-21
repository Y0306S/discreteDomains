#!/usr/bin/env python3
"""Inspect the verified VAE and latent perturbations, without oracle evaluations."""

import argparse
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

RELEASE = Path('/root/data/logs/peptide_bo_release_20261003')
sys.path.insert(0, str(RELEASE))
sys.path.insert(0, str(RELEASE / 'tools'))
import peptide_bo_campaign as campaign
from mindware.domains.protein.verified_bo import atomic_json, load_pool, similarity


def main(args):
    recipe = campaign.verify_recipe(args.campaign)
    output = dict(recipe_sha256=campaign.digest(args.campaign / 'recipe.json'),
                  oracle_evaluations=0, tasks=[])
    with campaign.resources(args):
        sys.path.insert(0, recipe['upstream'])
        os.chdir(recipe['upstream'])
        from uniref_vae.load_vae import load_vae, vae_forward
        vae, data = load_vae(recipe['vae'], max_string_length=30)
        for task in ('pep_003', 'pep_001', 'pep_014', 'pep_018', 'pep_050'):
            random.seed(1)
            np.random.seed(1)
            torch.manual_seed(1)
            reference, sequences, scores = load_pool(recipe, task)
            with torch.no_grad():
                z, _ = vae_forward(sequences, data, vae)
                samples = vae.sample(z=z)
                decoded = [data.decode(samples[i]) for i in range(len(sequences))]
            case = dict(task=task, reference=reference, raw_exact_reconstructions=sum(
                a == b for a, b in zip(sequences, decoded)), count=len(sequences),
                raw_feasible_reconstructions=sum(similarity(s, reference) >= .75 for s in decoded),
                latent_std=float(z.std()), latent_max_abs=float(z.abs().max()),
                reconstruction_examples=list(zip(sequences[:3], decoded[:3])), perturbations=[])
            center = z[int(np.argmax(scores))]
            # Same sparse coordinate perturbation family as upstream TuRBO, with fixed draws.
            noise = torch.rand(50, 256, device='cuda') * 2 - 1
            mask = torch.rand(50, 256, device='cuda') < 20 / 256
            for radius in (0.0, 3.2, 1.6, 0.4, 0.1, 0.025):
                candidates = center[None, :] + radius * noise * mask
                with torch.no_grad():
                    tokens = vae.sample(z=candidates)
                raw = [data.decode(tokens[i]).replace('X', 'A').replace('-', '') for i in range(50)]
                fixed = []
                for sequence in raw:
                    while sequence.count('C') > 1:
                        index = random.choice([i for i, aa in enumerate(sequence) if aa == 'C'])
                        aa = random.choice(list('ADEFGHIKLMNPQRSTVWY'))
                        sequence = sequence[:index] + aa + sequence[index + 1:]
                    fixed.append(sequence or 'AAA')
                similarities = [similarity(s, reference) for s in fixed]
                case['perturbations'].append(dict(radius=radius,
                    feasible=sum(s >= .75 for s in similarities), samples=50,
                    maximum_similarity=max(similarities), distinct=len(set(fixed))))
            output['tasks'].append(case)
            atomic_json(args.report, output)
            print(json.dumps(case), flush=True)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--cpus', required=True)
    raise SystemExit(main(parser.parse_args()))
