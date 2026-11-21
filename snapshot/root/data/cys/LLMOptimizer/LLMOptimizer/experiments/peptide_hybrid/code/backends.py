"""Adapters to existing peptide FSBO, frozen VAE, APEX and Qwen services."""
import functools
import json
import math
import sys
from pathlib import Path

import numpy as np
from method import seed_for
from peptide_prompt import file_sha, measure

TOOLS = Path('/root/data/cys/llmoptimizer_benchmark/tools')
TRAIN = Path('/root/data/cys/baseline/campaigns/peptide_our_orpo_rl800_20261111')


class LLM:
    def __init__(self, recipe, stage, url):
        from qwen_inference_transport import http_json
        self.http = http_json
        self.recipe, self.stage, self.url = recipe, stage, url.rstrip('/')
        self.identity = recipe['models'][stage]['identity']
        health = self.http(self.url+'/health', timeout=30)
        if not health.get('ready') or self.identity not in health.get('identities', []):
            raise RuntimeError('Inference service does not match the pinned peptide model')
        if self.identity['adapter'] and not any(v.get('all_tensors_exact') and
                v.get('sha256') == self.identity['adapter_sha256'] for v in health.get('adapter_verifications', [])):
            raise RuntimeError('Inference adapter tensors have not been verified')

    def __call__(self, request):
        from qwen_inference_transport import digest
        if self.stage == 'warm_start':
            sys.path.insert(0, str(TOOLS))
            from bolt_peptide_sft import messages
            prompt = messages(request['task'], self.recipe['references'][request['task']], 'FORMAT_ONLY')[:2]
        else:
            prompt = request['messages']
        settings = dict(self.recipe['sampling'], enable_thinking=False)
        body = dict(identity=self.identity, messages=prompt, settings=settings,
                    task=self.recipe['name']+'/'+self.stage+'/'+request['task'],
                    rng=dict(seed=request['seed'], cuda_rng=None),
                    # All three checkpoints used LLaMA-Factory qwen3_5_nothink.
                    use_s1b_template=True,
                    namespace=dict(campaign=self.recipe['name'], stage=self.stage, task=request['task'],
                                   attempt=request['attempt'], retry=request.get('retry', 0)))
        body['request_id'] = digest(body)
        response = self.http(self.url+'/generate', body, timeout=3600)
        if response.get('identity') != self.identity or response.get('request_id') != body['request_id']:
            raise RuntimeError('Mismatched inference response')
        return response


@functools.lru_cache(maxsize=1)
def apex():
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, '/root/data/logs/peptide_bo_train800_20261027_v5b/upstream_cpu')
    from apex_oracle.APEX_predict import apex_wrapper
    def evaluate(sequence):
        with torch.inference_mode():
            value = float(apex_wrapper([sequence])[0, 0])
        if not math.isfinite(value) or value < 0:
            raise RuntimeError('Invalid APEX bacteria_0 measurement')
        return value
    return evaluate


class FSBO:
    def __init__(self, recipe, task, server, recipe_sha):
        sys.path.insert(0, str(TOOLS))
        import peptide_transfer_eval as transfer
        import peptide_local_bo as local
        self.transfer, self.local = transfer, local
        self.torch = transfer.torch
        self.torch.set_num_threads(1)
        parent = Path(recipe['fsbo']['parent_recipe'])
        self.config = transfer.verify(parent.parent)
        self.config['local'] = recipe['fsbo']['local']
        self.config.update(initialization=50, batch_size=1,
                           n_candidates=recipe['fsbo']['candidate_pool'])
        self.objective, self.client = transfer.create_objective(self.config, task, server,
                                                               dict(recipe_sha256=recipe_sha))
        self.objective.vae.eval()
        for p in self.objective.vae.parameters():
            p.requires_grad_(False)
        self.model = transfer.ModelClient(parent, transfer.verify(parent.parent))

    def __call__(self, rows, reference, seed):
        import torch
        from peptide_local_policy import candidate_pool, new_region
        from lolbo.utils.bo_utils.turbo import TurboState
        from lolbo.utils.bo_utils.constrained_max_posterior_sampling import MaxPosteriorSampling
        # Fresh deterministic target fits preserve the source checkpoint. No VAE updates.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed % (2**63-1))
            sequences = [r['sequence'] for r in rows]
            features = self.transfer.encode_mean(self.client, sequences)
            constraints = np.asarray([[.75-measure(s, reference)[1]] for s in sequences], dtype=np.float32)
            state = dict(rows=rows, constraint_z=features, constraints=constraints,
                         progress=dict(events=0), last_event_size=len(rows))
            constraint, mll = self.local.make_gp(features)
            # Refit from all current measurements on each FSBO selection (20 epochs).
            constraint = self.local.fit_gp(constraint, mll, state, self.config, constraints)
            fit = self.model.request('fit', x=self.transfer.normalized(features, self.config),
                                     score=np.asarray([r['score'] for r in rows]), seed=seed % (2**31-1))
            region = new_region(rows, constraints, 1, self.config['local'], TurboState)
            center = torch.as_tensor(features[max(range(len(rows)), key=lambda i: rows[i]['score'])])
            rejected = []
            for retry in range(5):
                candidates = candidate_pool(center, region, self.config['n_candidates'],
                                            self.config['local']['perturb_dimensions'])
                bridge = self.transfer.BridgeModel(self.model, self.config)
                with torch.no_grad():
                    z = MaxPosteriorSampling(model=bridge, constraint_models=[constraint],
                        constrained=True, replacement=False)(candidates, num_samples=1)
                    sequence = list(self.objective.vae_decode(z))[0]
                if measure(sequence, reference)[0] and sequence not in set(sequences) | {reference}:
                    return dict(sequence=sequence, diagnostics=dict(fit=fit, rejected_decodes=rejected,
                        candidate_pool=self.config['n_candidates'], decoder_input=z.detach().cpu().tolist()))
                rejected.append(sequence)
            return dict(sequence=None, diagnostics=dict(fit=fit, rejected_decodes=rejected,
                         reason='five_invalid_or_duplicate_FSBO_decodes; no_random_replacement'))

    def close(self):
        try:
            self.model.close()
        finally:
            self.client.close()


class Retrieval:
    def __init__(self, recipe, task, vae_client):
        self.recipe, self.task, self.client = recipe, task, vae_client
        self.cards = json.loads((TRAIN/'assets/source_cards.json').read_text())
        self.inventory = json.loads((TRAIN/'assets/source_gp_inventory.json').read_text())
        self.train = recipe['train_tasks']
        if set(self.cards) != set(self.train):
            raise ValueError('Retrieval source cards must be exactly train800')
        self.models = {}

    def source(self, task):
        if task not in self.models:
            path = TRAIN/'assets/source_gp_mean'/f'{task}.npz'
            meta = json.loads(path.with_suffix('.json').read_text())
            if (not meta['reload_verified'] or meta['source_model_sha256'] != self.inventory[task]['sha256']
                    or file_sha(path) != meta['compact_sha256']):
                raise ValueError('Changed source GP: '+task)
            with np.load(path, allow_pickle=False) as data:
                self.models[task] = {k: data[k].copy() for k in data.files}
        return self.models[task]

    def __call__(self, rows):
        from Levenshtein import distance
        from scipy.spatial.distance import cdist
        from scipy.stats import rankdata
        import peptide_transfer_eval as transfer
        reference = self.recipe['references'][self.task]
        shortlist = sorted(self.train, key=lambda t: (
            -(1-distance(reference, self.cards[t]['task']['reference_sequence']) /
              max(len(reference), len(self.cards[t]['task']['reference_sequence']))),
            abs(len(reference)-len(self.cards[t]['task']['reference_sequence'])), t))[:16]
        by_mic = sorted(rows, key=lambda r:(r['mic'], r['observation_index']))
        evidence = sorted({r['observation_index']:r for r in rows[-50:]+by_mic[:5]+by_mic[-5:]}.values(),
                          key=lambda r:r['observation_index'])
        x = transfer.normalized(transfer.encode_mean(self.client, [r['sequence'] for r in evidence]),
                                 self.recipe['fsbo']['backend_recipe'])
        y = rankdata([r['mic'] for r in evidence])
        ranked = []
        for task in shortlist:
            gp = self.source(task)
            radius = np.sqrt(5.)*cdist(np.asarray(x, dtype=np.float64)/gp['length_scale'],
                                      gp['X']/gp['length_scale'])
            prediction = (gp['amplitude']*(1+radius+radius**2/3)*np.exp(-radius)).dot(gp['alpha'])
            ranks = rankdata(prediction)
            rho = float(np.corrcoef(ranks, y)[0,1]) if np.std(ranks)>0 and np.std(y)>0 else None
            sim = 1-distance(reference, self.cards[task]['task']['reference_sequence']) / max(
                len(reference), len(self.cards[task]['task']['reference_sequence']))
            ranked.append(dict(task=task, spearman=rho, reference_edit_similarity=sim,
                               source_gp_sha256=self.inventory[task]['sha256']))
        ranked.sort(key=lambda r:(-(r['spearman'] if r['spearman'] is not None else -2),
                                  -r['reference_edit_similarity'], r['task']))
        strong = [r for r in ranked if r['spearman'] is not None and r['spearman']>=.5]
        selected = (strong or ranked)[:3]
        ranking = dict(method='reference_shortlist16_then_exact_source_GP_mean_Spearman',
                       scores=ranked, selected_source_ids=[r['task'] for r in selected],
                       confidence='rank_agreement_ge_0.5' if strong else 'weak_or_undefined_agreement_DO_NOT_ASSUME_TRANSFER',
                       target_evidence_indices=[r['observation_index'] for r in evidence],
                       target_future_labels_used=False, source_label_scope='training800_only')
        return [self.cards[r['task']] for r in selected], ranking
