"""Offline causal preference construction. Does not call the oracle or launch training."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
from pathlib import Path
import tarfile
import time
from Levenshtein import distance
from peptide_prompt import atomic_json, build_prompt, canonical, digest, file_sha, valid_first_occurrences

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/root/data/logs/peptide_bo_train800_20261027_v5b')
AUDIT = Path('/root/reports/peptide_train800_test200_audit_20261027/train800_audit.json')


def load_history(task, expected_sha):
    with tarfile.open(SOURCE / 'runs' / f'{task}.tar.gz', 'r|gz') as archive:
        for member in archive:
            if member.name.rsplit('/', 1)[-1] == 'history.json':
                raw = archive.extractfile(member).read()
                if hashlib.sha256(raw).hexdigest() != expected_sha:
                    raise ValueError(f'{task}: audited history checksum mismatch')
                return json.loads(raw)
    raise ValueError(f'{task}: missing history')


def pick_negative(chosen, pool):
    """ML Pareto rule, using edit similarity for the peptide domain."""
    gaps = [r['mic'] - chosen['mic'] for r in pool]
    lo, hi = min(gaps), max(gaps)
    scored = []
    for row, gap in zip(pool, gaps):
        sim = 1 - distance(row['sequence'], chosen['sequence']) / max(len(row['sequence']), len(chosen['sequence']))
        norm = 1.0 if hi <= lo + 1e-15 else (gap - lo) / (hi - lo)
        scored.append(dict(row=row, similarity_to_chosen=sim, mic_gap=gap,
                           normalized_worse_gap=norm, weighted=.5 * sim + .5 * norm))
    # 2D Pareto in O(n log n). Equal-coordinate candidates are retained.
    ordered = sorted(scored, key=lambda s: (-s['similarity_to_chosen'], -s['normalized_worse_gap']))
    front, best_gap = [], float('-inf')
    for _, group in itertools.groupby(ordered, key=lambda s: s['similarity_to_chosen']):
        group = list(group)
        top_gap = group[0]['normalized_worse_gap']
        if top_gap > best_gap:
            front.extend(s for s in group if s['normalized_worse_gap'] == top_gap)
        best_gap = max(best_gap, top_gap)
    pick = max(front, key=lambda s: (s['weighted'], s['similarity_to_chosen'], s['normalized_worse_gap'], -s['row']['observation_index']))
    meta = {k: v for k, v in pick.items() if k != 'row'}
    meta.update(pool_size=len(pool), pareto_size=len(front), selection='pareto_weighted_0.5sim_0.5worse')
    return pick['row'], meta


def construct(task, history, reference, source_sha):
    if history['status'] != 'complete' or history['identity']['task'] != task or history['reference'] != reference:
        raise ValueError('Incomplete/mismatched archived task')
    records, valid = valid_first_occurrences(history['observations'], reference)
    prefix, all_seen, pairs, contexts = [], set(), [], []
    starts = []
    for _, group in itertools.groupby(records, key=lambda r: (r['source_phase'], r['round'])):
        block = list(group)
        start, end = block[0]['observation_index'], block[-1]['observation_index'] + 1
        starts.append((start, end))
        if block[0]['source_phase'] != 'initialization' and len(prefix) >= 50:
            messages, context = build_prompt(reference, prefix)
            baseline = context['current_best_mic']
            prompt = '\n'.join(m['content'] for m in messages)
            meta = dict(task=task, source_phase=block[0]['source_phase'], source_round=block[0]['round'],
                        prefix_end_exclusive=start, block_end_exclusive=end,
                        source_history_sha256=source_sha, current_best_mic=baseline,
                        all_seen_count=len(all_seen), all_seen_sha256=digest(sorted(all_seen)),
                        valid_prefix_count=len(prefix),
                        visible_indices={key: [r['observation_index'] for r in context[key]] for key in ('good', 'bad', 'recent50')},
                        prompt_sha256=digest(messages))
            contexts.append(dict(**meta, reference_sequence=reference))
            positives = [r for r in block if r['first_occurrence'] and r['feasible']
                         and r['mic'] < baseline and r['sequence'] not in all_seen and r['sequence'] not in prompt]
            if positives:
                chosen = min(positives, key=lambda r: (r['mic'], r['observation_index']))
                # Future measurements provide offline labels only; never enter the prompt prefix.
                pool = [r for r in valid if r['observation_index'] >= start and r['sequence'] not in all_seen
                        and r['mic'] >= baseline and r['sequence'] != chosen['sequence'] and r['sequence'] not in prompt]
                if pool:
                    rejected, selection = pick_negative(chosen, pool)
                    meta.update(chosen_index=chosen['observation_index'], rejected_index=rejected['observation_index'],
                                chosen_mic=chosen['mic'], rejected_mic=rejected['mic'],
                                improvement=baseline - chosen['mic'], negative_selection=selection,
                                source_method='LOLBO', full_prefix_unseen_verified=True,
                                chosen_decode_evidence=chosen['decode_evidence'], rejected_decode_evidence=rejected['decode_evidence'])
                    pairs.append(dict(conversations=[dict(**{'from': 'system'}, value=messages[0]['content']),
                                                     dict(**{'from': 'human'}, value=messages[1]['content'])],
                                      chosen=dict(**{'from': 'gpt'}, value=canonical(dict(sequence=chosen['sequence']))),
                                      rejected=dict(**{'from': 'gpt'}, value=canonical(dict(sequence=rejected['sequence']))),
                                      metadata=meta))
        for row in block:
            if row['first_occurrence'] and row['feasible']:
                prefix.append(row)
            all_seen.add(row['sequence'])
    return dict(task=task, source_history_sha256=source_sha, observations=len(records),
                valid_distinct_measurements=len(valid), pairs=pairs, causal_context_inventory=contexts,
                exclusion_reason=None if pairs else 'no_qualifying_improvement_with_unseen_nonimproving_negative')


def process(item):
    task, reference, sha = item
    result = construct(task, load_history(task, sha), reference, sha)
    atomic_json(ROOT / 'data' / 'tasks' / f'{task}.json', result)
    return dict(task=task, pairs=len(result['pairs']), contexts=len(result['causal_context_inventory']),
                observations=result['observations'], valid_distinct_measurements=result['valid_distinct_measurements'],
                exclusion_reason=result['exclusion_reason'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    started = time.time()
    recipe = json.loads((SOURCE / 'recipe.json').read_text())
    audit = json.loads(AUDIT.read_text())
    checks = {r['task']: r for r in audit['tasks']}
    assert audit['summary']['passed'] and len(recipe['train']) == 800 and len(recipe['test']) == 200
    assert not set(recipe['train']) & set(recipe['test'])
    tasks = recipe['train'][:args.limit] if args.limit else recipe['train']
    jobs = [(t, recipe['references'][t], checks[t]['member_checksums']['history.json']) for t in tasks]
    results, errors = {}, {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process, j): j[0] for j in jobs}
        for future in as_completed(futures):
            task = futures[future]
            try:
                results[task] = future.result()
            except Exception as error:
                errors[task] = repr(error)
            progress = dict(status='building', finished=len(results)+len(errors), total=len(tasks),
                            pairs=sum(r['pairs'] for r in results.values()), errors=errors, seconds=time.time()-started)
            atomic_json(ROOT/'reports/build_progress.json', progress)
            if (len(results)+len(errors)) % 25 == 0 or len(tasks) < 5:
                print(json.dumps(progress), flush=True)
    if errors:
        raise RuntimeError(f'Archive validation failed: {errors}')
    output = ROOT / 'data' / ('orpo_train.jsonl' if not args.limit else 'pilot.jsonl')
    with output.open('w') as stream:
        for task in tasks:
            for row in json.loads((ROOT/'data/tasks'/f'{task}.json').read_text())['pairs']:
                stream.write(canonical(row)+'\n')
    manifest = dict(status='built_pending_independent_audit', source=str(SOURCE), source_method='LOLBO',
                    all_800_scanned=len(tasks)==800, tasks_scanned=len(tasks), tasks_with_pairs=sum(r['pairs']>0 for r in results.values()),
                    pairs=sum(r['pairs'] for r in results.values()), excluded={t:results[t]['exclusion_reason'] for t in tasks if not results[t]['pairs']},
                    split=dict(train=recipe['train'], test=recipe['test']), test_tasks_used=0,
                    oracle_calls=0, training_started=False, source_recipe_sha256=file_sha(SOURCE/'recipe.json'),
                    source_audit_sha256=file_sha(AUDIT), output=str(output), output_sha256=file_sha(output),
                    code_sha256={f.name:file_sha(f) for f in (ROOT/'code').glob('*.py')},
                    negative_rule='unseen valid measured MIC >= prior current best; ML Pareto 50/50 edit similarity and normalized worse MIC gap',
                    recent_rule='last 50 chronological distinct valid measured first occurrences; initialization admitted; cached repeats excluded',
                    decode_provenance='Scored source VAE decode paths for acquisition/recenter; no individual decode flag and no roundtrip claim. Initialization imported measured sequences.',
                    seconds=time.time()-started, per_task=[results[t] for t in tasks])
    atomic_json(ROOT / ('manifest.json' if not args.limit else 'reports/pilot_manifest.json'), manifest)
    atomic_json(ROOT/'data/dataset_info.json', dict(peptide_our_single=dict(file_name='orpo_train.jsonl', formatting='sharegpt', ranking=True,
                columns=dict(messages='conversations', chosen='chosen', rejected='rejected'))))
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('split','per_task','excluded','code_sha256')}), flush=True)


if __name__ == '__main__':
    main()
