"""Durable direct proposal batches, with per-item constraint/novelty checks."""
import json
from protocol import canonical, sig4
from random_fallback import choose_fallback

SYSTEM = ('You are a black-box optimization assistant. Minimize the specified objective. '
          'Return exactly one compact JSON object with numbered keys "1" through the requested count. '
          'For peptides each value is a sequence string; for query plans each value is an integer array. '
          'Every value must obey ALL task constraints and be distinct from all other values and '
          'from evaluated and already accepted configurations. Never return the unmodified reference peptide. '
          'Use measured feedback to propose diverse improvements. End after the requested final key. '
          'No wrapper keys, explanations, markdown, latent vectors, or predicted measurements.')



def round_state(completed, size=50):
    return dict(completed_rounds=completed // size, round_offset=completed % size, batch_size=size)


def remaining_in_round(completed, total, size=50):
    return min(size - completed % size, total - completed)


def batch_messages(context, initial, rows, total, count, accepted, errors, window):
    good = [r for r in initial + rows if r['status'] == 'success' and not r.get('censored')]
    def compact(row):
        return {k: sig4(v) if k == 'objective' else v for k, v in row.items()
                if k in ('configuration', 'objective', 'status', 'censored', 'feasible', 'similarity')}
    # Incumbent + recent measurements stay inside the validated prompt envelope.
    selected = (initial + rows)[-window:]
    content = dict(**context, completed_proposals=len(rows), proposal_budget=total,
        **round_state(len(rows)), requested_configurations=count,
        best_observation=compact(min(good, key=lambda r:r['objective'])),
        measured_observations=[compact(r) for r in selected],
        already_accepted_in_batch=[a['answer']['next_configuration'] for a in accepted],
        recent_rejections=errors[-10:],
        instruction=f'Return exactly {count} new configurations in one JSON object with numbered keys 1 to {count}. '
                    'The whole batch is proposed before its measurements are available. '
                    'Preserve every task constraint. The evaluator rejects duplicates against the entire history.')
    if 'reference_sequence' in context:
        reference=context['reference_sequence'];length=len(reference)
        content['response_format'] = 'One numbered JSON object, each value a peptide sequence STRING; no nested objects.'
        content['diversity_instruction'] = ('Change residues at DIFFERENT positions across entries. '
            'Do not just vary the last residue; never repeat the original sequence. '
            'Use diverse substitutions, insertions and deletions within the similarity bound.')
        content['preferred_edit_positions_by_entry'] = {
            str(i+1): ([i % length + 1] if i < length or length < 8 else
                [i % length + 1, (i + i//length) % length + 1]) for i in range(count)}
    else:
        content['response_format'] = 'One numbered JSON object, each value an integer plan ARRAY; no nested objects.'
    return [dict(role='system', content=SYSTEM), dict(role='user', content=canonical(content))]


def parse_batch(text):
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    def unique_object(pairs):
        value={}
        for key,item in pairs:
            if key in value:raise ValueError('Duplicate JSON object key')
            value[key]=item
        return value
    value = json.loads(text, object_pairs_hook=unique_object)
    if not isinstance(value,dict):raise ValueError('Expected a JSON object')
    if set(value)=={'configurations'} and isinstance(value['configurations'],list):
        configs=value['configurations']
    elif value and set(value)=={str(i) for i in range(1,len(value)+1)}:
        configs=[value[str(i)] for i in range(1,len(value)+1)]
    else:raise ValueError('Expected numbered JSON configurations or a configurations array')
    # Lossless transport encoding only: never edit sequences or repair plans.
    return [dict(sequence=c) if isinstance(c,str) else dict(plan=c) if isinstance(c,list) else c for c in configs]



def accept_candidates(configs, validate, excluded, needed):
    accepted, errors = [], []
    for config in configs:
        if len(accepted) == needed: break
        try:
            validate(config)
            key = canonical(config)
            if key in excluded: raise ValueError('Duplicate configuration')
            excluded.add(key)
            accepted.append(config)
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(dict(configuration=config, error=str(exc)))
    return accepted, errors


def ensure_batch(db, get, put, *, recipe, domain, context, initial, rows, total,
                 validate, seen, tokenizer, client):
    policy = recipe['batch_policy']; sha = recipe['batch_policy_sha256']
    size = policy['size']; start = len(rows)
    saved = get(db, 'proposal_batch')
    if saved and saved['start'] <= start < saved['start'] + len(saved['items']):
        if saved['policy_sha256'] != sha: raise ValueError('Saved batch policy mismatch')
        return saved
    count = remaining_in_round(start, total, size)
    key = f'batch_build_{start}'
    build = get(db, key, dict(start=start, requested=count, items=[], errors=[], attempts=0, policy_sha256=sha))
    if build['policy_sha256'] != sha: raise ValueError('Partial batch policy mismatch')
    excluded = set(seen) | {canonical(a['answer']['next_configuration']) for a in build['items']}
    while len(build['items']) < count and build['attempts'] < policy['max_requests']:
        needed = count - len(build['items']); attempt = build['attempts']
        request_key = f'batch_request_{start}_{attempt}'
        request = get(db, request_key)
        if request is None:
            window = 32
            messages = batch_messages(context, initial, rows, total, needed, build['items'], build['errors'], window)
            while len(tokenizer(tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False))['input_ids']) > policy['max_input_tokens']:
                if window <= 1: raise ValueError('Batch task context exceeds validated inference envelope')
                window = max(1, window // 2)
                messages = batch_messages(context, initial, rows, total, needed, build['items'], build['errors'], window)
            # Query integer plans can be much longer than peptide strings.
            output_tokens = policy['max_new_tokens'][domain]
            if domain == 'query':
                representative = max((r['configuration'] for r in initial),
                    key=lambda c:len(canonical(c)))
                estimate = len(tokenizer(canonical(dict(configurations=[representative] * needed)))['input_ids'])
                if estimate * 1.2 + 128 > output_tokens:
                    raise ValueError('50 query plans exceed configured JSON output envelope; explicit policy adjustment required')
            request = dict(messages=messages, settings=dict(recipe['generation'], max_new_tokens=output_tokens))
            with db: put(db, request_key, request)
        # A completed durable request is parsed locally on resume. Replaying it
        # through the client would advance the local RNG state and create a new
        # request id instead of using the broker's journaled response.
        if isinstance(request, dict) and isinstance(request.get('response'), str):
            text = request['response']
        else:
            text = client.complete_messages(request['messages'], use_s1b_template=False, settings=request['settings'])
        try:
            configs = parse_batch(text)
            accepted, errors = accept_candidates(configs, validate, excluded, needed)
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            configs, accepted, errors = [], [], [dict(error=str(exc))]
        build['items'].extend(dict(answer=dict(next_configuration=c), provenance=dict(
            source='llm', selection='direct_batch', batch_start=start, batch_size=size,
            request_attempt=attempt+1, batch_policy_sha256=sha)) for c in accepted)
        build['errors'].extend(errors); build['attempts'] += 1
        with db:
            put(db, request_key, dict(request, response=text,
                generation=(client.last_response or {}).get('generation') if hasattr(client, 'last_response') else None))
            put(db, key, build); put(db, 'inference_rng', client.state)
    rejected = [r['configuration'] for r in build['errors'] if 'configuration' in r]
    while len(build['items']) < count:
        fallback = recipe.get('fallback')
        if not fallback or not fallback.get('enabled'): raise RuntimeError('Batch validation exhausted; fallback disabled')
        c, provenance = choose_fallback(domain, context, validate, excluded, rejected,
            seed=recipe['seed'], number=start+len(build['items']), policy=fallback)
        provenance.update(trigger='LLM_batch_requests_exhausted', llm_batch_requests=build['attempts'],
            batch_start=start, batch_size=size, batch_policy_sha256=sha)
        build['items'].append(dict(answer=dict(next_configuration=c), provenance=provenance))
        excluded.add(canonical(c))
    batch = dict(start=start, items=build['items'], policy_sha256=sha, requested=count,
        round=start // size, offset=start % size, llm_requests=build['attempts'])
    with db: put(db, 'proposal_batch', batch)
    return batch
