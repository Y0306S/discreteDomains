"""Restricted retrieval evidence for our peptide/query RL and evaluation prompts.

This helper is not a retrieval selector or rollout trainer. Callers supply verified
measured top/bottom examples and the allowed training task IDs. Unknown task/card
fields are rejected so old region summaries cannot silently pass through.
"""
import json
import math
from peptide_prompt import build_prompt, measure

TASK_FIELDS = {
    'peptide': {'task_id', 'description', 'objective', 'constraints', 'reference_sequence'},
    'query': {'task_id', 'description', 'objective', 'constraints', 'query_text', 'search_space'},
}


def validate_task(domain, task):
    if domain not in TASK_FIELDS or set(task) != TASK_FIELDS[domain]:
        raise ValueError('Require an explicit domain task description with no extra region fields')
    if any(task[key] is None or task[key] == '' for key in TASK_FIELDS[domain]):
        raise ValueError('Missing task metadata; do not invent source/target differences')
    if not isinstance(task['constraints'], dict) or not task['constraints']:
        raise ValueError('Explicit task constraints required')
    if not isinstance(task['task_id'], str) or not isinstance(task['description'], str) or not isinstance(task['objective'], str):
        raise ValueError('Task identity, description and objective must be text')
    if domain == 'peptide':
        constraints = task['constraints']
        expected = dict(alphabet='ACDEFGHIKLMNPQRSTVWY', min_length=1, max_length=30,
                        max_cysteine=1, min_reference_similarity=.75,
                        similarity_metric='1-Levenshtein(candidate,reference)/len(reference)')
        if constraints != expected or task['objective'] != 'minimize APEX bacteria_0 MIC':
            raise ValueError('Peptide contract differs from audited training rules')
        if not isinstance(task['reference_sequence'], str) or not task['reference_sequence']:
            raise ValueError('Peptide reference required')
    elif not isinstance(task['query_text'], str) or not task['query_text'].strip() or not isinstance(task['search_space'], dict) or not task['search_space']:
        raise ValueError('Actual query text and discrete search space required')
    json.dumps(task, allow_nan=False)


def differences(domain, current, source):
    """Describe actual metadata differences, never invented biological/SQL claims."""
    return [dict(field=key, same=current[key] == source[key],
                 current_task_value=current[key], similar_task_value=source[key])
            for key in sorted(TASK_FIELDS[domain] - {'task_id'})]


def validate_examples(domain, examples, source):
    if not isinstance(examples, list) or not examples:
        raise ValueError('Measured top and bottom examples required')
    expected = {'candidate', 'measured_value'}
    for example in examples:
        if set(example) != expected:
            raise ValueError('Examples may contain only a discrete candidate and its source measured value')
        value = example['measured_value']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Finite source measurements required')
        candidate = example['candidate']
        if domain == 'peptide':
            if not isinstance(candidate, str) or not measure(candidate, source['reference_sequence'])[0]:
                raise ValueError('Source peptide must satisfy source feasibility')
        elif not isinstance(candidate, dict) or not candidate:
            raise ValueError('Query examples must be concrete discrete configurations')


def retrieval_evidence(domain, current_task, cards, allowed_training_tasks):
    validate_task(domain, current_task)
    if not cards:
        raise ValueError('Similar-task stage requires real measured source-task evidence')
    result, seen = [], set()
    for card in cards:
        if set(card) != {'task', 'top', 'bottom'}:
            raise ValueError('Only source task description and measured top/bottom candidates allowed')
        source = card['task']
        validate_task(domain, source)
        task_id = source['task_id']
        if task_id == current_task['task_id'] or task_id not in allowed_training_tasks or task_id in seen:
            raise ValueError('Source must be a distinct allowed training task, used once')
        seen.add(task_id)
        validate_examples(domain, card['top'], source)
        validate_examples(domain, card['bottom'], source)
        result.append(dict(similar_task=source, differences_from_current_task=differences(domain, current_task, source),
                           top=card['top'], bottom=card['bottom'],
                           score_scope='Measured on the similar task only; not current-task scores.'))
    evidence = dict(current_task=current_task, similar_tasks=result,
        instructions=[
            'Before choosing a candidate, compare each source task with the current task using the explicit differences. Identify transferable patterns, conflicting constraints, and unknowns; do not invent causal claims from top/bottom examples.',
            'Adapt the evidence to the CURRENT task. Do not blindly copy a source candidate. Check current-task constraints and the full novelty history. Source feasibility does not establish current-task feasibility.',
            'For peptides, recompute similarity to the CURRENT reference; for queries, use CURRENT query text and allowed configuration space.',
            'Perform the comparison before recommending; output only the existing single-candidate JSON, without analysis text or extra keys.'
        ])
    # Detach input objects; later caller mutation must not insert extra fields.
    return json.loads(json.dumps(evidence, allow_nan=False))


def build_peptide_rl_prompt(task, valid_prefix, *, scenario, round_number,
                            cards=(), allowed_training_tasks=(), retry_sequences=()):
    validate_task('peptide', task)
    if scenario not in ('retrieval', 'normal', 'explore', 'exploit'):
        raise ValueError('Unknown RL scenario')
    if isinstance(round_number, bool) or not isinstance(round_number, int) or round_number < 1:
        raise ValueError('Require a positive 1-based round number')
    messages, context = build_prompt(task['reference_sequence'], valid_prefix, retry_sequences)
    context['current_task'] = task
    context['scenario'] = scenario
    context['round_number'] = round_number
    uses_retrieval = scenario == 'retrieval' or (scenario == 'exploit' and round_number > 50)
    if uses_retrieval:
        context['similar_task_evidence'] = retrieval_evidence('peptide', task, cards, allowed_training_tasks)
    elif cards:
        raise ValueError('Source evidence supplied outside RS or post-50 exploitation')
    messages[1]['content'] = json.dumps(context, indent=2, allow_nan=False)
    return messages, context
