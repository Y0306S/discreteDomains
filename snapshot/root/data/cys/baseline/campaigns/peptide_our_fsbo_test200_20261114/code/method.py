"""Peptide warm50 -> independent ORPO/ORPO+RL FSBO runs. No scheduler."""
import copy
import hashlib
import json
import math
import pickle
import sqlite3
from pathlib import Path

import numpy as np
from hybrid_selector import HybridSelector
from peptide_prompt import measure, parse_action, build_prompt, atomic_json
from rl_prompt_policy import build_peptide_rl_prompt


class IncompletePool(RuntimeError):
    pass


class UnresolvedMeasurement(RuntimeError):
    pass


def seed_for(*parts):
    return int(hashlib.sha256(':'.join(map(str, parts)).encode()).hexdigest()[:15], 16)


class Journal:
    """Local trusted checkpoints; an ambiguous oracle call is never replayed."""
    def __init__(self, path, identity):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=60)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY,value BLOB NOT NULL)')
        saved = self.get('identity')
        if saved is not None and saved != identity:
            self.db.close()
            raise ValueError('Task/checkpoint/recipe identity changed')
        self.put('identity', identity)

    def get(self, key, default=None):
        row = self.db.execute('SELECT value FROM kv WHERE key=?', (key,)).fetchone()
        return pickle.loads(row[0]) if row is not None else default

    def put(self, key, value):
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO kv VALUES (?,?)', (key, pickle.dumps(value, protocol=4)))

    def once(self, key, argument, call, *, measurement=False):
        saved = self.get(key)
        if saved is not None:
            if saved['argument'] != argument:
                raise ValueError('Journal operation identity changed: '+key)
            if saved['status'] == 'complete':
                return saved['result']
            if measurement:
                raise UnresolvedMeasurement('Audit interrupted APEX measurement before retry: '+key)
        self.put(key, dict(status='started', argument=argument))
        result = call()
        self.put(key, dict(status='complete', argument=argument, result=result))
        return result

    def close(self):
        self.db.close()


def observed(sequence, mic, reference, index, source):
    ok, similarity = measure(sequence, reference)
    if not ok or not math.isfinite(mic) or mic < 0:
        raise ValueError('Invalid measured feasible peptide')
    return dict(sequence=sequence, mic=float(mic), score=-float(mic), similarity=similarity,
                feasible=True, observation_index=index, source_phase=source)


def warm_start(journal, task, reference, recipe, generate, evaluate):
    settings = recipe['warm_start']
    state = journal.get('warm_state', dict(rows=[], attempts=0, rejected=[]))
    while len(state['rows']) < settings['count'] and state['attempts'] < settings['maximum_samples']:
        index = state['attempts']
        request = dict(stage='warm_start', task=task, attempt=index,
                       seed=seed_for(task, 'warm_start', index, recipe['seed']))
        reply = journal.once('generation/'+str(index), request, lambda: generate(request))
        try:
            sequence = parse_action(reply['text'], reference, {reference} | {r['sequence'] for r in state['rows']})
        except (ValueError, TypeError) as exc:
            state['rejected'].append(dict(attempt=index, reason=str(exc), text=reply['text']))
        else:
            mic = journal.once('measurement/'+str(index), dict(sequence=sequence),
                               lambda: evaluate(sequence), measurement=True)
            state['rows'].append(observed(sequence, mic, reference, len(state['rows']), 'initialization'))
        state['attempts'] += 1
        journal.put('warm_state', state)
    if len(state['rows']) != settings['count']:
        raise IncompletePool(f"{task}: only {len(state['rows'])}/{settings['count']} valid unique samples; no replacement")
    return state


def selector(optimization=None):
    settings = optimization or {}
    return HybridSelector(mode='pucb', bo_arm='fsbo',
                          pucb_p1=settings.get('llm_prior_start', .9),
                          pucb_p_end=settings.get('llm_prior_end', .3),
                          pucb_bo_start=settings.get('fsbo_prior_start', .3),
                          pucb_bo_end=settings.get('fsbo_prior_end', .9),
                          pucb_q='arm_best', pucb_c2=settings.get('pucb_c2', 1.4),
                          rng=np.random.RandomState(1))


def select_arm(bandit, events, attempt, optimization=None):
    # Optimization attempts are counted separately from the50 warm initializations.
    if attempt <= 8:
        return 'llm', dict(reason='first8_post_initialization_llm', schedule_index=None)
    denominator = (optimization or {}).get('schedule_denominator', 91)
    arm = bandit.choose(attempt-9, denominator, higher_is_better=True)
    cooldown = 10 <= attempt <= 20 and events[-1]['selected_arm'] == 'llm'
    details = dict(bandit.snapshot(), reason='llm_cooldown' if cooldown else 'pucb',
                   cooldown_active=cooldown, pucb_proposed_arm=arm, schedule_index=attempt-9,
                   schedule_denominator=denominator)
    return ('fsbo' if cooldown else arm), details


def task_description(task, reference):
    return dict(task_id=task, description=f'Minimize antimicrobial MIC near reference {reference}, length {len(reference)}.',
                objective='minimize APEX bacteria_0 MIC', reference_sequence=reference,
                constraints=dict(alphabet='ACDEFGHIKLMNPQRSTVWY', min_length=1, max_length=30,
                    max_cysteine=1, min_reference_similarity=.75,
                    similarity_metric='1-Levenshtein(candidate,reference)/len(reference)'))


def optimization_prompt(stage, task, reference, rows, attempt, retry_sequences, retrieval):
    if stage == 'orpo':
        return build_prompt(reference, rows, retry_sequences)[0]
    scenario = 'retrieval' if attempt <= 10 else 'normal'
    cards, ranking = retrieval(rows) if scenario == 'retrieval' else ([], None)
    messages, context = build_peptide_rl_prompt(task_description(task, reference), rows,
        scenario=scenario, round_number=attempt, cards=cards,
        allowed_training_tasks=[c['task']['task_id'] for c in cards], retry_sequences=retry_sequences)
    if ranking is not None:
        context['retrieval_ranking'] = ranking
    context['phase_instruction'] = ('Analyze source/target differences before adapting measured top/bottom peptides.'
                                    if scenario == 'retrieval' else
                                    'Balance novel feasible candidates using current measured evidence.')
    messages[1]['content'] = json.dumps(context, indent=2, allow_nan=False)
    return messages


def optimize(journal, stage, task, reference, recipe, pool, generate, evaluate, fsbo, retrieval,
             *, stop_after=None):
    if stage not in ('orpo', 'orpo_rl'):
        raise ValueError('Unknown optimization stage')
    if len(pool) != recipe['warm_start']['count'] or len({r['sequence'] for r in pool}) != len(pool):
        raise ValueError('Require the exact frozen50 pool')
    for row in pool:
        observed(row['sequence'], row['mic'], reference, row['observation_index'], 'initialization')
    state = journal.get('optimization_state', dict(rows=copy.deepcopy(pool), events=[],
                        selector=selector(recipe['optimization'])))
    while len(state['events']) < recipe['optimization']['attempts']:
        attempt = len(state['events'])+1
        bandit = copy.deepcopy(state['selector'])
        selected, details = select_arm(bandit, state['events'], attempt, recipe['optimization'])
        seen = {reference} | {r['sequence'] for r in state['rows']}
        rejected, sequence, actual = [], None, selected
        if selected == 'llm':
            for retry in range(recipe['optimization']['llm_generation_limit']):
                messages = optimization_prompt(stage, task, reference, state['rows'], attempt, rejected, retrieval)
                request = dict(stage=stage, task=task, attempt=attempt, retry=retry, messages=messages,
                               seed=seed_for(task, stage, attempt, retry, recipe['seed']))
                reply = journal.once(f'generation/{attempt}/{retry}', request, lambda: generate(request))
                try:
                    sequence = parse_action(reply['text'], reference, seen | set(rejected))
                    break
                except (ValueError, TypeError):
                    # Keep exact returned sequence when parseable; do not repair model text.
                    try:
                        value = json.loads(reply['text']); candidate = value.get('sequence')
                    except (ValueError, AttributeError):
                        candidate = None
                    if isinstance(candidate, str):
                        rejected.append(candidate)
            if sequence is None:
                actual = 'fsbo'
        diagnostics = None
        if actual == 'fsbo':
            argument = dict(stage=stage, task=task, attempt=attempt,
                            seed=seed_for(task, stage, attempt, 'fsbo', recipe['seed']))
            proposal = journal.once(f'fsbo/{attempt}', argument,
                                    lambda: fsbo(state['rows'], reference, argument['seed']))
            sequence, diagnostics = proposal['sequence'], proposal['diagnostics']
            if sequence is not None:
                sequence = parse_action(json.dumps(dict(sequence=sequence)), reference, seen)
        prior = max(r['score'] for r in state['rows'])
        event = dict(attempt=attempt, selected_arm=selected, actual_arm=actual,
                     rejected_sequences=rejected, selection=details, fsbo_diagnostics=diagnostics)
        if sequence is None:
            event.update(status='NO_VALID_PROPOSAL', sequence=None, mic=None, best_mic=-prior)
            perf = float('-inf')
        else:
            mic = journal.once(f'measurement/{attempt}', dict(sequence=sequence),
                               lambda: evaluate(sequence), measurement=True)
            row = observed(sequence, mic, reference, len(state['rows']), 'acquisition')
            state['rows'].append(row)
            perf = row['score']
            event.update(status='SUCCESS', sequence=sequence, mic=mic, best_mic=-max(prior, perf))
        bandit.update(actual, perf, beat_incumbent=perf>prior)
        if actual != selected:
            bandit.record_attempt(selected)
        state['selector'] = bandit
        state['events'].append(event)
        journal.put('optimization_state', state)
        if stop_after is not None and len(state['events']) >= stop_after:
            break
    return state


def verify_completed_stage(root, stage, recipe_sha, tasks, rounds):
    """Handoff validates each committed task, not just a controller exit."""
    for task in tasks:
        p = Path(root)/stage/'runs'/task/'complete.json'
        if not p.exists():
            raise RuntimeError(f'Previous stage {stage} incomplete: {task}')
        record = json.loads(p.read_text())
        if (record.get('task') != task or record.get('stage') != stage
                or record.get('recipe_sha256') != recipe_sha or record.get('attempts') != rounds):
            raise RuntimeError('Previous stage completion identity/budget mismatch: '+task)
    return True
