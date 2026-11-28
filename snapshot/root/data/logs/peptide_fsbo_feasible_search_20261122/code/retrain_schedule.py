"""Independent, checkpointed online-training patience and total attempt budget."""


def initialize(state):
    return state.setdefault('retrain_schedule', dict(rounds_without_improvement=0, attempts=0))


def observe(state, phase, before, after):
    schedule = initialize(state)
    if after > before:  # Scores are negative MIC, so larger is better.
        schedule['rounds_without_improvement'] = 0
    elif phase == 'acquisition':
        schedule['rounds_without_improvement'] += 1


def due(state, cfg):
    schedule = initialize(state)
    return (schedule['attempts'] < cfg['max_retrain_attempts'] and
            schedule['rounds_without_improvement'] >= cfg['stagnation_rounds'])


def reserve(state, cfg):
    """Persist this reservation with the OLD model before performing any training.

    If interrupted before publication, the checkpoint still holds the old models
    and this rejected attempt. A restart cannot replay it or exceed the cap.
    """
    if not due(state, cfg):
        raise ValueError('Retraining is not due or its attempt budget is exhausted')
    schedule = initialize(state)
    schedule['attempts'] += 1
    event = dict(accepted=False, vae_change_l2=0., status='reserved',
                 rejection_reasons=['interrupted_before_atomic_publication'],
                 attempt=schedule['attempts'],
                 trigger_nonimproving_rounds=schedule['rounds_without_improvement'],
                 round=state['progress']['rounds'], observations=len(state['rows']),
                 fresh_evaluations=state['progress']['fresh_evaluations'])
    schedule['rounds_without_improvement'] = 0
    state['adaptation_events'].append(event)
    return len(state['adaptation_events']) - 1


def recover(state):
    """Mark checkpoint reservations as interrupted; old models are retained."""
    recovered = []
    for event in state['adaptation_events']:
        if event.get('status') == 'reserved':
            event['status'] = 'interrupted_rolled_back'
            recovered.append(event)
    return recovered
