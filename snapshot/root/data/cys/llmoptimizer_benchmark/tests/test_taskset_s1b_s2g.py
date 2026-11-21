import importlib.util
import json
from pathlib import Path
import sys

DOMAINS = Path(__file__).resolve().parents[1] / 'examples/benchmark/domains'
sys.path.insert(0, str(DOMAINS))
from taskset_s1b_s2g_common import ML, improvement_events, preference_pair


def row(trial, config, loss, ok=True):
    return dict(trial=trial, config=config, loss=loss, successful=ok)


def test_failed_trials_preserve_causal_position():
    rows = [row(1, {'x': 0.1}, 1.0), row(2, {'x': 0.2}, None, False),
            row(3, {'x': 0.3}, 0.8), row(4, {'x': 0.4}, 0.7)]
    events = list(improvement_events(rows))
    assert len(events) == 1
    assert events[0]['trial'] == 4 and events[0]['position'] == 3
    assert events[0]['prefix_successes'] == 2


def test_selector_matches_original_ml_with_cross_family_candidate(tmp_path):
    space = {'optimizer_family': {'kind': 'categorical'},
             'a:lr': {'kind': 'float', 'low': 0.001, 'high': 1.0, 'log': True},
             'b:lr': {'kind': 'float', 'low': 0.001, 'high': 1.0, 'log': True}}
    configs = [{'optimizer_family': 'a', 'a:lr': x} for x in (0.01, 0.02, 0.1, 0.8)]
    configs.append({'optimizer_family': 'b', 'b:lr': 0.2})
    losses = [1.0, 0.8, 0.4, 0.5, 0.9]
    rows = [row(i + 1, cfg, loss) for i, (cfg, loss) in enumerate(zip(configs, losses))]
    event = list(improvement_events(rows))[0]
    pair = preference_pair(rows, event, space)
    history = str(tmp_path / 'history.json')
    observations = [{'config': c, 'trial_state': 0, 'objectives': [y]} for c, y in zip(configs, losses)]
    prompt = 'Search space:\n- optimizer_family: categorical, choices=[a, b]\n- a:lr: float, range=[0.001, 1.0], log_scale=True\n- b:lr: float, range=[0.001, 1.0], log_scale=True\n\n'
    metadata = dict(history_path=history, improvement_trial=3, task_type='taskset', dataset='task', algorithm='cash', metric='loss')
    original, status = ML.build_pair({'metadata': metadata, 'target_config': configs[2]},
                                   {'messages': [{'content': 'system'}, {'content': prompt}]}, {history: observations})
    assert status == 'ok'
    assert pair['rejected'] == json.loads(original['rejected']['value'])['config']
    for name in ('sim', 'worse_norm', 'weighted', 'pool_size', 'pareto_size'):
        assert pair[name] == original['metadata'][name]
    assert pair['rejected'] not in configs[:3]
    assert pair['chosen_loss'] < pair['rejected_loss']


def test_no_valid_negative_drops_pair():
    rows = [row(1, {'x': 0.1}, 1.0), row(2, {'x': 0.2}, 0.8), row(3, {'x': 0.3}, 0.4)]
    event = list(improvement_events(rows))[0]
    assert preference_pair(rows, event, {'x': {'kind': 'float', 'low': 0.0, 'high': 1.0}}) is None
