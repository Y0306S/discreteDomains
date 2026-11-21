"""Strict SemanticOpt proposals with explicit, limited metadata normalization."""

import json
import math
from numbers import Real

from ConfigSpace import Configuration, Constant, UnParametrizedHyperparameter
from ConfigSpace import UniformFloatHyperparameter, UniformIntegerHyperparameter


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate JSON key: ' + key)
        value[key] = item
    return value


def _nonfinite(value):
    raise ValueError('Non-finite JSON number: ' + value)


def finite_number(value, name, lower=None, upper=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + ' must be a finite number')
    if lower is not None and value < lower or upper is not None and value > upper:
        raise ValueError(name + ' is outside its allowed range')
    return value


def validate_configuration(values, cs, algorithm):
    if not isinstance(values, dict):
        raise ValueError('next_configuration must be an object')
    identifier = cs.get_hyperparameter('algorithm')
    if not isinstance(identifier, (Constant, UnParametrizedHyperparameter)) or identifier.value != algorithm:
        raise ValueError('The configuration space must fix the requested algorithm')
    supplied = dict(values)
    changes = []
    if 'algorithm' not in supplied:
        supplied['algorithm'] = algorithm
        changes.append('inserted_fixed_algorithm')
    if supplied['algorithm'] != algorithm:
        raise ValueError('Incorrect fixed algorithm')
    for key, value in supplied.items():
        hp = cs.get_hyperparameter(key)
        if isinstance(hp, UniformIntegerHyperparameter) and type(value) is not int:
            raise ValueError(key + ' must be an integer')
        if isinstance(hp, UniformFloatHyperparameter):
            finite_number(value, key)
    configuration = Configuration(cs, values=supplied)
    cs.check_configuration(configuration)
    actual = dict(configuration)
    if set(actual) != set(supplied):
        raise ValueError('Active parameter names differ; no tunable defaults are allowed')
    for key, value in actual.items():
        original = supplied[key]
        equal = (math.isclose(value, original, rel_tol=1e-12, abs_tol=1e-12)
                 if type(original) in (int, float) else value == original)
        if not equal:
            raise ValueError('ConfigSpace changed parameter ' + key)
    return supplied, changes


def normalize_response(raw_response, cs, algorithm, had_failure):
    payload = json.loads(raw_response, object_pairs_hook=_object, parse_constant=_nonfinite)
    required = {'next_configuration', 'mean', 'std'}
    if not isinstance(payload, dict) or not required <= set(payload) or set(payload) - required - {'failure_probability'}:
        raise ValueError('Incorrect top-level SemanticOpt fields')
    finite_number(payload['mean'], 'mean')
    finite_number(payload['std'], 'std', lower=0)
    configuration, changes = validate_configuration(payload['next_configuration'], cs, algorithm)
    normalized = dict(payload, next_configuration=configuration)
    if had_failure:
        finite_number(payload.get('failure_probability'), 'failure_probability', 0, 1)
    elif 'failure_probability' in normalized:
        normalized.pop('failure_probability')
        changes.append('removed_failure_probability_before_failure')
    return normalized, changes


def objective_feedback(result, task_type):
    """The benchmark already returns a minimization objective; never negate it again."""
    if result['timeout']:
        return {'status': 'failure'}, 'TIMEOUT'
    if result['traceback'] is not None:
        return {'status': 'failure'}, 'FAILED'
    objectives = result['result']['objectives']
    if len(objectives) != 1:
        raise ValueError('Expected one ML objective')
    if isinstance(objectives[0], bool) or not isinstance(objectives[0], Real):
        raise ValueError('The measured objective must be a real number')
    objective = finite_number(float(objectives[0]), 'objective', -1 if task_type == 'cls' else 0,
                              0 if task_type == 'cls' else None)
    return {'status': 'success', 'objective': objective}, 'SUCCESS'


def prompt_observation(row):
    feedback = {'status': row['status']}
    if row['status'] == 'success':
        feedback['objective'] = float(format(row['objective'], '.4g'))
    return feedback


def build_messages(system, profile, cs, algorithm, task_type, total, initial, adaptive):
    from mindware.components.optimizers.transfer_hpo.schema import configspace_spec, schema_spec

    def encode(value):
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

    space = configspace_spec(cs)
    context = {
        'dataset_profile': profile, 'fixed_algorithm': algorithm,
        'objective': {'name': 'negative validation balanced accuracy' if task_type == 'cls'
                      else 'validation mean squared error', 'direction': 'minimize'},
        'total_evaluations_including_initial_observations': total,
        'ConfigSpace': {'hyperparameters': schema_spec(cs), 'conditions': space['conditions'],
                        'forbiddens': space['forbiddens']},
        'initial_evaluations': [dict(configuration=row['configuration'], **prompt_observation(row))
                                for row in initial],
    }
    messages = [{'role': 'system', 'content': system},
                {'role': 'user', 'content': encode(context) + '\nOutput the next configuration.'}]
    for index, row in enumerate(adaptive, len(initial) + 1):
        messages.append({'role': 'assistant', 'content': encode(row['proposal'])})
        messages.append({'role': 'user', 'content': encode(dict(evaluation=index, **prompt_observation(row)))
                         + '\nOutput the next configuration.'})
    return messages
