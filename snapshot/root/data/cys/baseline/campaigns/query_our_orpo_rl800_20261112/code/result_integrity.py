"""Preserve measured results when auxiliary TaskSet diagnostics contain NaN/Inf."""
import copy
import math


def prepare_result(result):
    result = copy.deepcopy(result)
    invalid_objective = result.get('status') == 'success' and (
        not isinstance(result.get('objective'), (int, float))
        or isinstance(result.get('objective'), bool)
        or not math.isfinite(result['objective']))
    if invalid_objective:
        result['status'] = 'failure'
        result['error'] = 'Evaluator reported success without a finite measured objective'
        result['invalid_reported_objective'] = repr(result.get('objective'))
        for key in ('objective', 'perf', 'normalized_score'):
            result.pop(key, None)
    replaced = []

    def clean(value, path):
        if isinstance(value, float) and not math.isfinite(value):
            replaced.append(dict(path=path, value=repr(value)))
            return None
        if isinstance(value, dict):
            return {key: clean(item, path + [key]) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item, path + [index]) for index, item in enumerate(value)]
        return value

    result = clean(result, [])
    if replaced:
        result['nonfinite_diagnostics'] = replaced
    return result
