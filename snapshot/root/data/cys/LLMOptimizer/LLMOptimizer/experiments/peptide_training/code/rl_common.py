"""Shared identities and frozen TaskSet GRPO assets."""
import hashlib
import json
import math
import random
from functools import lru_cache
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = Path('/root/data/cys/llmoptimizer_benchmark')
for path in (REPO / 'examples/benchmark', REPO / 'examples/benchmark/domains'):
    sys.path.append(str(path))


@lru_cache(maxsize=16)
def read(name):
    return json.loads((ROOT / name).read_text())


def require_execution_enabled():
    if read('settings.json').get('status') != 'ready_to_launch':
        raise RuntimeError('RL execution remains on hold at the user request; configuration review and live launch validation are pending')
    from campaign_io import require_live_gates
    require_live_gates(ROOT)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@lru_cache(maxsize=16)
def adapter_digest(parent):
    # Phase adapters are immutable and identified once per worker.
    return digest(Path(parent) / 'adapter_model.safetensors')


def seed_for(group, index):
    return int(hashlib.sha256(f'{group}:{index}:1'.encode()).hexdigest()[:15], 16)


def display_signature(config):
    """The same six-significant-digit precision used by TaskSet prompt histories."""
    shown={k:float(f'{v:.6g}') if isinstance(v,float) else v for k,v in config.items()}
    return json.dumps(shown,sort_keys=True,separators=(',',':'))


def parent_for(phase):
    settings = read('settings.json')
    phases = [p['id'] for p in settings['phases']]
    index = phases.index(phase)
    parent = Path(settings['parent_adapter']) if index == 0 else ROOT / 'models' / phases[index - 1] / 'final'
    if not (parent / 'adapter_model.safetensors').exists():
        raise RuntimeError('Previous phase has not produced a complete adapter: ' + str(parent))
    if index == 0 and settings.get('parent_adapter_sha256') != adapter_digest(parent):
        raise RuntimeError('Original peptide ORPO adapter differs from the verified parent')
    if index and not (parent / 'complete.json').exists():
        raise RuntimeError('Previous phase adapter lacks its completion record')
    if index and json.loads((parent / 'complete.json').read_text())['adapter_sha256'] != adapter_digest(parent):
        raise RuntimeError('Previous phase adapter differs from its committed completion record')
    return parent


def group_improved(records):
    return any(r.get('evaluation_status') == 'SUCCESS' and r['loss'] < r['prefix_best_loss'] for r in records)


def stop_sampling(records):
    if len(records) < 4:
        return False
    return (len(records) >= 10 or group_improved(records)
            or any(r.get('evaluation_status') not in ('SUCCESS', 'PARSE_FAILURE', 'DUPLICATE') for r in records))


def candidate_pool_status(records):
    """Validate the full chronological sampling audit, before selecting four."""
    if not 4 <= len(records) <= 10 or [r.get('sample_i') for r in records] != list(range(len(records))):
        return False, 'incomplete_adaptive_group'
    if any(r.get('evaluation_status') not in ('SUCCESS', 'PARSE_FAILURE', 'DUPLICATE') for r in records):
        return False, 'measurement_failed'
    if any(not r.get('completion', '').strip() for r in records):
        return False, 'empty_completion'
    if not stop_sampling(records):
        return False, 'adaptive_group_not_finished'
    if any(stop_sampling(records[:n]) for n in range(4, len(records))):
        return False, 'sampling_continued_after_stop'
    successes = [r for r in records if r['evaluation_status'] == 'SUCCESS']
    if any(not all(isinstance(r.get(k), (int, float)) and not isinstance(r[k], bool)
                   and math.isfinite(r[k]) for k in ('loss', 'prefix_best_loss')) for r in successes):
        return False, 'invalid_measurement'
    rewards = [(r.get('reward') or {}).get('reward') for r in records]
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in rewards):
        return False, 'invalid_reward'
    return True, 'ok'


def select_training_members(records, group_id, seed=1):
    """ML v14: original four, randomly replace one with a later improver.

    Randomness is local to the group, so worker order and restarts cannot change
    membership. All candidates stay in the audit; only these four train.
    """
    valid, reason = candidate_pool_status(records)
    result = dict(train_members=[], replaced_sample_i=None, injected_sample_i=None,
                  rng_seed=f'{group_id}:{seed}', trainable=False, reason=reason)
    if not valid:
        return result
    members = list(range(4))
    if not group_improved(records[:4]):
        extras = [i for i in range(4, len(records)) if group_improved([records[i]])]
        if extras:
            injected = max(extras, key=lambda i: (records[i]['reward']['delta_best'],
                                                   records[i]['reward']['reward'], i))
            dropped = random.Random(result['rng_seed']).choice(members)
            members = [i for i in members if i != dropped] + [injected]
            result.update(replaced_sample_i=dropped, injected_sample_i=injected)
    rewards = [records[i]['reward']['reward'] for i in members]
    if len({round(v, 12) for v in rewards}) < 2:
        if any(records[i]['evaluation_status'] == 'DUPLICATE' for i in members):
            reason = 'duplicate_auxiliary_only'
        else:
            result['reason'] = 'no_reward_signal'
            return result
    result.update(train_members=members, trainable=True, reason=reason)
    return result


def trainable_group(records):
    selection = select_training_members(records, records[0].get('group_id', '') if records else '')
    return selection['trainable'], selection['reason']
