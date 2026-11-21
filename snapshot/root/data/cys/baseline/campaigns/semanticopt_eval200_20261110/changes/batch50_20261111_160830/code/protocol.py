"""Shared direct-space SemanticOpt prompts, validation, and acquisition."""
import hashlib
import json
import math
import os
from pathlib import Path

BASE = '/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B'
ROOT = Path(__file__).resolve().parents[1]
ALPHABET = frozenset('ACDEFGHIKLMNPQRSTVWY')
SYSTEM = ('You are a black-box optimization assistant. Recommend one configuration in the '
          'specified discrete representation. Return exactly one JSON object with '
          'next_configuration, mean, and std. Predict the objective mean and nonnegative '
          'standard deviation on the original objective scale. After an observed evaluation '
          'failure also include failure_probability between zero and one. Obey every '
          'constraint, avoid evaluated configurations, and minimize the objective. '
          'Do not return latent vectors, explanations, or markdown.')

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def value_hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)

def sig4(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Nonfinite objective or prediction')
    return float(format(value, '.4g'))

def peptide_measure(sequence, reference):
    from Levenshtein import distance
    valid = (isinstance(sequence, str) and 0 < len(sequence) <= 30
             and set(sequence) <= ALPHABET and sequence.count('C') <= 1)
    if not valid:
        raise ValueError('Invalid peptide alphabet, length, or cysteine count')
    similarity = 1 - distance(sequence, reference) / len(reference)
    return dict(similarity=similarity, feasible=similarity >= .75)

def peptide_context(task, reference):
    return dict(task_id=task, domain='antimicrobial_peptide_optimization',
        reference_sequence=reference,
        objective=dict(name='APEX bacteria_0 predicted mean MIC', direction='minimize'),
        configuration_schema=dict(sequence='amino-acid string'),
        constraints=dict(alphabet=''.join(sorted(ALPHABET)), maximum_length=30,
            maximum_cysteine_count=1, minimum_reference_similarity=.75,
            similarity_definition='1 - Levenshtein_distance(sequence, reference) / len(reference)'))

def validate_query_plan(plan, definition, codec, require_canonical=True):
    if (not isinstance(plan, list) or not plan or len(plan) % 5
            or any(type(v) is not int for v in plan)):
        raise ValueError('Plan must contain integer join groups of five')
    if any(plan[i] not in (1, 2, 3) for i in range(4, len(plan), 5)):
        raise ValueError('Unsupported join operator')
    expected = {(t, a) for t, n in definition.query_tables for a in range(1, n+1)}
    if len(plan) != (len(expected)-1)*5:
        raise ValueError('Wrong number of joins')
    for i in range(0, len(plan), 5):
        for table, alias in ((plan[i], plan[i+1]), (plan[i+2], plan[i+3])):
            if not 0 <= table < len(definition.all_tables) or (definition.all_tables[table], alias) not in expected:
                raise ValueError('Plan references a table/alias outside this query')
    tree = codec.decode(definition.query_tables, plan.copy())
    canonical_plan = codec.encode(tree)
    restored = codec.decode(definition.query_tables, canonical_plan.copy())
    if tree.tables_aliases() != expected or not tree.equal(restored):
        raise ValueError('Query plan round-trip changed the join tree')
    if require_canonical and canonical_plan != plan:
        raise ValueError('Plan requires implicit codec repair or is noncanonical')
    return canonical_plan

def select_candidate(answers, best, validate, had_failure=False):
    """Validate two direct-space proposals and evaluate exactly one using weighted EI."""
    if len(answers) != 2:
        raise ValueError('SemanticOpt samples exactly two candidate responses')
    ranked = []
    for answer in answers:
        try:
            validate(answer['next_configuration'])
            mean, std = float(answer['mean']), float(answer['std'])
            failure = float(answer['failure_probability']) if had_failure else 0.
            if not all(math.isfinite(v) for v in (mean, std, failure, best)) or std < 0 or not 0 <= failure <= 1:
                raise ValueError('Invalid predictive moments')
            improvement = best - mean
            if std == 0:
                ei = max(improvement, 0.)
            else:
                z = improvement / std
                ei = improvement * .5 * (1 + math.erf(z / math.sqrt(2))) + std * math.exp(-z*z/2) / math.sqrt(2*math.pi)
            ranked.append(((1-failure)*ei, answer))
        except (KeyError, TypeError, ValueError):
            continue
    if not ranked:
        raise ValueError('Neither proposed candidate passed validation; resample without an oracle call')
    return max(ranked, key=lambda row: row[0])[1]
