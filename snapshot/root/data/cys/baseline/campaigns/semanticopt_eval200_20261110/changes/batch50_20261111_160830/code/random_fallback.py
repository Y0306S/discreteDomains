"""Deterministic constrained random proposals; never fabricate LLM predictions."""
import random
from protocol import ALPHABET, canonical, value_hash


def peptide_proposal(reference, rng):
    # Bounded random edits, followed by the full production validator. This is
    # a mutation distribution, not a uniform sample of the feasible space.
    sequence = list(reference)
    alphabet = sorted(ALPHABET)
    for _ in range(rng.randint(1, max(1, len(reference) // 4))):
        operations = ['substitute']
        if len(sequence) < 30: operations.append('insert')
        if len(sequence) > 1: operations.append('delete')
        operation = rng.choice(operations)
        if operation == 'insert':
            sequence.insert(rng.randrange(len(sequence) + 1), rng.choice(alphabet))
        elif operation == 'delete':
            del sequence[rng.randrange(len(sequence))]
        else:
            i = rng.randrange(len(sequence))
            sequence[i] = rng.choice([a for a in alphabet if a != sequence[i]])
    return dict(sequence=''.join(sequence))


def query_proposal(definition, codec, rng):
    from codec.codec import JoinTreeLeaf, JoinTreeBranch, JoinOperator
    trees = [JoinTreeLeaf(table, alias) for table, count in definition.query_tables
             for alias in range(1, count + 1)]
    graph = definition.schema.query_join_graph
    while len(trees) > 1:
        pairs = [(i, j) for i in range(len(trees)) for j in range(i + 1, len(trees))]
        connected = [(i, j) for i, j in pairs if any(
            graph.has_edge(a, b) for a in trees[i].tables_aliases()
            for b in trees[j].tables_aliases())]
        # A disconnected SQL join graph necessarily needs a Cartesian merge.
        i, j = rng.choice(connected or pairs)
        left, right = trees[i], trees[j]
        if rng.randrange(2): left, right = right, left
        branch = JoinTreeBranch(left, right, rng.choice(list(JoinOperator)))
        del trees[j]; del trees[i]; trees.append(branch)
    return dict(plan=codec.encode(trees[0]))


def choose_fallback(domain, context, validate, seen, rejected, *, seed, number, policy):
    seed_value = value_hash(dict(seed=seed, domain=domain, task=context['task_id'],
                                evaluation=number, policy=policy))
    rng = random.Random(seed_value)
    excluded = set(seen)
    for candidate in rejected:
        try: excluded.add(canonical(candidate))
        except (TypeError, ValueError): pass
    for attempt in range(1, policy['max_random_attempts'] + 1):
        config = (peptide_proposal(context['reference_sequence'], rng)
                  if domain == 'peptide' else validate.sample_random(rng))
        try:
            validate(config)
            if canonical(config) in excluded: continue
        except (ValueError, TypeError, KeyError): continue
        return config, dict(source='random_fallback', seed=seed_value, attempts=attempt,
                            trigger='LLM_candidate_pairs_exhausted',
                            llm_candidate_pairs=policy['after_candidate_pairs'],
                            policy_sha256=policy['sha256'])
    raise RuntimeError('No novel feasible random proposal within the fallback attempt limit')
