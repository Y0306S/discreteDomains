"""Shared discrete peptide contract for preference data and future rollout clients."""
import hashlib
import json
import math
from pathlib import Path
import os
from Levenshtein import distance

ALPHABET = frozenset('ACDEFGHIKLMNPQRSTVWY')
SYSTEM = '''Recommend exactly one new discrete peptide amino-acid sequence to minimize the measured APEX bacteria_0 MIC. Lower MIC is better. Use the good, bad, and recent valid evaluated peptides as evidence; do not invent measured results. Return exactly one JSON object with the single key "sequence", for example {"sequence":"AKLR"}. Return no explanation, batch, latent vector, or 3D coordinates. The example is only a format example and must also meet the real task constraints before being proposed.'''


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def measure(sequence, reference):
    sim = 1.0 - distance(sequence, reference) / len(reference)
    valid = (1 <= len(sequence) <= 30 and not (set(sequence) - ALPHABET)
             and sequence.count('C') <= 1 and sim >= .75)
    return bool(valid), sim


def display(row):
    return {key: row[key] for key in ('sequence', 'mic', 'similarity', 'observation_index', 'source_phase')}


def build_prompt(reference, valid_prefix, retry_sequences=()):
    """Prefix contains chronological first measured occurrences, never cached repeats."""
    if len(valid_prefix) < 50:
        raise ValueError('Need at least 50 distinct valid evaluated peptides')
    if len({r['sequence'] for r in valid_prefix}) != len(valid_prefix):
        raise ValueError('Valid history must contain distinct sequences')
    if any(not measure(r['sequence'], reference)[0] or not math.isfinite(r['mic']) for r in valid_prefix):
        raise ValueError('Invalid observation in prompt history')
    ranked = sorted(valid_prefix, key=lambda r: (r['mic'], r['observation_index']))
    context = dict(
        reference_sequence=reference,
        reference_length=len(reference),
        feasible_length_range=dict(minimum=max(1, len(reference)-len(reference)//4), maximum=min(30, len(reference)+len(reference)//4), note='Necessary length bounds only; the actual sequence must still pass the full edit-distance similarity and every other rule.'),
        objective='Minimize APEX bacteria_0 MIC; lower is better; all listed MIC values were measured.',
        current_best_mic=ranked[0]['mic'],
        rules=[
            dict(rule='Use only uppercase amino acids ACDEFGHIKLMNPQRSTVWY.',
                 example='AKLR passes the alphabet rule; AKXR fails because X is not allowed.'),
            dict(rule='Length must be between 1 and 30 amino acids inclusive.',
                 example='AKLR passes the length rule; an empty string or a sequence of 31 A residues fails.'),
            dict(rule='Use at most one cysteine (C).',
                 example='AKCR passes the cysteine rule; ACCR fails because it has two C residues.'),
            dict(rule='Similarity = 1 - Levenshtein_distance(candidate, reference) / length(reference); must be >= 0.75.',
                 example='For illustrative reference AKLRAKLR (length 8), GGLRAKLR has distance 2 and similarity 0.75 (passes); GGGRAKLR has distance 3 and similarity 0.625 (fails). Use the actual reference_sequence for this task.'),
            dict(rule='Recommend a sequence that has never previously been proposed/evaluated in this task, including retry proposals.',
                 example='If AKLRAKLR was already evaluated or proposed, proposing AKLRAKLR again fails novelty. The listed history is a summary; the caller also checks the full history.'),
            dict(rule='Output exactly one JSON object with only the sequence key, containing a single peptide string.',
                 example='{"sequence":"AKLR"} has the required shape; ["AKLR","AKCR"] and a latent vector do not. This example is not a scored recommendation.'),
        ],
        examples_notice='Examples illustrate individual rules only. Every real proposal must satisfy ALL rules against the actual task reference.',
        valid_distinct_evaluations_so_far=len(valid_prefix),
        good=[display(r) for r in ranked[:5]],
        bad=[display(r) for r in sorted(valid_prefix, key=lambda r: (-r['mic'], r['observation_index']))[:5]],
        recent50=[display(r) for r in valid_prefix[-50:]],
        recent50_definition='The latest 50 distinct valid measured peptide sequences in chronological evaluation order. VAE-generated entries have a scored decoded sequence and pass the similarity constraint. Failed decodes, infeasible sequences, and cached repeats are excluded. Initialization sequences are imported measured peptides, not claimed VAE decodes.',
        rejected_retry_proposals=list(retry_sequences),
        request='Recommend one novel feasible peptide sequence that is likely to reduce the current_best_mic. Output only {"sequence":"..."}.',
    )
    return [dict(role='system', content=SYSTEM), dict(role='user', content=json.dumps(context, indent=2, allow_nan=False))], context


def parse_action(text, reference, all_seen):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    value = json.loads(text, object_pairs_hook=unique_keys)
    if not isinstance(value, dict) or set(value) != {'sequence'} or not isinstance(value['sequence'], str):
        raise ValueError('Expected exactly one sequence string')
    sequence = value['sequence']
    if not measure(sequence, reference)[0]:
        raise ValueError('Invalid or infeasible peptide')
    if sequence in all_seen:
        raise ValueError('Duplicate previously proposed/evaluated sequence')
    return sequence


def valid_first_occurrences(observations, reference):
    """Validate every score/constraint; separate all seen proposals from valid measurements."""
    seen, valid, records = set(), [], []
    for index, row in enumerate(observations):
        sequence = row['sequence']
        if not isinstance(sequence, str):
            raise ValueError('Non-string sequence')
        ok, sim = measure(sequence, reference)
        mic, score = float(row['mic']), float(row['score'])
        if not math.isfinite(mic) or not math.isfinite(score) or abs(mic + score) > 1e-6:
            raise ValueError(f'Invalid measured score at {index}')
        if bool(row['feasible']) != ok:
            raise ValueError(f'Archived and recomputed feasibility disagree at {index}')
        first = sequence not in seen
        phase = row['phase']
        if phase not in ('initialization', 'acquisition', 'recenter'):
            raise ValueError('Unknown source phase')
        # A newly encountered cached post-initialization result lacks fresh-evaluation evidence.
        if first and phase != 'initialization' and row['cached']:
            raise ValueError(f'Unexplained cached first occurrence at {index}')
        record = dict(sequence=sequence, mic=mic, similarity=sim,
                      observation_index=index, source_phase=phase,
                      round=row['round'], feasible=ok, first_occurrence=first,
                      decode_evidence=('imported_measured_initialization' if phase == 'initialization'
                          else 'archived_scored_vae_decode_path_no_individual_decode_flag'))
        records.append(record)
        if first and ok:
            valid.append(record)
        seen.add(sequence)
    return records, valid
