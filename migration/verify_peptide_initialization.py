#!/usr/bin/env python3
"""Validate the committed, measured peptide initialization without oracle calls."""
import hashlib
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j-1] + (x != y)))
        previous = current
    return previous[-1]


def main():
    manifest = json.loads((REPO / 'docs/peptide_initialization_manifest.json').read_text())
    documents = {}
    for name, item in manifest['files'].items():
        raw = (REPO / item['snapshot']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item['sha256'], name
        documents[name] = json.loads(raw)
    original, heldout = documents['original'], documents['heldout_revalidated']
    train = set(manifest['train_tasks'])
    test = set(manifest['test_tasks'])
    assert len(train) == 800 and len(test) == 200 and not train & test
    assert set(original) == train | test and set(heldout) == test
    checked = 0
    for name, document in documents.items():
        for task, pool in document.items():
            sequences, scores = pool['sequences'], pool['scores']
            reference = pool['reference']
            assert len(sequences) == len(scores) == len(pool['sources']) == 100, (name, task)
            assert len(set(sequences)) == 100, (name, task)
            if name == 'heldout_revalidated':
                assert reference == original[task]['reference']
                assert sequences == original[task]['sequences']
                assert pool['original_scores'] == original[task]['scores']
            for sequence, score in zip(sequences, scores):
                assert 1 <= len(sequence) <= 30
                assert set(sequence) <= set('ACDEFGHIKLMNPQRSTVWY')
                assert sequence.count('C') <= 1 and sequence != reference
                assert distance(sequence, reference) * 4 <= len(reference)
                assert math.isfinite(score) and score <= 0
                checked += 1
    print(f'PASS: {checked:,} stored entries; 100 unique feasible peptides per task; '
          '800 training / 200 held-out tasks; hashes and revalidated pool alignment.')


if __name__ == '__main__':
    main()
