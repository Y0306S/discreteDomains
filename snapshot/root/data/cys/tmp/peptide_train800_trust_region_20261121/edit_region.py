"""Opt-in exact sequence gate for a future pilot; not imported by production."""
from dataclasses import dataclass
from Levenshtein import distance

AMINO_ACIDS = frozenset('ACDEFGHIKLMNPQRSTVWY')


@dataclass(frozen=True)
class EditRegion:
    reference: str
    center: str
    local_edits: int = 2

    def __post_init__(self):
        if not self.reference or not self.center or self.local_edits not in (1, 2, 3):
            raise ValueError('Expected nonempty reference/center and local edit budget1,2or3')
        if distance(self.center, self.reference) > self.reference_edits:
            raise ValueError('Search center must satisfy the reference constraint')

    @property
    def reference_edits(self):
        return len(self.reference) // 4

    def rejection_reason(self, sequence, seen):
        if (not sequence or len(sequence) > 30 or sequence.count('C') > 1
                or set(sequence) - AMINO_ACIDS):
            return 'sequence_rule'
        if distance(sequence, self.reference) > self.reference_edits:
            return 'reference_edit_limit'
        if distance(sequence, self.center) > self.local_edits:
            return 'local_edit_limit'
        if sequence in seen:
            return 'already_observed_or_duplicate'
        return None

    def filter_batch(self, sequences, seen):
        # Preserve decoded indices so latent vectors remain aligned with sequences.
        known = set(seen)
        selected, rejected = [], []
        for index, sequence in enumerate(sequences):
            reason = self.rejection_reason(sequence, known)
            if reason is None:
                selected.append(index)
                known.add(sequence)
            else:
                rejected.append((index, reason))
        return selected, rejected
