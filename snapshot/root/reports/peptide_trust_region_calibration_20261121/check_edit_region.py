"""Boundary checks for the isolated pilot gate, including indels and alignment."""
from edit_region import EditRegion

ref = 'KLYFLSFAITLAL'
center = 'KLYWRSFAITALAL'
region = EditRegion(ref, center, 2)
assert region.reference_edits == 3
# Known frozen winner requires substitutions plus deletion relative to this center.
assert region.rejection_reason('KLYWRSFAITKAL', set()) is None
assert EditRegion(ref, center, 1).rejection_reason('KLYWRSFAITKAL', set()) == 'local_edit_limit'
assert region.rejection_reason('AAA', set()) == 'reference_edit_limit'
assert region.rejection_reason('KLYWRSFAITKAL', {'KLYWRSFAITKAL'}) == 'already_observed_or_duplicate'
assert region.rejection_reason('CC', set()) == 'sequence_rule'
selected, rejected = region.filter_batch(['AAA', 'KLYWRSFAITKAL', 'KLYWRSFAITKAL'], set())
assert selected == [1] and [i for i, _ in rejected] == [0, 2]
# Insertion and deletion count as edits, with global distance measured from reference.
region = EditRegion(ref, ref, 1)
assert region.rejection_reason(ref + 'A', set()) is None
assert region.rejection_reason(ref[:-1], set()) is None
assert region.rejection_reason('A' + ref + 'A', set()) == 'local_edit_limit'
try:
    EditRegion(ref, 'AAA', 2)
except ValueError:
    pass
else:
    raise AssertionError('Infeasible search center was accepted')
print('PASS: reference/local boundaries, indels, duplicates and latent-index alignment')
