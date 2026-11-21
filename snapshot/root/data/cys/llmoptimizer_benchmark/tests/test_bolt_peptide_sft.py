"""Ranking and provenance checks for measured peptide SFT targets."""

import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('bolt_peptide_sft',
    Path(__file__).resolve().parents[1]/'tools/bolt_peptide_sft.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def observation(sequence, mic, feasible=True):
    return dict(sequence=sequence, mic=mic, score=-mic, feasible=feasible,
                cached=False, phase='acquisition', round=1)


class PeptideTargets(unittest.TestCase):
    def candidates(self):
        reference = 'AAAAAAAAAA'
        rows = [observation('A'*9+letter, float(i+1))
                for i,letter in enumerate('ADEFGHIKLMNP')]
        return reference, rows

    def test_minimum_mic_and_distinct_targets(self):
        reference, rows = self.candidates()
        rows.append(dict(rows[0], cached=True))
        top, excluded, valid = module.select_top10(rows, reference)
        self.assertEqual([r['mic'] for r in top], list(map(float,range(1,11))))
        self.assertEqual(len({r['sequence'] for r in top}),10)
        self.assertEqual(valid,12)
        self.assertEqual(excluded['duplicate_sequence'],1)

    def test_infeasible_low_mic_does_not_win(self):
        reference, rows = self.candidates()
        rows.insert(0, observation('RRRRRRRRRR',0.01,False))
        top, excluded, _ = module.select_top10(rows,reference)
        self.assertEqual(top[0]['mic'],1.0)
        self.assertEqual(excluded['infeasible'],1)

    def test_conflicting_duplicate_is_rejected(self):
        reference, rows = self.candidates()
        rows.append(observation(rows[0]['sequence'], 50.0))
        with self.assertRaisesRegex(ValueError,'conflicting'):
            module.select_top10(rows,reference)

    def test_feasibility_mismatch_is_rejected(self):
        reference, rows = self.candidates()
        rows[0]['feasible'] = False
        with self.assertRaisesRegex(ValueError,'feasibility'):
            module.select_top10(rows,reference)

    def test_direct_sequence_json_and_no_score_in_prompt(self):
        messages = module.messages('pep_001','AAAAAAAAAA','AAAAAAAAAD')
        self.assertEqual(json.loads(messages[-1]['content']),dict(sequence='AAAAAAAAAD'))
        self.assertNotIn('"mic":', messages[1]['content'])
        self.assertNotIn('AAAAAAAAAD', messages[1]['content'])


if __name__ == '__main__':
    unittest.main()
