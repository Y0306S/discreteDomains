"""Completion checks for truncated, padded, cached, and supervised rationales."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/benchmark/domains'))
from taskset_rationale_validation import (generation_details, generation_error,
                                        rationale_text, record_error, text_error)
from train_taskset_s1b_s2g import (atomic_json, audit_rationales, digest,
                                 digest_arrays, validate_rationale_gate)


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(chr(i) for i in ids if i not in (0, 1))


class RationaleValidation(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()

    def record(self, text='Rationale: A smaller learning rate may help.'):
        ids = self.tokenizer.encode(text) + [0]
        details = generation_details(self.tokenizer, ids, 192, [0])
        return dict(event_index=0, event_sha256=digest_arrays({'event': 0}),
                    rationale=rationale_text(text), attempts=[text],
                    generation_attempts=[details], validation_version=2)

    def test_cutoff_is_rejected_even_after_complete_sentence(self):
        ids = self.tokenizer.encode('This looks complete.')
        details = generation_details(self.tokenizer, ids, len(ids), [0])
        self.assertEqual(generation_error(details), 'generation_length')

    def test_eos_at_limit_and_batch_padding(self):
        ids = self.tokenizer.encode('A complete sentence.') + [0]
        details = generation_details(self.tokenizer, ids + [1, 1], len(ids), [0])
        self.assertIsNone(generation_error(details))
        self.assertEqual(details['generated_tokens'], len(ids))
        self.assertEqual(details['token_ids'], ids)

    def test_eos_does_not_allow_incomplete_sentence(self):
        details = generation_details(self.tokenizer,
                                     self.tokenizer.encode('This optimizer family has') + [0], 192, [0])
        self.assertEqual(generation_error(details), 'incomplete_sentence')

    def test_multiline_is_rejected_instead_of_shortened(self):
        self.assertEqual(text_error('Rationale: First line.\nMore text.'), 'multiline_response')
        self.assertEqual(record_error(dict(attempts=['First line.\nMore text.'], rationale='First line.'),
                                      self.tokenizer), 'stored_rationale_differs_from_response')

    def test_legacy_cap_and_completion(self):
        record = dict(attempts=['A complete sentence.'], rationale='A complete sentence.', max_new_tokens=192)
        self.assertIsNone(record_error(record, self.tokenizer))
        record['max_new_tokens'] = len(record['rationale'])
        self.assertEqual(record_error(record, self.tokenizer), 'legacy_response_at_token_limit')

    def test_generation_evidence_is_checked(self):
        record = self.record()
        self.assertIsNone(record_error(record, self.tokenizer))
        modified = copy.deepcopy(record)
        modified['generation_attempts'][0]['generated_tokens'] -= 1
        self.assertEqual(record_error(modified, self.tokenizer), 'generation_evidence_mismatch')

    def test_failed_first_attempt_then_complete_retry(self):
        record = self.record()
        failed = generation_details(self.tokenizer, self.tokenizer.encode('The family has'), 14, [0])
        record['generation_attempts'].insert(0, failed)
        record['attempts'].insert(0, failed['text'])
        self.assertIsNone(record_error(record, self.tokenizer))

    def test_audit_gate_and_append_only_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, campaign = root / 'data', root / 'campaign'
            data.mkdir()
            events = [{'event': 0}]
            (data / 'sft_events.jsonl').write_text(json.dumps(events[0]) + '\n')
            record = self.record()
            invalid = dict(record, rationale='This optimizer family has',
                           attempts=['Rationale: This optimizer family has'], validation_version=1)
            cache = data / 'rationales.jsonl'
            cache.write_text(json.dumps(invalid) + '\n')
            with self.assertRaisesRegex(ValueError, 'completion audit failed'):
                audit_rationales(data, campaign, self.tokenizer, events, require_complete=True)
            with cache.open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            records, audit = audit_rationales(data, campaign, self.tokenizer, events, require_complete=True)
            self.assertTrue(audit['passed'])
            self.assertEqual(records[0]['rationale'], record['rationale'])
            self.assertEqual(audit['eos_verified_records'], 1)
            training = data / 'sft_train.jsonl'
            training.write_text('{}\n')
            atomic_json(data / 'sft_manifest.json', dict(examples=1, training_sha256=digest(training),
                        rationale_audit_sha256=digest(campaign / 'rationale_audit.json')))
            validate_rationale_gate(data, campaign)
            with cache.open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            with self.assertRaisesRegex(ValueError, 'unchanged rationale audit'):
                validate_rationale_gate(data, campaign)


if __name__ == '__main__':
    unittest.main()
