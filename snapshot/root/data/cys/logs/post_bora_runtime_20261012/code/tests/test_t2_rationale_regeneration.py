import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from examples.benchmark.p5_response_format import (
    CHAT_TEMPLATE_VERSION, format_response, parse_response, rationale_conversations, validate_rollout_format,
)
from examples.benchmark.regenerate_t2_rationales import (
    TEACHER_SYSTEM, decode_generated_batch, finalize, prepare, rationale_errors, read_jsonl,
)

DATA = Path('/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data')


class RationaleRegenerationTests(unittest.TestCase):
    def test_formatter_preserves_quoted_braces_and_rejects_multiple_payloads(self):
        config = {'model:category': 'quoted "value" with }', 'model:rate': 0.02}
        answer = format_response('This trades fit against variance.', config)
        self.assertEqual(parse_response(answer), ('This trades fit against variance.', config))
        with self.assertRaises(ValueError):
            parse_response(answer + '\n{"config":{"other":1}}')
        with self.assertRaises(ValueError):
            format_response('', config)

    def test_batch_decode_uses_padded_width_not_attention_mask_length(self):
        tokenizer = SimpleNamespace(decode=lambda ids, **kw: ' '.join(map(str, ids)))
        outputs = [[0, 0, 17, 18, 91, 92], [15, 16, 17, 18, 93, 94]]
        self.assertEqual(decode_generated_batch(tokenizer, outputs, 4), ['91 92', '93 94'])

    def test_detects_empty_label_and_outcome_shortcuts(self):
        for text in ('', 'This repaired configuration is preferred because it is better.',
                     'This configuration achieved a score of 0.98 and therefore clearly performs better than other configurations in this dataset.'):
            self.assertTrue(rationale_errors(text))
        self.assertFalse(rationale_errors(
            'A smaller learning rate can moderate each update, while deeper trees can capture interactions. '
            'This combination may improve flexibility but also increase variance, so validation is needed.'))

    def test_rollout_format_gate_does_not_rewrite_or_drop_malformed_negative_examples(self):
        row = {'response_format': 'rationale_json', 'completion': 'malformed response',
               'chat_template_contract': CHAT_TEMPLATE_VERSION,
               'messages': [{'content': 'Rationale: <explanation>\n\nJSON:\n{"config":{}}'}]}
        validate_rollout_format(row, 'rationale_json')
        self.assertEqual(row['completion'], 'malformed response')
        row['response_format'] = 'json_only'
        with self.assertRaises(ValueError):
            validate_rollout_format(row, 'rationale_json')

    def test_preparation_and_export_are_label_blind_and_keep_both_configurations(self):
        source = read_jsonl(DATA / 'p5_attr_t2.jsonl')[0]
        duplicate = copy.deepcopy(source)
        visible = json.loads(source['conversations'][1]['value'].split(', config=', 1)[1].splitlines()[0])
        duplicate['metadata']['chosen_config'] = visible
        duplicate['chosen']['value'] = format_response('Repeated source candidate.', visible)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src = root / 'source_input.jsonl'
            src.write_text('\n'.join(json.dumps(row) for row in (source, duplicate)) + '\n')
            out = root / 'output'
            prepare(SimpleNamespace(source=src, output=out, train_manifest=DATA / 'ml_train_80_fixed.jsonl',
                                    sft_reference=DATA / 'p5_s1b_sft_train.jsonl'))
            requests = read_jsonl(out / 'requests.jsonl')
            self.assertEqual(len(requests), 2)
            contexts = [row['messages'][1]['content'].split('\n\nFIXED CANDIDATE:')[0] for row in requests]
            self.assertEqual(contexts[0], contexts[1])
            for request in requests:
                self.assertEqual(request['messages'][0]['content'], TEACHER_SYSTEM)
                self.assertNotIn('blamed_hps', json.dumps(request['messages']))
                self.assertNotIn('chosen_score', json.dumps(request['messages']))
                self.assertNotIn('Repair blamed', json.dumps(request['messages']))
            rationales = []
            for request in requests:
                rationales.append({'id': request['id'], 'valid': True, 'teacher_provenance': 'test',
                    'rationale': 'A smaller learning rate can moderate each update, while deeper trees can capture interactions. Validation is needed to assess this trade-off.'})
            (out / 'generations_0.jsonl').write_text('\n'.join(json.dumps(row) for row in rationales) + '\n')
            with self.assertRaises(RuntimeError):
                finalize(SimpleNamespace(output=out, exclude_quarantined=False))
            finalize(SimpleNamespace(output=out, exclude_quarantined=True))
            result = read_jsonl(out / 'p5_attr_t2_rationale.jsonl')[0]
            self.assertIsNone(result['metadata']['chosen_score'])
            self.assertEqual(result['metadata']['source_good_score'], source['metadata']['chosen_score'])
            self.assertEqual(result['metadata']['rejected_score'], source['metadata']['rejected_score'])
            for side in ('chosen', 'rejected'):
                self.assertEqual(parse_response(result[side]['value'])[1], source['metadata'][side + '_config'])
            self.assertNotIn('No rationale.', json.dumps(result['conversations']))
            self.assertNotIn('metadata', read_jsonl(out / 'p5_attr_t2_rationale_train.jsonl')[0])


if __name__ == '__main__':
    unittest.main()
