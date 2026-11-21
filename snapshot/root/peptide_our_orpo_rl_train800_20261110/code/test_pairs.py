import itertools
import json
import unittest
from build_pairs import construct, pick_negative
from peptide_prompt import build_prompt, measure, parse_action, valid_first_occurrences

REF = 'AKLRAKLR'


def variants(n=100):
    return [''.join(s) + REF[2:] for s in list(itertools.product('ADEFGHIKLMNPQRSTVWY', repeat=2))[:n]]


def observation(sequence, mic, phase='initialization', round=0, cached=True):
    return dict(sequence=sequence, mic=mic, score=-mic, phase=phase, round=round,
                cached=cached, feasible=measure(sequence, REF)[0])


class ContractTests(unittest.TestCase):
    def test_examples(self):
        self.assertTrue(measure('GGLRAKLR', REF)[0])
        self.assertEqual(measure('GGLRAKLR', REF)[1], .75)
        self.assertEqual(measure('GGGRAKLR', REF)[1], .625)
        for seq in ('', 'A'*31, 'AKXR', 'ACCR'):
            self.assertFalse(measure(seq, REF)[0])

    def test_valid_recent50_filters(self):
        xs = variants(80)
        obs = [observation(s, 100+i) for i, s in enumerate(xs[:60])]
        obs += [observation('WWWWWWWW', 1, 'acquisition', 1, False),
                observation(xs[0], 100, 'acquisition', 1, True)]
        obs += [observation(s, 200+i, 'acquisition', 2, False) for i, s in enumerate(xs[60:])]
        _, valid = valid_first_occurrences(obs, REF)
        _, context = build_prompt(REF, valid)
        self.assertEqual([r['sequence'] for r in context['recent50']], xs[-50:])
        self.assertEqual(len(valid), 80)
        self.assertEqual(context['current_best_mic'], 100)

    def test_batch_causality_and_baseline_negative(self):
        xs = variants(65)
        obs = [observation(s, 100+i) for i, s in enumerate(xs[:60])]
        obs += [observation(xs[60], 50, 'acquisition', 1, False),
                observation(xs[61], 70, 'acquisition', 1, False),
                observation(xs[62], 120, 'acquisition', 1, False)]
        h = dict(status='complete', identity=dict(task='synthetic'), reference=REF, observations=obs)
        result = construct('synthetic', h, REF, 'test-hash')
        self.assertEqual(len(result['pairs']), 1)
        pair = result['pairs'][0]
        self.assertEqual(json.loads(pair['chosen']['value'])['sequence'], xs[60])
        # MIC 70 is worse than chosen but improves baseline 100: it must NOT be a rejected target.
        self.assertEqual(json.loads(pair['rejected']['value'])['sequence'], xs[62])
        text = '\n'.join(m['value'] for m in pair['conversations'])
        for seq in xs[60:63]:
            self.assertNotIn(seq, text)
        self.assertEqual(pair['metadata']['prefix_end_exclusive'], 60)
        self.assertEqual(pair['metadata']['current_best_mic'], 100)

    def test_previously_invalid_is_not_novel(self):
        with self.assertRaises(ValueError):
            parse_action('{"sequence":"AKLRAKLR"}', REF, {REF})
        for bad in ('{"sequence":"AKLRAKLR","other":1}', '["AKLRAKLR"]',
                    '{"sequence":"AKLRAKLR","sequence":"AKLRAKLR"}'):
            with self.assertRaises(ValueError):
                parse_action(bad, REF, set())
        self.assertEqual(parse_action('{"sequence":"GGLRAKLR"}', REF, set()), 'GGLRAKLR')

    def test_pareto_matches_bruteforce(self):
        xs = variants(65)
        chosen = dict(sequence=xs[0], mic=10, observation_index=100)
        pool = [dict(sequence=s, mic=20+(i%9), observation_index=i) for i, s in enumerate(xs[1:])]
        pick, meta = pick_negative(chosen, pool)
        from Levenshtein import distance
        lo, hi = min(r['mic']-10 for r in pool), max(r['mic']-10 for r in pool)
        candidates = []
        for r in pool:
            sim = 1-distance(r['sequence'], chosen['sequence'])/max(len(r['sequence']),len(chosen['sequence']))
            gap = (r['mic']-10-lo)/(hi-lo)
            candidates.append((sim,gap,r))
        front = [(s,g,r) for s,g,r in candidates if not any(s1>=s and g1>=g and (s1>s or g1>g) for s1,g1,_ in candidates)]
        expected = max(front, key=lambda x:(.5*x[0]+.5*x[1],x[0],x[1],-x[2]['observation_index']))
        self.assertEqual(pick, expected[2])
        self.assertEqual(meta['pareto_size'], len(front))


if __name__ == '__main__':
    unittest.main()
