import copy
import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from common import ROOT,BOLT,OPS
sys.path.insert(0,str(OPS));sys.path.insert(0,str(BOLT))
import run_lolbo as runner
import random_feasible_fallback as fallback

class FakeVAE:
    def encode(self,plans):return [[0.0]*64 for _ in plans]

class FallbackTests(unittest.TestCase):
    def test_native_random_plans_are_unique_and_valid_for_both_tasks(self):
        for task in ['CEB_2B326','CEB_2C202']:
            rows=runner.load_pool(task);seen={runner.plan_key(r['plan']) for r in rows}
            rng=random.Random(41)
            for _ in range(10):
                plan=fallback.draw_random(runner,task,seen,rng)
                runner.canonicalize(task,plan,strict=True)
                self.assertNotIn(runner.plan_key(plan),seen);seen.add(runner.plan_key(plan))

    def test_random_fill_preserves_accepted_latent_items(self):
        task='CEB_2B326';rows=runner.load_pool(task);rng=random.Random(5)
        seen={runner.plan_key(r['plan']) for r in rows}
        retained=dict(plan=fallback.draw_random(runner,task,seen,rng),proposed_z=[1.0]*64,proposal_source='lolbo')
        checkpoint=dict(accepted=[retained],rejected=[],fit={})
        policy=dict(max_random_draws=100,policy_path='test',policy_sha256='test')
        with tempfile.TemporaryDirectory() as folder:
            fallback.fill_random(runner,task,dict(rows=rows),FakeVAE(),71,checkpoint,Path(folder)/'p',policy)
        self.assertEqual(checkpoint['accepted'][0],retained)
        self.assertEqual(len(checkpoint['accepted']),10)
        self.assertEqual(checkpoint['fit']['random_fallback_count'],9)
        self.assertEqual(len({runner.plan_key(i['plan']) for i in checkpoint['accepted']}),10)

    def test_random_draw_limit_fails_closed(self):
        checkpoint=dict(accepted=[],rejected=[],fit={})
        with tempfile.TemporaryDirectory() as folder,patch.object(fallback,'draw_random',side_effect=ValueError('duplicate')):
            with self.assertRaises(RuntimeError):
                fallback.fill_random(runner,'CEB_2B326',dict(rows=[]),FakeVAE(),71,checkpoint,
                    Path(folder)/'p',dict(max_random_draws=3))
        self.assertEqual(checkpoint['random_draws'],3)
        self.assertEqual(checkpoint['accepted'],[])

    def test_random_generation_is_repeatable(self):
        task='CEB_2C202'
        self.assertEqual(fallback.draw_random(runner,task,set(),random.Random(91)),
            fallback.draw_random(runner,task,set(),random.Random(91)))

    def test_other_tasks_are_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder,fallback.POLICY_NAME).write_text(json.dumps(dict(tasks={'CEB_2B326':{}})))
            original=lambda *args:None
            fake=types.SimpleNamespace(next_batch=original)
            fallback.install(fake,'CEB_2A350',folder)
            self.assertIs(fake.next_batch,original)

    def test_invalid_encoder_response_fails_closed(self):
        task='CEB_2C202';vae=types.SimpleNamespace(encode=lambda p:[[float('nan')]*64 for _ in p])
        checkpoint=dict(accepted=[],rejected=[],fit={})
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                fallback.fill_random(runner,task,dict(rows=[]),vae,65,checkpoint,Path(folder)/'p',
                    dict(max_random_draws=100))

if __name__=='__main__':unittest.main()
