import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import evaluate
from common import ROOT, load
from protocol import atomic, canonical, digest, peptide_measure
from random_fallback import choose_fallback, peptide_proposal

POLICY=dict(enabled=True,after_candidate_pairs=5,max_random_attempts=10000,sha256='test-policy')


class FallbackChecks(unittest.TestCase):
    def test_reference_mutations_pass_full_constraints_and_vary_length(self):
        lengths=set()
        reference='AKLR'*5
        for seed in range(300):
            rng=random.Random(seed)
            config,meta=choose_fallback('peptide',dict(task_id='p',reference_sequence=reference),
                lambda c: self.require_valid(c,reference),{canonical(dict(sequence=reference))},[],
                seed=seed,number=0,policy=POLICY)
            self.assertTrue(peptide_measure(config['sequence'],reference)['feasible'])
            self.assertNotEqual(config['sequence'],reference)
            lengths.add(len(config['sequence']))
        self.assertGreater(len(lengths),1)

    @staticmethod
    def require_valid(config,reference):
        if not peptide_measure(config['sequence'],reference)['feasible']:raise ValueError('infeasible')

    def test_reproducible_and_excludes_seen_and_rejected(self):
        context=dict(task_id='p',reference_sequence='AKLR'*4)
        kwargs=dict(seed=42,number=10,policy=POLICY)
        validate=lambda c:self.require_valid(c,context['reference_sequence'])
        first,meta=choose_fallback('peptide',context,validate,set(),[],**kwargs)
        self.assertEqual((first,meta),choose_fallback('peptide',context,validate,set(),[],**kwargs))
        second,_=choose_fallback('peptide',context,validate,{canonical(first)},[],**kwargs)
        third,_=choose_fallback('peptide',context,validate,set(),[first,second],**kwargs)
        self.assertNotIn(canonical(third),{canonical(first),canonical(second)})

    def test_exhaustion_never_returns_invalid(self):
        def reject(config):raise ValueError('invalid')
        with self.assertRaisesRegex(RuntimeError,'No novel feasible'):
            choose_fallback('peptide',dict(task_id='p',reference_sequence='AKLR'*4),reject,set(),[],
                seed=42,number=0,policy=dict(POLICY,max_random_attempts=8))

    def test_real_query_codec_all200(self):
        recipe=load(ROOT/'recipe.json')
        count=0
        for task in recipe['query']['test']:
            context,initial,validate,_=evaluate.query_setup(task,recipe['query'],0)
            seen={canonical(r['configuration']) for r in initial}
            for n in range(3):
                candidate,_=choose_fallback('query',context,validate,seen,[],seed=42,number=n,policy=POLICY)
                validate(candidate)
                self.assertNotIn(canonical(candidate),seen)
                seen.add(canonical(candidate));count+=1
        self.assertEqual(count,600)

    def test_all200_peptide_references(self):
        recipe=load(ROOT/'recipe.json');pool=load(recipe['peptide']['pool'])
        for task in recipe['peptide']['test']:
            data=pool[task];reference=data['reference']
            seen={canonical(dict(sequence=s)) for s in data['sequences']}
            for n in range(5):
                candidate,_=choose_fallback('peptide',dict(task_id=task,reference_sequence=reference),
                    lambda c:self.require_valid(c,reference),seen,[],seed=42,number=n,policy=POLICY)
                self.assertNotIn(canonical(candidate),seen)
                seen.add(canonical(candidate))

    def test_production_retry_fallback_commit_resume_no_fake_prediction(self):
        recipe=copy.deepcopy(load(ROOT/'recipe.json'))
        recipe.update(fallback=POLICY,runtime_policy_sha256='test-policy')
        recipe['peptide'].update(test=['p'],proposals=3)
        reference='AKLR'*4
        initial=[dict(configuration=dict(sequence=reference),objective=10.,status='success',censored=False,feasible=True)]
        calls=[];generations=[]
        class Client:
            def __init__(self,*args):self.state={}
            def import_state(self,value):self.state=value or {}
            def ensure_loaded(self):pass
            def complete_messages(self,*args,**kwargs):
                generations.append(1)
                return canonical(dict(next_configuration=dict(sequence='INVALID'),mean=5.,std=1.))
        class Tokenizer:
            def apply_chat_template(self,*args,**kwargs):return 'prompt'
            def __call__(self,*args,**kwargs):return dict(input_ids=[1,2])
        def oracle(config):
            calls.append(config)
            return dict(configuration=config,objective=5.,status='success',censored=False,feasible=True)
        def setup(*args):
            return dict(task_id='p',reference_sequence=reference),initial,lambda c:self.require_valid(c,reference),oracle
        real_status=evaluate.status
        interrupted=False
        def status(path,**values):
            nonlocal interrupted
            real_status(path,**values)
            if not interrupted and values.get('evaluations')==1:
                interrupted=True;raise InterruptedError('after committed fallback')
        with tempfile.TemporaryDirectory(dir=ROOT/'logs',prefix='fallback_test_') as tmp:
            root=Path(tmp);atomic(root/'recipe.json',recipe)
            with patch.object(evaluate,'ROOT',root),patch.object(evaluate,'verify',lambda:recipe),\
                 patch.object(evaluate,'peptide_setup',setup),\
                 patch('qwen_inference_transport.SharedQwenClient',Client),\
                 patch('transformers.AutoTokenizer.from_pretrained',return_value=Tokenizer()):
                with patch.object(evaluate,'status',status):
                    with self.assertRaises(InterruptedError):evaluate.run('peptide','p',0)
                evaluate.run('peptide','p',0)
                evaluate.run('peptide','p',0)
                self.assertEqual(len(calls),3)
                self.assertEqual(len(generations),30)
                self.assertEqual(len({canonical(c) for c in calls}),3)
                folder=root/'peptide/runs/p';db=evaluate.connect(folder)
                rows=[json.loads(r[0]) for r in db.execute('select value from records order by n')]
                self.assertTrue(all(r['proposal_source']=='random_fallback' and 'prediction' not in r for r in rows))
                done=load(folder/'status.json')
                self.assertEqual(done['random_fallback_evaluations'],3)
                messages=evaluate.prompt(dict(task_id='p',reference_sequence=reference),initial,rows,3,[])
                self.assertFalse(any(m['role']=='assistant' for m in messages))
                self.assertIn('random_fallback',messages[1]['content'])
                recipe['peptide']['proposals']=4
                with db:evaluate.put(db,'pending',dict(status='in_flight',answer={},number=3))
                with self.assertRaisesRegex(RuntimeError,'Unresolved oracle call'):evaluate.run('peptide','p',0)
                self.assertEqual(len(calls),3)
                db.close()


if __name__=='__main__':unittest.main()
