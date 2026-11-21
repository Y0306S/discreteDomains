import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import evaluate
from batch import parse_batch, accept_candidates, remaining_in_round, round_state
from common import ROOT, load
from protocol import atomic, canonical, peptide_measure
from random_fallback import peptide_proposal
import random

class BatchChecks(unittest.TestCase):
    def test_partial_round_and_budget(self):
        self.assertEqual(remaining_in_round(79,5000),21)
        self.assertEqual(remaining_in_round(100,5000),50)
        self.assertEqual(remaining_in_round(4999,5000),1)
        self.assertEqual(round_state(79)['completed_rounds'],1)
        self.assertEqual(round_state(79)['round_offset'],29)

    def test_invalid_and_duplicate_items_filtered(self):
        def validate(c):
            if not isinstance(c,dict) or set(c)!={'sequence'} or c['sequence']=='BAD':raise ValueError('invalid')
        configs=parse_batch('{"configurations":[{"sequence":"seen"},{"sequence":"new"},{"sequence":"new"},{"sequence":"BAD"},3,{"sequence":"next"}]}')
        accepted,errors=accept_candidates(configs,validate,{canonical({'sequence':'seen'})},50)
        self.assertEqual(accepted,[dict(sequence='new'),dict(sequence='next')]);self.assertEqual(len(errors),4)
        for text in ['{"configurations":[', '[]', '{"next_configuration":{}}']:
            with self.assertRaises(ValueError):parse_batch(text)

    def test_numbered_json_preserves_values_and_rejects_repeated_keys(self):
        self.assertEqual(parse_batch('{"2":"ACDK","1":"ACDE"}'),
            [dict(sequence='ACDE'),dict(sequence='ACDK')])
        self.assertEqual(parse_batch('{"1":[0,1,2,1,3]}'),[dict(plan=[0,1,2,1,3])])
        with self.assertRaises(ValueError):parse_batch('{"1":"ACDE","1":"ACDK"}')
        with self.assertRaises(ValueError):parse_batch('{"2":"ACDE"}')

    def exercise(self, invalid=False, crash_in_flight=False, cpu_batch=False):
        recipe=copy.deepcopy(load(ROOT/'recipe.json'))
        recipe.update(runtime_policy_sha256='runtime',batch_policy_sha256='batch',
            batch_policy=dict(size=50,max_requests=2,max_input_tokens=8192,max_new_tokens=dict(peptide=4096)),
            fallback=dict(enabled=True,after_candidate_pairs=5,max_random_attempts=10000,sha256='fallback'))
        recipe['peptide'].update(test=['p'],proposals=150)
        if cpu_batch:recipe.update(peptide_oracle_policy=dict(enabled=True),peptide_oracle_policy_sha256='cpu_batch')
        reference='AKLR'*5
        initial=[dict(configuration=dict(sequence=reference),objective=10.,status='success',censored=False,feasible=True)]
        configs=[];seen={reference};rng=random.Random(123)
        while len(configs)<220:
            c=peptide_proposal(reference,rng)
            if c['sequence'] not in seen and peptide_measure(c['sequence'],reference)['feasible']:
                configs.append(c);seen.add(c['sequence'])
        generations=[];calls=[]
        class Client:
            def __init__(self,*args):self.state={};self.last_response=None
            def import_state(self,v):self.state=v or {}
            def ensure_loaded(self):pass
            def complete_messages(self,messages,**kwargs):
                content=json.loads(messages[1]['content']);n=content['requested_configurations'];generations.append(n)
                if invalid:return '{truncated'
                return canonical(dict(configurations=configs[79:79+n] if len(generations)==1 else configs[100:100+n]))
        class Tokenizer:
            def apply_chat_template(self,*args,**kwargs):return 'prompt'
            def __call__(self,*args,**kwargs):return dict(input_ids=[1,2])
        def validate(c):
            if not peptide_measure(c['sequence'],reference)['feasible']:raise ValueError('invalid')
        def oracle(c):
            calls.append(c)
            if crash_in_flight:raise InterruptedError('oracle interrupted')
            return dict(configuration=c,objective=5.,status='success',censored=False,feasible=True)
        batch_calls=[]
        def evaluate_many(configs):
            batch_calls.append(len(configs))
            return [oracle(c) for c in configs]
        oracle.batch=evaluate_many
        real_status=evaluate.status;interrupted=False
        def status(path,**values):
            nonlocal interrupted
            real_status(path,**values)
            if not interrupted and values.get('evaluations')==(100 if cpu_batch else 80):
                interrupted=True;raise InterruptedError('after commit')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);atomic(root/'recipe.json',recipe)
            folder=root/'peptide/runs/p';db=evaluate.connect(folder)
            historical=[dict(configuration=c,objective=6.,status='success',censored=False,feasible=True,evaluation=i+1) for i,c in enumerate(configs[:79])]
            with db:
                for i,r in enumerate(historical):db.execute('insert into records values (?,?)',(i,canonical(r)))
            before=db.execute('select * from records order by n').fetchall();db.close()
            with patch.object(evaluate,'ROOT',root),patch.object(evaluate,'verify',lambda:recipe),\
                 patch.object(evaluate,'peptide_setup',lambda *args:(dict(task_id='p',reference_sequence=reference),initial,validate,oracle)),\
                 patch('qwen_inference_transport.SharedQwenClient',Client),\
                 patch('transformers.AutoTokenizer.from_pretrained',return_value=Tokenizer()):
                with patch.object(evaluate,'status',status):
                    with self.assertRaises(InterruptedError):evaluate.run('peptide','p',0)
                if crash_in_flight:
                    with self.assertRaisesRegex(RuntimeError,'Unresolved oracle call'):evaluate.run('peptide','p',0)
                    self.assertEqual(len(calls),1);return
                evaluate.run('peptide','p',0);evaluate.run('peptide','p',0)
                self.assertEqual(len(calls),71);self.assertEqual(len({canonical(c) for c in calls}),71)
                db=evaluate.connect(folder)
                self.assertEqual(db.execute('select * from records where n<79 order by n').fetchall(),before)
                rows=[json.loads(v) for _,v in db.execute('select * from records where n>=79 order by n')]
                self.assertTrue(all(r['batch_policy_sha256']=='batch' for r in rows))
                self.assertTrue(all('prediction' not in r for r in rows))
                self.assertEqual(load(folder/'status.json')['completed_rounds'],3)
                if cpu_batch:
                    self.assertEqual(batch_calls,[21,50])
                    self.assertTrue(all(r['oracle_execution']=='cpu_batch' for r in rows))
                if invalid:
                    self.assertEqual(generations,[21,21,50,50]);self.assertTrue(all(r['proposal_source']=='random_fallback' for r in rows))
                else:
                    self.assertEqual(generations,[21,50]);self.assertTrue(all(r['proposal_source']=='llm' for r in rows))
                db.close()

    def test_partial_round_persisted_queue_resume(self):self.exercise()
    def test_cpu_batch_commit_resume(self):self.exercise(cpu_batch=True)
    def test_truncated_json_fallback_resume(self):self.exercise(invalid=True)
    def test_inflight_requires_audit(self):self.exercise(crash_in_flight=True)

if __name__=='__main__':unittest.main()
