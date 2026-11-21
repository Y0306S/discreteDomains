"""Behavior checks with synthetic callbacks; no model services or oracle calls."""
import itertools
import json
from pathlib import Path
import tempfile
import unittest

from method import (Journal, IncompletePool, UnresolvedMeasurement, warm_start, optimize,
                    observed, selector, select_arm, verify_completed_stage, optimization_prompt)

REFERENCE = 'A'*20


def variants():
    for left, right in itertools.combinations(range(20), 2):
        chars=list(REFERENCE);chars[left]='K';chars[right]='R'
        yield ''.join(chars)


class MethodTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)
        self.config=dict(seed=1,warm_start=dict(count=50,maximum_samples=60),
                         optimization=dict(attempts=12,llm_generation_limit=5))
        self.sequences=list(variants())
        self.pool=[observed(s,100.+i,REFERENCE,i,'initialization') for i,s in enumerate(self.sequences[:50])]
        self.journals=[]

    def tearDown(self):
        for j in self.journals:j.close()
        self.tmp.cleanup()

    def journal(self,name='task'):
        j=Journal(self.path/(name+'.sqlite'),dict(task='fixture',stage=name));self.journals.append(j);return j

    def test_warm_acceptance_rejections_order_and_no_rescore(self):
        outputs=[REFERENCE,'AX','BAD_JSON',self.sequences[0],self.sequences[0]]+self.sequences[1:50]
        calls=[]
        def generate(request):
            seq=outputs[request['attempt']]
            return dict(text=seq if seq=='BAD_JSON' else json.dumps(dict(sequence=seq)))
        def evaluate(seq):calls.append(seq);return 100.+len(calls)
        j=self.journal();state=warm_start(j,'fixture',REFERENCE,self.config,generate,evaluate)
        self.assertEqual([r['sequence'] for r in state['rows']],self.sequences[:50])
        self.assertEqual(len(calls),50);self.assertEqual(len(state['rejected']),4)
        warm_start(j,'fixture',REFERENCE,self.config,lambda _:self.fail('regenerated'),lambda _:self.fail('rescored'))
        self.assertEqual(len(calls),50)

    def test_underfilled_pool_fails_without_random_replacement(self):
        self.config['warm_start']['maximum_samples']=4
        with self.assertRaises(IncompletePool):
            warm_start(self.journal(),'fixture',REFERENCE,self.config,
                       lambda _:dict(text=json.dumps(dict(sequence=REFERENCE))),lambda _:self.fail('scored invalid'))

    def test_ambiguous_measurement_does_not_replay(self):
        j=self.journal();counter=[]
        def interrupted():counter.append(1);raise RuntimeError('interruption after call started')
        with self.assertRaises(RuntimeError):j.once('measurement',dict(sequence='K'),interrupted,measurement=True)
        with self.assertRaises(UnresolvedMeasurement):j.once('measurement',dict(sequence='K'),interrupted,measurement=True)
        self.assertEqual(len(counter),1)

    def test_result_journal_replays_without_external_call(self):
        j=self.journal();j.once('measurement',dict(sequence='K'),lambda:42.,measurement=True)
        self.assertEqual(j.once('measurement',dict(sequence='K'),lambda:self.fail('rescored'),measurement=True),42.)
        with self.assertRaises(ValueError):j.once('measurement',dict(sequence='R'),lambda:43.,measurement=True)

    def test_pucb_schedule_and_cooldown(self):
        b=selector()
        self.assertEqual(select_arm(b,[],1)[0],'llm')
        arm,detail=select_arm(b,[dict(selected_arm='llm')]*8,9)
        self.assertFalse(detail['cooldown_active']);self.assertEqual(detail['schedule_denominator'],91)
        arm,detail=select_arm(b,[dict(selected_arm='llm')]*9,10)
        self.assertEqual(arm,'fsbo');self.assertTrue(detail['cooldown_active'])
        _,detail=select_arm(b,[dict(selected_arm='llm')]*20,21)
        self.assertFalse(detail['cooldown_active'])
        self.assertAlmostEqual(b.omega('llm',0,91),.7)
        self.assertAlmostEqual(b.omega('llm',91,91),.3)

    def callbacks(self,duplicate=False):
        generated=[];measured=[];fsbo_calls=[]
        def generate(request):
            generated.append(request)
            seq=self.pool[0]['sequence'] if duplicate else self.sequences[50+request['attempt']]
            return dict(text=json.dumps(dict(sequence=seq)))
        def evaluate(seq):measured.append(seq);return 80.-len(measured)
        def fsbo(rows,reference,seed):
            fsbo_calls.append(seed)
            seq=next(s for s in self.sequences[100:] if s not in {r['sequence'] for r in rows})
            return dict(sequence=seq,diagnostics=dict(fixture=True))
        return generate,evaluate,fsbo,generated,measured,fsbo_calls

    def test_five_duplicate_generations_fall_back_to_fsbo_only(self):
        self.config['optimization']['attempts']=1
        g,e,f,gs,es,fs=self.callbacks(duplicate=True)
        state=optimize(self.journal(),'orpo','fixture',REFERENCE,self.config,self.pool,g,e,f,lambda _:([],None))
        self.assertEqual(len(gs),5);self.assertEqual(len(es),1);self.assertEqual(len(fs),1)
        event=state['events'][0]
        self.assertEqual((event['selected_arm'],event['actual_arm']),('llm','fsbo'))
        self.assertEqual(state['selector'].attempt_counts(),dict(llm=1,fsbo=1))
        self.assertEqual(len(state['rows']),51)

    def test_resume_identical_selection_and_no_committed_rescore(self):
        g,e,f,gs,es,fs=self.callbacks()
        j=self.journal('resumed')
        first=optimize(j,'orpo','fixture',REFERENCE,self.config,self.pool,g,e,f,lambda _:([],None),stop_after=5)
        self.assertEqual(len(first['events']),5)
        final=optimize(j,'orpo','fixture',REFERENCE,self.config,self.pool,g,e,f,lambda _:([],None))
        self.assertEqual(len(es),12)
        g2,e2,f2,*_=self.callbacks()
        whole=optimize(self.journal('whole'),'orpo','fixture',REFERENCE,self.config,self.pool,g2,e2,f2,lambda _:([],None))
        self.assertEqual(final['events'],whole['events'])
        optimize(j,'orpo','fixture',REFERENCE,self.config,self.pool,
                 lambda _:self.fail('generated after done'),lambda _:self.fail('rescored'),
                 lambda *_:self.fail('FSBO after done'),lambda _:([],None))

    def test_fsbo_exhaustion_counts_attempt_without_oracle_or_fake_score(self):
        self.config['optimization']['attempts']=1
        state=optimize(self.journal(),'orpo','fixture',REFERENCE,self.config,self.pool,
            lambda _:dict(text='INVALID'),lambda _:self.fail('oracle called'),
            lambda *_:dict(sequence=None,diagnostics=dict(exhausted=True)),lambda _:([],None))
        self.assertEqual(state['events'][0]['status'],'NO_VALID_PROPOSAL')
        self.assertIsNone(state['events'][0]['mic']);self.assertEqual(len(state['rows']),50)

    def test_stage_identity_cannot_change(self):
        j=self.journal()
        with self.assertRaises(ValueError):Journal(self.path/'task.sqlite',dict(task='other'))

    def test_all_task_stage_gate_checks_identity_and_budget(self):
        tasks=['fixture','second']
        for task in tasks:
            p=self.path/'orpo/runs'/task/'complete.json';p.parent.mkdir(parents=True)
            p.write_text(json.dumps(dict(task=task,stage='orpo',recipe_sha256='abc',attempts=80)))
        self.assertTrue(verify_completed_stage(self.path,'orpo','abc',tasks,80))
        with self.assertRaises(RuntimeError):verify_completed_stage(self.path,'orpo','wrong',tasks,80)
        with self.assertRaises(RuntimeError):verify_completed_stage(self.path,'orpo','abc',tasks,81)
        (self.path/'orpo/runs/second/complete.json').unlink()
        with self.assertRaises(RuntimeError):verify_completed_stage(self.path,'orpo','abc',tasks,80)

    def test_normal_late_rl_prompt_and_measured_history(self):
        messages=optimization_prompt('orpo_rl','fixture',REFERENCE,self.pool,20,[],
                                     lambda _:self.fail('retrieval in later normal phase'))
        context=json.loads(messages[1]['content'])
        self.assertEqual(context['scenario'],'normal');self.assertEqual(len(context['recent50']),50)
        self.assertNotIn('similar_task_evidence',context)
        self.assertEqual(context['current_best_mic'],100.)


if __name__=='__main__':unittest.main(verbosity=2)
