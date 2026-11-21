import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from campaign_io import TaskStore
from peptide_prompt import valid_first_occurrences
from test_pairs import REF,observation,variants
import peptide_context as peptide
import collect_rollouts as collector
from rl_reward_utils import compute_trial_reward,RewardWeights

class PeptideTrainingTests(unittest.TestCase):
    def setUp(self):
        self.xs=variants(70)
        _,self.valid=valid_first_occurrences([observation(s,100+i) for i,s in enumerate(self.xs[:60])],REF)

    def test_chat_renderer_is_identical_to_ML(self):
        import ast
        from typing import Iterable,Dict
        from mindware.domains.dlhpo.taskset_llm.spec import render_nothink
        path=Path('/root/data/cys/llmoptimizer_benchmark/mindware/domains/dlhpo/taskset_llm/spec.py')
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='render_nothink')
        namespace=dict(Iterable=Iterable,Dict=Dict)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
        for messages in [[dict(role='system',content='rules'),dict(role='user',content='peptide')],
                         [dict(role='user',content='one'),dict(role='assistant',content='two'),dict(role='user',content='three')]]:
            self.assertEqual(render_nothink(messages),namespace['render_nothink'](messages))

    def test_exact_ml_reward_and_invalid_duplicate_no_oracle(self):
        calls=[]
        def evaluate(s):calls.append(s);return dict(objective=90.,status='success')
        for text,status in [('{"sequence":"WWWWWWWW"}','PARSE_FAILURE'),(json.dumps({'sequence':self.xs[0]}),'DUPLICATE')]:
            record=peptide.sample(text,REF,self.valid,self.valid,[],evaluate)
            self.assertEqual(record['evaluation_status'],status)
        self.assertEqual(calls,[])
        record=peptide.sample(json.dumps({'sequence':self.xs[60]}),REF,self.valid,self.valid,[],evaluate)
        expected=compute_trial_reward(perf=-90.,prior_perfs=[-r['mic'] for r in self.valid],parse_ok=True,duplicate=False,eval_ok=True,weights=RewardWeights()).to_dict()
        self.assertEqual(record['reward'],expected)
        sibling=peptide.sample(json.dumps({'sequence':self.xs[60]}),REF,self.valid,self.valid,[record],evaluate)
        self.assertEqual(sibling['evaluation_status'],'DUPLICATE');self.assertEqual(len(calls),1)

    def test_oracle_journal_resume_no_implicit_replay(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'s.sqlite';store=TaskStore(path,{})
            with patch.object(peptide,'oracle',return_value=lambda _:30.):
                result=peptide.measured(self.xs[60],store)
            with patch.object(peptide,'oracle',side_effect=AssertionError('repeat oracle')):
                self.assertEqual(peptide.measured(self.xs[60],store),result)
            store.close()
            store=TaskStore(Path(d)/'unknown.sqlite',{});store.put('evaluation_started',{})
            with self.assertRaisesRegex(RuntimeError,'Unresolved APEX'):peptide.measured(self.xs[60],store)
            store.close()

    def test_rollout_selected_four_export_and_interrupted_resume(self):
        job=dict(task_id='synthetic',phase='phase',prefix_k=3,prefix_end_exclusive=60,scenario='normal',group_id='phase/synthetic/k3')
        messages=[dict(role='system',content='fixed'),dict(role='user',content='fixed')]
        h=dict(reference=REF);generated=[];evaluated=[]
        class Client:
            def generate(client,*args,**kwargs):
                i=len(generated);generated.append(i)
                return [dict(completion=json.dumps({'sequence':self.xs[60+i]}),prompt_ids=[1,2],answer_ids=[3])]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);parent=root/'parent';parent.mkdir();(parent/'adapter_model.safetensors').write_bytes(b'weights')
            def evaluate(seq,store):
                result=store.get('evaluation_result')
                if result:return result
                i=self.xs.index(seq)-60;evaluated.append(i)
                result=dict(status='success',objective=90. if i==5 else 110.+i);store.put('evaluation_result',result);return result
            with patch.object(collector,'ROOT',root),patch.object(peptide,'prefix',return_value=(h,self.valid,self.valid)),patch.object(peptide,'prompt',return_value=(messages,{},self.valid,self.valid)),patch.object(peptide,'measured',side_effect=evaluate):
                collector.collect(job,Client(),parent,'identity')
                collector.collect(job,Client(),parent,'identity')
            self.assertEqual(generated,list(range(6)));self.assertEqual(evaluated,list(range(6)))
            target=root/'rollouts/phase/synthetic/k3/r0/rollouts_synthetic.jsonl'
            rows=[json.loads(x) for x in target.read_text().splitlines()]
            selected=[r['sample_i'] for r in rows if r['train_member']]
            self.assertEqual(len(selected),4);self.assertIn(5,selected)
            self.assertTrue(all(r['messages']==messages for r in rows))

    def test_actual_job_prefixes_never_heldout_or_relabelled(self):
        from rl_common import read
        tasks={t['task_id'] for t in read('assets/tasks.json')}
        source=json.loads(Path('/root/data/logs/peptide_bo_train800_20261027_v5b/recipe.json').read_text())
        self.assertEqual(tasks,set(source['train']));self.assertFalse(tasks&set(source['test']))
        for job in read('assets/jobs.json'):
            self.assertIn(job['task_id'],tasks)
            if job['scenario']=='exploit':self.assertGreater(job['prefix_k'],50)

if __name__=='__main__':unittest.main()
