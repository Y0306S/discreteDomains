"""Offline selection integrity and real optimizer-path regression checks."""
import ast
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from campaign_io import InfrastructureError, TaskStore, canonical
from rl_common import digest, read, select_training_members
import train_grpo as trainer


def pool(count, improver=None):
    return [dict(sample_i=i, group_id='offline:k3:r0', evaluation_status='SUCCESS',
        prefix_best_loss=1., loss=.9 if i == improver else 1.1 + .01*i,
        reward=dict(reward=.5 if i == improver else .01*i, delta_best=.1 if i == improver else 0.),
        completion='{}', messages=[dict(role='user', content='same')], policy_sha256='parent',
        task_id='task', prefix_k=3, phase='offline', scenario='normal') for i in range(count)]


class SelectionIntegrity(unittest.TestCase):
    def test_matches_previous_ml_v14_random_replacement(self):
        path=Path('/root/data/cys/llmoptimizer_benchmark/examples/benchmark/domains/collect_taskset_grpo_rollouts.py')
        tree=ast.parse(path.read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_members')
        namespace=dict(N_BASE=4)
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
        for count,improver in ((4,0),(5,4),(9,8),(10,None)):
            rows=pool(count,improver)
            old=[dict(r,excluded_from_training=False,improver=r['loss']<1.) for r in rows]
            expected,_=namespace['train_members'](old,random.Random('offline:k3:r0:1'))
            actual=select_training_members(rows,'offline:k3:r0',1)
            self.assertEqual(actual['train_members'],expected)
            self.assertEqual(len(expected),4)

    def test_first_four_improver_keeps_original_four(self):
        selection=select_training_members(pool(4,1),'group')
        self.assertEqual(selection['train_members'],list(range(4)))
        self.assertIsNone(selection['replaced_sample_i'])

    def test_extra_improver_replaces_one_original_reproducibly(self):
        rows=pool(9,8)
        a=select_training_members(rows,'group',1)
        random.seed(999);random.random()
        self.assertEqual(a,select_training_members(rows,'group',1))
        dropped=random.Random('group:1').choice(list(range(4)))
        self.assertEqual(a['replaced_sample_i'],dropped)
        self.assertEqual(a['injected_sample_i'],8)
        self.assertEqual(a['train_members'],[i for i in range(4) if i!=dropped]+[8])
        # Different group-local seeds can select every original slot.
        self.assertEqual({select_training_members(rows,f'group{i}')['replaced_sample_i']
                          for i in range(100)},set(range(4)))

    def test_no_improvement_by_ten_keeps_only_first_four(self):
        self.assertEqual(select_training_members(pool(10),'group')['train_members'],list(range(4)))

    def test_extra_reward_variance_cannot_create_selected_training_signal(self):
        rows=pool(10)
        for row in rows[:4]:row['reward']['reward']=.1
        self.assertEqual(select_training_members(rows,'group')['reason'],'no_reward_signal')

    def test_failure_anywhere_excludes_pool_and_oversampling_rejected(self):
        rows=pool(9,8);rows[5].update(evaluation_status='TIMEOUT',reward=None)
        self.assertFalse(select_training_members(rows,'group')['trainable'])
        self.assertEqual(select_training_members(pool(7,4),'group')['reason'],'sampling_continued_after_stop')

    def load_fixture(self, tamper=None):
        rows=pool(9,8);selection=select_training_members(rows,rows[0]['group_id'])
        for row in rows:row.update(train_member=row['sample_i'] in selection['train_members'],group_status=selection['reason'])
        job=dict(group_id=rows[0]['group_id'],task_id='task',phase='offline',prefix_k=3,repeat=0,scenario='normal')
        if tamper:tamper(rows,selection)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);directory=root/'rollouts/offline/task/k3/r0';directory.mkdir(parents=True)
            path=directory/'rollouts_task.jsonl';path.write_text(''.join(canonical(row)+'\n' for row in rows))
            store=TaskStore(directory/'group.sqlite',{})
            store.put('complete',dict(rows=len(rows),output_sha256=digest(path),selection=selection,
                                      trainable=selection['trainable'],reason=selection['reason']))
            store.close()
            with patch.object(trainer,'ROOT',root),patch.object(trainer,'read',side_effect=lambda name:
                    [job] if name=='assets/jobs.json' else dict(training=dict(seed=1))):
                return trainer.load_groups('offline','parent'),rows,selection

    def test_training_loader_uses_only_selected_four_and_original_ids(self):
        (groups,excluded),rows,selection=self.load_fixture()
        self.assertFalse(excluded)
        self.assertEqual([r['sample_i'] for r in groups[0]],selection['train_members'])
        self.assertEqual(len(groups[0]),4)
        from grpo_math import advantages
        a=advantages([r['reward']['reward'] for r in groups[0]])
        expected=advantages([rows[i]['reward']['reward'] for i in selection['train_members']])
        self.assertEqual(a.tolist(),expected.tolist())

    def test_training_loader_rejects_tampered_flags_and_selection(self):
        def flags(rows,selection):rows[0]['train_member']=not rows[0]['train_member']
        def membership(rows,selection):selection['train_members']=[0,1,2,3]
        for tamper in (flags,membership):
            with self.assertRaisesRegex(InfrastructureError,'membership'):self.load_fixture(tamper)

    def test_membership_commit_survives_interrupted_export(self):
        from test_revision import AdaptiveRollouts
        rows,generated,evaluated=AdaptiveRollouts().run_group(improvement=8,export_interrupt=True)
        self.assertEqual(generated,list(range(9)))
        self.assertEqual(evaluated,list(range(9)))
        selected=[r['sample_i'] for r in rows if r['train_member']]
        self.assertEqual(len(selected),4)
        self.assertIn(8,selected)
        expected=select_training_members(rows,rows[0]['group_id'])
        self.assertEqual(selected,expected['train_members'])


class OptimizerPath(unittest.TestCase):
    def fixture(self, all_repeat=False):
        import torch
        from types import SimpleNamespace
        from training_metrics import snapshot
        torch.manual_seed(73)
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding=torch.nn.Embedding(16,8)
                self.base_head=torch.nn.Linear(8,16,bias=False)
                self.embedding.requires_grad_(False);self.base_head.requires_grad_(False)
                self.lora_delta=torch.nn.Parameter(torch.zeros(16,8))
            def forward(self,input_ids,logits_to_keep,**kwargs):
                hidden=self.embedding(input_ids)[:,logits_to_keep]
                return SimpleNamespace(logits=self.base_head(hidden)+torch.nn.functional.linear(hidden,self.lora_delta))
        model=Tiny();named=trainer.validate_trainable_parameters(model)
        groups=[[dict(reward=dict(reward=r),evaluation_status='DUPLICATE' if all_repeat or i==0 else 'SUCCESS')
                 for i,r in enumerate([-1.]*4 if all_repeat else [-1.,.1,.2,.9])]]
        batches=[[(torch.tensor([[1,2,3+i]]),2) for i in range(4)]]
        with torch.no_grad():old=[[trainer.token_logprobs(model,ids,n).clone() for ids,n in batches[0]]]
        cache=dict(old=old,reference=copy.deepcopy(old),initial_lora=snapshot(named))
        cfg=read('settings.json')['training']
        optimizer=torch.optim.AdamW([p for _,p in named],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        return torch,model,named,groups,batches,cache,cfg,optimizer

    def test_production_update_changes_only_adapter_and_moves_likelihoods(self):
        torch,model,named,groups,batches,cache,cfg,optimizer=self.fixture()
        frozen={n:p.clone() for n,p in model.named_parameters() if not p.requires_grad}
        metrics=trainer.update_batch(model,optimizer,named,groups,batches,cache,[0],cfg,torch.device('cpu'))
        self.assertGreater(metrics['step_weight_delta']['l2'],0.)
        self.assertGreater(metrics['grad_norm'],0.)
        for n,p in model.named_parameters():
            if n in frozen:torch.testing.assert_close(p,frozen[n],rtol=0,atol=0)
        self.assertLess(trainer.token_logprobs(model,*batches[0][0]).item(),cache['old'][0][0].item())
        self.assertGreater(trainer.token_logprobs(model,*batches[0][3]).item(),cache['old'][0][3].item())

    def test_all_repeat_production_batch_still_updates_adapter(self):
        torch,model,named,groups,batches,cache,cfg,optimizer=self.fixture(True)
        metrics=trainer.update_batch(model,optimizer,named,groups,batches,cache,[0],cfg,torch.device('cpu'))
        self.assertGreater(metrics['duplicate_rejection_loss'],0.)
        after=sum(trainer.token_logprobs(model,*batch).item() for batch in batches[0])
        self.assertLess(after,sum(p.item() for p in cache['old'][0]))

    def test_accumulation_averages_groups_and_clears_old_gradients(self):
        torch,model,named,groups,batches,cache,cfg,optimizer=self.fixture()
        other=copy.deepcopy(model);other_named=trainer.validate_trainable_parameters(other)
        other_opt=torch.optim.AdamW([p for _,p in other_named],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        for _,p in named:p.grad=torch.full_like(p,1000.)
        trainer.update_batch(model,optimizer,named,groups,batches,cache,[0],cfg,torch.device('cpu'))
        trainer.update_batch(other,other_opt,other_named,groups,batches,cache,[0,0],cfg,torch.device('cpu'))
        torch.testing.assert_close(model.lora_delta,other.lora_delta,atol=1e-10,rtol=1e-6)

    def test_optimizer_save_restore_and_phase_handoff_match_uninterrupted(self):
        torch,model,named,groups,batches,cache,cfg,optimizer=self.fixture()
        trainer.update_batch(model,optimizer,named,groups,batches,cache,[0],cfg,torch.device('cpu'))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.pt'
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),parameter_names=[n for n,_ in named]),path)
            saved=torch.load(path,weights_only=True)
        resumed=copy.deepcopy(model);resumed.load_state_dict(saved['model'])
        resumed_named=trainer.validate_trainable_parameters(resumed)
        resumed_opt=torch.optim.AdamW([p for _,p in resumed_named],lr=cfg['learning_rate'])
        trainer.restore_optimizer(resumed_opt,saved,[n for n,_ in resumed_named])
        self.assertEqual(resumed_opt.state_dict()['state'][0]['step'].item(),1.)
        for m,o,n in ((model,optimizer,named),(resumed,resumed_opt,resumed_named)):
            trainer.update_batch(m,o,n,groups,batches,cache,[0],cfg,torch.device('cpu'))
        torch.testing.assert_close(model.lora_delta,resumed.lora_delta,atol=0,rtol=0)
        with self.assertRaisesRegex(InfrastructureError,'parameter order'):
            trainer.restore_optimizer(resumed_opt,saved,['wrong_name'])

    def test_post_update_kl_overshoot_rejected_before_publication(self):
        torch,model,named,groups,batches,cache,cfg,optimizer=self.fixture()
        cfg=dict(cfg,max_reference_kl=0.)
        optimizer.param_groups[0]['lr']=1.
        with self.assertRaisesRegex(InfrastructureError,'after optimizer update'):
            trainer.update_batch(model,optimizer,named,groups,batches,cache,[0],cfg,torch.device('cpu'))

    def test_frozen_cache_single_commit_and_input_identity(self):
        torch,model,named,groups,batches,cache,cfg,optimizer=self.fixture()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'cache.pt';inputs=dict(phase='one',parent='hash')
            trainer.save_frozen_cache(path,inputs,cache)
            loaded=trainer.load_frozen_cache(path,inputs)
            torch.testing.assert_close(loaded['old'][0][0],cache['old'][0][0])
            self.assertEqual(list(Path(d).iterdir()),[path])
            with self.assertRaisesRegex(InfrastructureError,'different rollouts'):
                trainer.load_frozen_cache(path,dict(phase='two'))

    def test_saved_optimizer_corruption_rejected(self):
        torch,*_=self.fixture()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);state=path/'training_state.pt'
            torch.save(dict(optimizer='example'),state)
            (path/'complete.json').write_text(canonical(dict(training_state_sha256=digest(state))))
            self.assertEqual(trainer.load_training_state(path),dict(optimizer='example'))
            state.write_bytes(b'corrupt')
            with self.assertRaisesRegex(InfrastructureError,'corrupt'):trainer.load_training_state(path)

    def test_negative_advantage_lower_clip_and_frozen_probability_gradients(self):
        torch,*_=self.fixture()
        from grpo_math import token_loss
        current=torch.tensor([-2.],requires_grad=True)
        old=torch.tensor([-1.],requires_grad=True);ref=old.detach().clone().requires_grad_()
        token_loss(current,old,ref,-1.,beta=0.).backward()
        self.assertEqual(current.grad.item(),0.)
        self.assertIsNone(old.grad);self.assertIsNone(ref.grad)


if __name__=='__main__':unittest.main(verbosity=2)
