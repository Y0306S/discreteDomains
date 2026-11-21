"""Behavioral checks for the frozen local/recenter adapters, no APEX calls."""
import copy
import importlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import peptide_local_bo as runner
from peptide_local_bo import policy, bo, torch, storage

torch.set_num_threads(1)
CPU = min(os.sched_getaffinity(0))
UPSTREAM = '/root/data/logs/peptide_bo_train800_20261027_v5b/upstream_cpu'
sys.path.insert(0,UPSTREAM)
from lolbo.utils.bo_utils.turbo import TurboState,update_state


def recipe(method='tpe'):
    return dict(method=method,local_variant=method+'-local-recenter',local=dict(policy.DEFAULTS),
        batch_size=1,proposals=5,fresh_budget=5,n_candidates=64,initialization=3,
        similarity=.75,seed=7,gp_initial_epochs=1,gp_update_epochs=1,learning_rate=.001,
        references={'pep_fixture':'AAAA'},allowed_tasks=['pep_fixture'],cpu_upstream=UPSTREAM)


def state(method='tpe'):
    r = recipe(method)
    rows = [dict(sequence=s,score=y,mic=-y,feasible=c<=0,cached=True,
                 phase='initialization',round=0,elapsed=0.)
            for s,y,c in [('AAAA',-1.,-.25),('AAAG',-2.,0.),('AAGA',-3.,0.)]]
    z = np.random.RandomState(7).normal(0,.1,(3,256)).astype(np.float32)
    cs = np.array([[-.25],[0.],[0.]],dtype=np.float32)
    s = dict(rows=rows,features=z.copy(),constraint_z=z.copy(),constraints=cs,
        trust_region=policy.new_region(rows,cs,1,r['local'],TurboState),
        progress=dict(rounds=0,proposals=0,fresh_evaluations=0,cache_hits=0,
            acquisition_fresh=0,recenter_fresh=0,recenter_proposals=0,events=0,seconds=0.),
        stagnation=0,recenter_plan=None,tpe_trials=[],sampler=None,
        last_event_size=3,last_phase=None,constraint_weights=None,objective_weights=None)
    s['tpe_trials']=[runner.tpe_record(x,row['score'],c[0],(np.full(256,-4.),np.full(256,4.)))
                      for x,row,c in zip(z,rows,cs)]
    return s


def event(s,phase='acquisition',score=-.5,sequence='AATA',fresh=1):
    return dict(event=s['progress']['events']+1,phase=phase,
        z=np.full((1,256),3.,dtype=np.float32),canonical_z=np.full((1,256),.25,dtype=np.float32),
        sequences=[sequence],constraints=[[-.1]],next_cursor=1,
        scored=dict(scores=[score],fresh_evaluations=fresh,cache_hits=[not bool(fresh)]))


class PolicyTest(unittest.TestCase):
    def test_actual_constraint_coordinates_and_canonical_transfer_are_distinct(self):
        s=state('fsbo'); p=event(s)
        policy.commit(s,p,recipe('fsbo'),update_state,TurboState)
        np.testing.assert_array_equal(s['constraint_z'][-1],p['z'][0])
        np.testing.assert_array_equal(s['features'][-1],p['canonical_z'][0])
        self.assertEqual(s['progress']['fresh_evaluations'],1)

    def test_latest_batch_is_used_for_restart(self):
        s=state(); s['trust_region'].restart_triggered=True
        policy.commit(s,event(s),recipe(),update_state,TurboState)
        self.assertEqual(s['trust_region'].best_value,-.5)
        self.assertAlmostEqual(float(s['trust_region'].best_constraint_values[0]),-.1,places=6)

    def test_recenter_is_finite_and_does_not_update_acquisition_counters_or_region(self):
        s=state(); s['stagnation']=10
        self.assertEqual(policy.next_phase(s,recipe()),'recenter')
        plan=copy.deepcopy(s['recenter_plan']); before=pickle.dumps(s['trust_region'])
        self.assertEqual(len(plan['inputs']),2*(1+3))
        p=event(s,phase='recenter',fresh=0)
        policy.commit(s,p,recipe(),update_state,TurboState)
        self.assertEqual(pickle.dumps(s['trust_region']),before)
        self.assertEqual(s['progress']['proposals'],0)
        self.assertEqual(s['progress']['rounds'],0)
        self.assertEqual(s['progress']['recenter_proposals'],1)
        self.assertEqual(s['progress']['cache_hits'],1)
        s['recenter_plan']['cursor']=len(plan['inputs'])
        self.assertEqual(policy.next_phase(s,recipe()),'acquisition')

    def test_recenter_top_archive_excludes_infeasible_and_deduplicates(self):
        s=state(); s['rows'] += [dict(s['rows'][0],score=-.1),
                                 dict(s['rows'][0],sequence='GGGG',score=100.,feasible=False)]
        plan=policy.recenter_plan(s['rows'],1,dict(policy.DEFAULTS,recenter_epochs=1))
        self.assertEqual(plan['inputs'][0],'GGGG') # recent data retained for feedback
        self.assertEqual(plan['inputs'][1:],['AAAA','AAAG','AAGA'])

    def test_native_pool_is_sparse_and_bounded(self):
        torch.manual_seed(17); s=state(); center=torch.as_tensor(s['constraint_z'][0])
        pool=policy.candidate_pool(center,s['trust_region'],1024)
        changed=(pool!=center).sum(1).float()
        self.assertTrue(torch.all(changed>=1))
        self.assertLess(abs(float(changed.mean())-20.),1.)
        self.assertLessEqual(float((pool-center).abs().max()),3.2+1e-6)

    def test_tpe_records_the_actual_masked_proposal_and_restores_sampler(self):
        s=state(); torch.manual_seed(17)
        z,extra=runner.propose_tpe(s,recipe())
        np.testing.assert_allclose(z[0],[extra['tpe_pending'][0]['params'][f'z{i:03d}'] for i in range(256)],rtol=1e-6)
        changed=np.flatnonzero(z[0]!=s['constraint_z'][0])
        self.assertGreater(len(changed),0); self.assertLess(len(changed),60)
        p=event(s);p.update(extra,z=z)
        runner.tell_tpe(s,p,recipe())
        self.assertEqual(s['sampler']._constraints_func.__module__,'peptide_local_policy')
        clone=pickle.loads(pickle.dumps(s))
        rng=bo.capture_rng(); a=runner.propose_tpe(s,recipe())
        bo.restore_rng(rng); b=runner.propose_tpe(clone,recipe())
        np.testing.assert_array_equal(a[0],b[0])
        self.assertEqual(a[1]['tpe_pending'],b[1]['tpe_pending'])

    def test_tpe_batch50_records_every_proposal_and_obeys_remaining_budget(self):
        s=state();r=recipe()
        r.update(batch_size=50,proposals=100,fresh_budget=100)
        z,extra=runner.propose_tpe(s,r)
        self.assertEqual(z.shape,(50,256))
        p=event(s)
        p.update(extra,z=z,sequences=['AATA']*50,
                 constraints=np.full((50,1),-.1),
                 scored=dict(scores=[-.5]*50,fresh_evaluations=1,cache_hits=[False]+[True]*49))
        runner.tell_tpe(s,p,r)
        self.assertEqual(len(s['tpe_trials']),53)
        for i,record in enumerate(s['tpe_trials'][3:]):
            np.testing.assert_allclose(z[i],[record['params'][f'z{j:03d}'] for j in range(256)],rtol=1e-6)
            self.assertEqual(record['constraints'],[float(p['constraints'][i,0])])
        s['progress'].update(proposals=98,fresh_evaluations=99)
        limited,_=runner.propose_tpe(s,r)
        self.assertEqual(limited.shape,(1,256))

    def test_bounded_prefix_counts_unique_fresh_across_recenter_and_acquisition(self):
        got=runner.protocol.bounded_prefix(['AAAA','AAAG','AAAG','AAGA','GGGG'],{'AAAA':1.},2)
        self.assertEqual(got,['AAAA','AAAG','AAAG','AAGA'])

    def test_duplicate_commit_is_rejected(self):
        s=state();p=event(s)
        policy.commit(s,p,recipe(),update_state,TurboState)
        with self.assertRaisesRegex(ValueError,'out of order'):
            policy.commit(s,p,recipe(),update_state,TurboState)

    def test_real_constraint_and_gp_backends_generate_finite_batch(self):
        s=state('gp'); r=recipe('gp');r['batch_size']=2
        c,cm=runner.make_gp(s['constraint_z'])
        c=runner.fit_gp(c,cm,s,r,s['constraints'])
        g,gm=runner.make_gp(s['constraint_z'])
        g=runner.fit_gp(g,gm,s,r,np.array([row['score'] for row in s['rows']]))
        z,_=runner.acquisition(s,r,c,g,None)
        self.assertEqual(z.shape,(2,256));self.assertTrue(np.isfinite(z).all())

    def test_smac_keeps_rf_ei_with_sampled_constraint_mask(self):
        s=state('smac');r=recipe('smac')
        c,_=runner.make_gp(s['constraint_z'])
        z,info=runner.acquisition(s,r,c,None,None)
        self.assertEqual(z.shape,(1,256));self.assertTrue(np.isfinite(z).all())
        self.assertEqual(info['acquisition'],'smac-native-acquisition-local-pool')

    def test_hebo_keeps_mace_with_sampled_constraint_mask(self):
        s=state('hebo');r=recipe('hebo')
        # HEBO's stochastic GP optimizer is intended for a measured initial pool.
        # Three observations in 256 dimensions can make its native SGLD unstable.
        rng=np.random.RandomState(7)
        s['constraint_z']=rng.normal(0,.1,(100,256)).astype(np.float32)
        s['rows']=[dict(s['rows'][i%3],score=-float(i+1),sequence='AAAA') for i in range(100)]
        c,_=runner.make_gp(s['constraint_z'])
        z,info=runner.acquisition(s,r,c,None,None)
        self.assertEqual(z.shape,(1,256));self.assertTrue(np.isfinite(z).all())
        self.assertEqual(info['acquisition'],'hebo-native-acquisition-local-pool')

    def test_transfer_backends_receive_canonical_training_and_raw_queries(self):
        class Client:
            def request(self,op,**values):
                if op=='fit':
                    self.training=values['x'];return {'method':'fixture'}
                self.queries=values['candidates']
                return dict(draws=np.random.RandomState(values['seed']).normal(
                    size=(values['count'],len(values['candidates']))),info={})
        for method in ('fsbo','rgpe'):
            s=state(method);r=recipe(method)
            r.update(lower=[-4.]*256,upper=[4.]*256)
            s['features'][:]=.7
            client=Client();c,_=runner.make_gp(s['constraint_z'])
            z,_=runner.acquisition(s,r,c,None,client)
            np.testing.assert_allclose(client.training,(s['features']+4.)/8.)
            self.assertEqual(z.shape,(1,256));self.assertTrue(np.isfinite(z).all())


class FixtureObjective:
    """Stochastic deterministic-seed fixture; no peptide inference or scoring."""
    def __init__(self):
        self.vae=torch.nn.Linear(1,1)
        self.original=copy.deepcopy(self.vae.state_dict())
        self.num_calls=0;self.xs_to_scores_dict={}
    def vae_forward(self,sequences):
        assert not torch.is_grad_enabled()
        assert all(not p.requires_grad for p in self.vae.parameters())
        return torch.randn(len(sequences),256)*.2,torch.tensor(0.)
    def vae_decode(self,z):
        # Generate feasible, sometimes duplicate outcomes from each seeded latent.
        alphabet='ADEFGHIKLMNPQRSTVWY'
        indices=((z[:,0].abs()*100).long()%len(alphabet)).tolist()
        return ['AAA'+alphabet[i] for i in indices]
    def compute_constraints(self,sequences):
        return torch.tensor([[.75-bo.similarity(s,'AAAA')] for s in sequences])


class FixtureClient:
    def __init__(self,objective,forbid=False):
        self.objective=objective;self.forbid=forbid;self.scored=[]
    def request(self,value):
        if self.forbid:
            raise AssertionError('Durably scored batch evaluated twice')
        self.scored+=value['sequences']
        return dict(scores=[-float(2+ord(s[-1])%5) for s in value['sequences']])
    def close(self):
        for name,p in self.objective.vae.state_dict().items():
            assert torch.equal(p,self.objective.original[name])
    def inference(self,op,**values):
        assert op=='encode_mean'
        return dict(z=torch.stack([torch.full((256,),ord(s[-1])*.001) for s in values['sequences']]))


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def invoke(self,method,path,stage=None,events=0,forbid=False):
        r=recipe(method); r['local']['recenter_every']=1
        r['local']['top_k']=2
        path.mkdir(parents=True,exist_ok=True)
        (path/'recipe.json').write_text(json.dumps(r))
        obj=FixtureObjective(); client=FixtureClient(obj,forbid)
        args=SimpleNamespace(campaign=path,task='pep_fixture',cpu=CPU,server='fixture',
                             stop_after_stage=stage,stop_after_events=events)
        with patch.object(runner,'verify',return_value=r), \
             patch.object(runner.transfer,'create_objective',return_value=(obj,client)), \
             patch.object(bo,'load_pool',return_value=('AAAA',['AAAA','AAAG','AAGA'],[-1.,-2.,-3.])), \
             patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            result=runner.run(args)
        return result,client

    def read(self,path):
        with storage.connect(path/'runs/pep_fixture/runtime.sqlite') as db:
            return storage.get(db,'state'),storage.get(db,'pending')

    def test_scored_pending_resume_without_scoring_and_finite_recenter(self):
        path=self.root/'original'
        self.invoke('tpe',path,stage='score')
        initial,pending=self.read(path)
        self.assertIn('scored',pending)
        clone=self.root/'clone';shutil.copytree(path,clone)
        self.invoke('tpe',path,events=1,forbid=True)
        self.invoke('tpe',clone,events=1,forbid=True)
        left,_=self.invoke('tpe',path)
        right,_=self.invoke('tpe',clone)
        signature=lambda h:[(r['sequence'],r['score'],r['phase'],r.get('proposed_z')) for r in h['observations']]
        self.assertEqual(signature(left),signature(right))
        self.assertGreater(left['recenter_proposals'],0)
        self.assertLessEqual(left['fresh_evaluations'],5)
        self.assertEqual(left['e2e_updates'],0)
        self.assertTrue((path/'runs/pep_fixture.tar.gz').exists())
        self.assertFalse((path/'runs/pep_fixture').exists())

    def test_proposal_pending_resume_is_deterministic(self):
        path=self.root/'original'
        self.invoke('tpe',path,stage='proposal')
        clone=self.root/'clone';shutil.copytree(path,clone)
        a,_=self.invoke('tpe',path);b,_=self.invoke('tpe',clone)
        self.assertEqual([(r['sequence'],r['score']) for r in a['observations']],
                         [(r['sequence'],r['score']) for r in b['observations']])

    def test_fresh_process_replays_scored_recenter_without_rescoring(self):
        path=self.root/'fresh_process'
        self.invoke('tpe',path,events=1)
        self.invoke('tpe',path,stage='score')
        _,pending=self.read(path)
        self.assertEqual(pending['phase'],'recenter')
        code='''import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import test_peptide_local_bo as f
f.RunnerTest().invoke('tpe',Path(sys.argv[2]),events=2,forbid=True)
'''
        result=subprocess.run([sys.executable,'-c',code,str(Path(__file__).parent),str(path)],
            cwd='/tmp',env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue((path/'runs/pep_fixture.tar.gz').exists())

    def test_gp_loop_restores_constraint_weights_and_preserves_frozen_vae(self):
        path=self.root/'gp'
        self.invoke('gp',path,stage='score')
        clone=self.root/'gp_clone';shutil.copytree(path,clone)
        self.invoke('gp',path,events=1,forbid=True)
        self.invoke('gp',clone,events=1,forbid=True)
        left,_=self.invoke('gp',path);right,_=self.invoke('gp',clone)
        self.assertEqual([(r['sequence'],r['score']) for r in left['observations']],
                         [(r['sequence'],r['score']) for r in right['observations']])


if __name__=='__main__':
    unittest.main()
