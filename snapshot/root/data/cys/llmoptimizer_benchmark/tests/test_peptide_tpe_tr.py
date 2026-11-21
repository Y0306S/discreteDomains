"""Behavioral TPE-TR checks using real Optuna and synthetic decoder/oracle only."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0, str(TOOLS))
import peptide_tpe_tr_policy as policy
import peptide_tpe_tr as runner

runner.torch.set_num_threads(1)


def records():
    return [dict(z=[0., 0.], score=-1., constraints=[-.25]),
            dict(z=[.1, -.1], score=-2., constraints=[0.]),
            dict(z=[3.9, -3.9], score=-3., constraints=[0.])]


def settings(**kwargs):
    return dict(policy.DEFAULTS, n_ei_candidates=4, **kwargs)


class RegionTest(unittest.TestCase):
    def test_global_bounds_and_feasible_center(self):
        rr=records()+[dict(z=[4.,4.],score=100.,constraints=[.01])]
        s=policy.new_region(rr,settings())
        self.assertEqual(s['center_index'],0)
        low,high=policy.bounds(rr,s,[-4.,4.])
        np.testing.assert_array_equal(low,[-3.2,-3.2])
        np.testing.assert_array_equal(high,[3.2,3.2])
        rr.append(dict(z=[3.9,-3.9],score=-.5,constraints=[0.]))
        policy.observe(s,rr,settings())
        low,high=policy.bounds(rr,s,[-4.,4.])
        np.testing.assert_allclose(high,[4.,-.7],rtol=0,atol=1e-14)
        self.assertGreaterEqual(low.min(),-4.)

    def test_infeasible_best_cannot_move_or_expand_region(self):
        rr=records();cfg=settings(failure_tolerance=2);s=policy.new_region(rr,cfg)
        for _ in range(2):
            rr.append(dict(z=[1.,1.],score=100.,constraints=[.01]))
            policy.observe(s,rr,cfg)
        self.assertEqual(s['best_value'],-1.)
        self.assertEqual(s['center_index'],0)
        self.assertEqual(s['length'],.4)
        self.assertEqual(s['last_action'],'shrink')

    def test_expand_shrink_and_reset_preserve_incumbent(self):
        rr=records();cfg=settings(failure_tolerance=1,success_tolerance=2,length_min=.4)
        s=policy.new_region(rr,cfg)
        for score in [-.5,-.25]:
            rr.append(dict(z=[.1,.1],score=score,constraints=[0.]))
            policy.observe(s,rr,cfg)
        self.assertEqual(s['length'],1.6)
        for _ in range(3):
            rr.append(dict(z=[.1,.1],score=-2.,constraints=[0.]))
            policy.observe(s,rr,cfg)
        self.assertEqual(s['length'],.8)
        self.assertEqual(s['resets'],1)
        self.assertEqual(s['best_value'],-.25)
        self.assertEqual(s['last_action'],'reset_at_incumbent')

    def test_nonfinite_measurement_leaves_region_unchanged(self):
        rr=records();s=policy.new_region(rr,settings());before=copy.deepcopy(s)
        rr.append(dict(z=[0.,0.],score=float('nan'),constraints=[0.]))
        with self.assertRaises(ValueError):policy.observe(s,rr,settings())
        self.assertEqual(s,before)


class SamplerTest(unittest.TestCase):
    def test_tpe_is_local_and_sequential_with_imported_observations(self):
        rr=records();s=policy.new_region(rr,settings())
        engine=policy.SequentialTPE(settings(),[-4.,4.],7)
        p=engine.ask(rr,s)
        self.assertEqual(engine.study.sampler.__class__.__name__,'TPESampler')
        self.assertEqual(p['local_observations'],2) # outside point is not clipped into the box
        self.assertEqual(len(engine.study.trials),3)
        self.assertTrue(np.all(np.abs(p['z'])<=3.2))
        with self.assertRaisesRegex(ValueError,'one recommendation'):engine.ask(rr,s)
        engine.tell(-5.,[.1])
        self.assertEqual(len(engine.study.trials),3)
        self.assertEqual(list(engine.study.trials[-1].system_attrs['constraints']),[.1])
        with self.assertRaisesRegex(ValueError,'No recommendation'):engine.tell(-1.,[0.])

    def test_pending_resume_and_region_rebuild_reproduce_candidates(self):
        cfg=settings(failure_tolerance=2)
        rr=records();s=policy.new_region(rr,cfg)
        first=policy.SequentialTPE(cfg,[-4.,4.],11)
        pending=first.ask(rr,s)
        sampler=pickle.loads(pickle.dumps(first.checkpoint()))
        resumed=policy.SequentialTPE(cfg,[-4.,4.],11,sampler)
        resumed.restore_pending(rr,s,pending)
        for engine in [first,resumed]:engine.tell(-4.,[0.])
        rr.append(dict(z=pending['z'],u=pending['u'],region_revision=s['revision'],score=-4.,constraints=[0.]))
        policy.observe(s,rr,cfg)
        for _ in range(5):
            a=first.ask(rr,s);b=resumed.ask(rr,s)
            self.assertEqual(a,b)
            for engine in [first,resumed]:engine.tell(-4.,[0.])
            rr.append(dict(z=a['z'],u=a['u'],region_revision=s['revision'],score=-4.,constraints=[0.]))
            policy.observe(s,rr,cfg)

    def test_checkpoint_is_importable_in_a_fresh_process(self):
        cfg=settings();rr=records();s=policy.new_region(rr,cfg)
        first=policy.SequentialTPE(cfg,[-4.,4.],11);p=first.ask(rr,s)
        saved=dict(settings=cfg,records=rr,region=s,pending=p,sampler=first.checkpoint())
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'state.pkl';path.write_bytes(pickle.dumps(saved))
            code='''import pickle,sys,json
import peptide_tpe_tr_policy as p
s=pickle.load(open(sys.argv[1],"rb"))
e=p.SequentialTPE(s["settings"],[-4.,4.],11,s["sampler"])
e.restore_pending(s["records"],s["region"],s["pending"])
e.tell(-4.,[0.])
print(json.dumps({"callback_module":e.checkpoint()._constraints_func.__module__,"trials":len(e.study.trials)}))
'''
            result=subprocess.run([sys.executable,'-c',code,str(path)],env=dict(os.environ,PYTHONPATH=str(TOOLS)),
                capture_output=True,text=True,check=True,timeout=60)
            got=json.loads(result.stdout)
            self.assertEqual(got,dict(callback_module='peptide_tpe_tr_policy',trials=3))


class FakeClient:
    def __init__(self,ledger):self.ledger=ledger
    def request(self,request):
        assert request['op']=='oracle'
        assert len(request['sequences'])==1
        key=json.dumps(request,sort_keys=True)
        if key not in self.ledger:
            self.ledger[key]=dict(scores=[-5. for _ in request['sequences']])
        return self.ledger[key]
    def close(self):pass


class FakeObjective:
    def vae_decode(self,z):
        assert tuple(z.shape)==(1,256)
        # All outputs are one mutation from AAAA; repetition exercises the cache.
        k=int(abs(float(z[0,0]))*1000)%5
        return [['AAAT','AATA','ATAA','TAAA','AAAC'][k]]


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.recipe=dict(seed=1,initialization=3,proposals=6,fresh_budget=6,batch_size=1,
            similarity=.75,tpe_bounds=[-4.,4.],test=['pep_fixture'],references={'pep_fixture':'AAAA'},
            trust_region=settings(failure_tolerance=2),budget_definition='fixture')
        self.ledger={}
        self.patchers=[patch.object(runner,'create_objective',side_effect=lambda *a:(FakeObjective(),FakeClient(self.ledger))),
            patch.object(runner.bo,'load_pool',return_value=('AAAA',['AAAA','AAAG','AAGA'],[-1.,-2.,-3.])),
            patch.object(runner.bo,'encode',return_value=runner.torch.zeros(3,256)),
            patch.object(runner.protocol,'create_state',side_effect=AssertionError('No LOLBO/GP state allowed'))]
        for p in self.patchers:p.start()
    def tearDown(self):
        for p in reversed(self.patchers):p.stop()
        self.temp.cleanup()
    def args(self,name,stage=None):
        p=self.root/name;p.mkdir(exist_ok=True)
        (p/'recipe.json').write_text(json.dumps(self.recipe,sort_keys=True))
        return SimpleNamespace(campaign=p,task='pep_fixture',server='unused',stop_after_events=None,stop_after_stage=stage)
    def run_fixture(self,name,stage=None):
        return runner.run_task(self.args(name,stage),self.recipe)
    def archived_records(self,name):
        with tarfile.open(self.root/name/'runs/pep_fixture.tar.gz') as t:
            db=sqlite3.connect(':memory:')
            db.deserialize(t.extractfile('runtime.sqlite').read())
            try:return runner.storage.trials(db)
            finally:db.close()

    def test_scored_pending_commits_without_rescore_and_matches_uninterrupted(self):
        expected=self.run_fixture('uninterrupted')
        self.ledger={}
        partial=self.run_fixture('resumed','score')
        self.assertEqual(partial['proposals'],0)
        self.assertEqual(len(self.ledger),1)
        # Resume only one committed proposal: the saved scored event needs no RPC.
        args=self.args('resumed');args.stop_after_events=1
        with patch.object(FakeClient,'request',side_effect=AssertionError('Unexpected re-score')):
            resumed=runner.run_task(args,self.recipe)
        self.assertEqual(resumed['proposals'],1)
        self.assertEqual(resumed['fresh_evaluations'],1)
        result=self.run_fixture('resumed')
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['proposals'],6)
        self.assertEqual(result['recenter_proposals'],0)
        self.assertEqual(result['trust_region'],expected['trust_region'])
        a=self.archived_records('uninterrupted');b=self.archived_records('resumed')
        self.assertEqual([r['z'] for r in a],[r['z'] for r in b])
        self.assertEqual([r['row']['sequence'] for r in a],[r['row']['sequence'] for r in b])
        self.assertEqual(result['fresh_evaluations']+result['cache_hits'],6)
        self.assertEqual(len(self.ledger),result['fresh_evaluations'])

    def test_unscored_pending_resumes_same_proposal(self):
        self.run_fixture('pending','proposal')
        self.assertEqual(len(self.ledger),0)
        with patch.object(FakeObjective,'vae_decode',side_effect=AssertionError('Pending proposal must not be decoded again')):
            args=self.args('pending');args.stop_after_events=1
            result=runner.run_task(args,self.recipe)
        self.assertEqual(result['proposals'],1)
        self.assertEqual(len(self.ledger),1)

    def test_unknown_oracle_response_reuses_durable_request_receipt(self):
        self.run_fixture('unknown','proposal')
        original=FakeClient.request
        def lose_reply(client,request):
            original(client,request)
            raise ConnectionError('Response lost after server recorded result')
        with patch.object(FakeClient,'request',lose_reply):
            with self.assertRaises(ConnectionError):self.run_fixture('unknown')
        self.assertEqual(len(self.ledger),1)
        args=self.args('unknown');args.stop_after_events=1
        result=runner.run_task(args,self.recipe)
        self.assertEqual(len(self.ledger),1)
        self.assertEqual(result['fresh_evaluations'],1)


if __name__=='__main__':unittest.main()
