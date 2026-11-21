"""Behavioral tests of rollback and probe isolation; CPU, no oracle calls."""
import copy
import sys
import unittest
from pathlib import Path

import torch
import common
from target_gate import (NumericalVAEError, checked_training_step, guarded_retrain,
                         initial_anchors, target_decision, target_probe)
from uniref_vae.data import DatasetKmers
import pandas as pd


class TinyVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
        self.register_buffer('version', torch.tensor(0))

    def forward(self, tokens):
        loss = self.weight.square() + torch.rand(())*.0001
        return dict(loss=loss, recon_loss=loss, kldiv=loss*0)


def source_ok(model, data):
    return dict(feasible=128, reconstruction_loss=1.)


class GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vocab = pd.read_csv(common.UPSTREAM/'uniref_vae/1mer_vocab.csv', header=None).values.squeeze().tolist()
        cls.data = DatasetKmers(k=1, load_data=False, vocab=vocab)

    def setUp(self):
        self.model = TinyVAE()
        self.state = dict(rows=[dict(sequence='ACDE', phase='initialization'),
                                dict(sequence='ACDF', phase='initialization'),
                                dict(sequence='WWWW', phase='acquisition')])
        self.initial = copy.deepcopy(self.model.state_dict())

    def update(self, value, source=source_ok):
        def train(records):
            with torch.no_grad():
                self.model.weight.fill_(value)
                self.model.version.add_(1)
            records.append(dict(loss=value*value))
        return guarded_retrain(self.model, self.data, self.state, train, source,
                               source_ok(None,None))

    def assert_restored(self):
        self.assertTrue(all(torch.equal(v,self.initial[k]) for k,v in self.model.state_dict().items()))

    def test_worse_loss_rolls_back_parameters_and_buffers(self):
        event=self.update(2.)
        self.assertFalse(event['accepted']); self.assert_restored()
        self.assertEqual(event['vae_change_l2'],0.)

    def test_improvement_is_retained(self):
        event=self.update(.5)
        self.assertTrue(event['accepted']); self.assertEqual(float(self.model.weight),.5)

    def test_equal_loss_is_retained(self):
        self.assertTrue(self.update(1.)['accepted'])

    def test_improving_target_cannot_bypass_source_gate(self):
        def source(model, data):
            return dict(feasible=128, reconstruction_loss=1. if float(model.weight)==1. else 2.)
        event=self.update(.5,source)
        self.assertIn('source_retention_failed',event['rejection_reasons']); self.assert_restored()

    def test_fixed_baseline_prevents_cumulative_drift(self):
        with torch.no_grad(): self.model.weight.fill_(.5)
        self.state['target_retention_initial_baseline']=target_probe(self.model,self.data,initial_anchors(self.state))
        self.model.load_state_dict(self.initial)
        event=self.update(.8)
        self.assertIn('target_loss_worse_than_initial_baseline',event['rejection_reasons']); self.assert_restored()

    def test_nonfinite_candidate_rolls_back(self):
        event=self.update(float('nan'))
        self.assertFalse(event['accepted']); self.assert_restored()

    def test_numerical_exception_after_update_rolls_back(self):
        def train(records):
            with torch.no_grad(): self.model.weight.fill_(2.)
            raise NumericalVAEError('nonfinite gradient')
        event=guarded_retrain(self.model,self.data,self.state,train,source_ok,source_ok(None,None))
        self.assertTrue(event['rolled_back']); self.assert_restored()

    def test_infrastructure_error_propagates_after_restoration(self):
        def train(records):
            with torch.no_grad(): self.model.weight.fill_(2.)
            raise OSError('short read')
        with self.assertRaises(OSError):
            guarded_retrain(self.model,self.data,self.state,train,source_ok,source_ok(None,None))
        self.assert_restored()

    def test_probe_is_reproducible_and_preserves_rng_and_mode(self):
        self.model.train(); torch.manual_seed(193)
        rng=torch.get_rng_state().clone()
        first=target_probe(self.model,self.data,initial_anchors(self.state))
        self.assertTrue(torch.equal(torch.get_rng_state(),rng)); self.assertTrue(self.model.training)
        second=target_probe(self.model,self.data,initial_anchors(self.state))
        self.assertEqual(first,second); self.assertEqual(first['sequences'],2)
        self.assertEqual(initial_anchors(self.state),['ACDE','ACDF'])

    def test_changed_probe_is_not_compared(self):
        before=target_probe(self.model,self.data,['ACDE'])
        after=target_probe(self.model,self.data,['ACDF'])
        with self.assertRaises(ValueError): target_decision(before,after,before)

    def test_checked_step_runs_optimizer_and_reports_gradient(self):
        optimizer=torch.optim.Adam(self.model.parameters(),lr=.1)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        record=checked_training_step(self.model,torch.zeros(2,2),optimizer,scaler)
        self.assertTrue(record['optimizer_step_executed'])
        self.assertGreater(record['gradient_norm'],0.)
        self.assertLess(float(self.model.weight),1.)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
