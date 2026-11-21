"""Loss and gradient parity for the TaskSet answer-only vocabulary projection."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from transformers.loss.loss_utils import ForCausalLMLoss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/benchmark/domains'))
from taskset_answer_logits import install_orpo_answer_logits, install_sft_answer_logits


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(19, 8, dtype=torch.float64)
        self.head = torch.nn.Linear(8, 19, dtype=torch.float64)

    def forward(self, input_ids, labels=None, logits_to_keep=0, **kwargs):
        hidden = self.embed(input_ids)
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        logits = self.head(hidden)
        loss = ForCausalLMLoss(logits, labels, 19) if labels is not None else None
        return SimpleNamespace(logits=logits, loss=loss)


class AnswerLogitsParity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.full, self.tail = Toy(), Toy()
        self.tail.load_state_dict(self.full.state_dict())
        self.inputs = torch.randint(0, 19, (2, 12))
        self.labels = self.inputs.clone()
        self.labels[:, :8] = -100
        self.labels[0, -2:] = -100

    def verify(self, full_loss, tail_loss):
        full_loss.backward()
        tail_loss.backward()
        self.assertTrue(torch.allclose(full_loss, tail_loss, atol=1e-7, rtol=1e-7))
        for full, tail in zip(self.full.parameters(), self.tail.parameters()):
            self.assertTrue(torch.allclose(full.grad, tail.grad, atol=1e-7, rtol=1e-7))

    def test_sft(self):
        install_sft_answer_logits(self.tail)
        self.verify(self.full(input_ids=self.inputs, labels=self.labels).loss,
                    self.tail(input_ids=self.inputs, labels=self.labels).loss)

    def test_original_orpo(self):
        from llamafactory.train.dpo.trainer import CustomDPOTrainer
        original = CustomDPOTrainer.concatenated_forward
        trainer = CustomDPOTrainer.__new__(CustomDPOTrainer)
        trainer.loss_type, trainer.beta, trainer.ld_alpha = 'orpo', 0.3, None
        trainer.finetuning_args = SimpleNamespace(use_ref_model=False)
        batch = {'input_ids': self.inputs, 'labels': self.labels}
        full = trainer.concatenated_forward(self.full, dict(batch))
        try:
            install_orpo_answer_logits()
            tail = trainer.concatenated_forward(self.tail, dict(batch))
            self.verify(trainer.odds_ratio_loss(full['chosen_logps'], full['rejected_logps']).mean(),
                        trainer.odds_ratio_loss(tail['chosen_logps'], tail['rejected_logps']).mean())
        finally:
            CustomDPOTrainer.concatenated_forward = original


if __name__ == '__main__':
    unittest.main()
