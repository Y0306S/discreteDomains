"""Avoid vocabulary projection on ignored prompt positions; keep the original losses."""

from functools import wraps


def answer_start(labels):
    positions = (labels != -100).nonzero(as_tuple=False)
    if not len(positions):
        raise ValueError('A training batch has no supervised answer tokens')
    return max(0, int(positions[:, 1].min().item()) - 1)


def install_sft_answer_logits(model):
    original = model.forward

    @wraps(original)
    def forward(*args, **kwargs):
        labels = kwargs.get('labels')
        if labels is not None:
            start = answer_start(labels)
            kwargs['labels'] = labels[:, start:]
            kwargs['logits_to_keep'] = labels.shape[1] - start
        return original(*args, **kwargs)

    model.forward = forward


def install_orpo_answer_logits():
    from llamafactory.train.dpo.trainer import CustomDPOTrainer

    original = CustomDPOTrainer.concatenated_forward

    @wraps(original)
    def concatenated_forward(self, model, batch, is_ref_model=False):
        if self.loss_type == 'orpo':
            labels = batch['labels']
            start = answer_start(labels)
            batch = dict(batch, labels=labels[:, start:], logits_to_keep=labels.shape[1] - start)
        return original(self, model, batch, is_ref_model=is_ref_model)

    CustomDPOTrainer.concatenated_forward = concatenated_forward
