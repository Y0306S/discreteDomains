"""Sequence I/O around the released BORA three-draft/merge assistant."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from common import ROOT, valid

sys.path.insert(0, str(ROOT/'vendor'))
from bora import assistant as official
from bora.experiment import Type

# Strings have a canonical equality key, rather than upstream numeric rounding.
official.hashable = lambda values: tuple(values)


class SequenceComment(official.Comment):
    def __init__(self, response, iteration, experiment, *args, **kwargs):
        self._iteration, self._experiment = iteration, experiment
        self._comment, self._hypotheses = None, []
        self.type = official.CommentType.POINTS
        self._keys, self._pbounds = ['sequence'], {}
        if response is not None:
            try:
                self._process_response(response)
            except (ValueError, KeyError, TypeError, AttributeError):
                self._comment, self._hypotheses = None, []

    def _chat_completion(self, messages):
        return self._experiment.bridge.create(messages=messages).choices[0].message.content

    def _clean_hypothesis_keys(self, hypothesis):
        if not isinstance(hypothesis, dict) or not all(k in hypothesis for k in ['name', 'rationale', 'confidence', 'points']):
            return False, {}
        points = hypothesis['points']
        if isinstance(points, dict):
            points = [points]
        if not isinstance(points, list):
            return False, {}
        clean = {k:hypothesis[k] for k in ['name', 'rationale', 'confidence']}
        clean['points'] = []
        for point in points:
            if isinstance(point, dict) and set(point) == {'sequence'} and valid(point['sequence'], self._experiment.reference):
                clean['points'].append(dict(point))
            else:
                self._experiment.owner.record(dict(stage='invalid_assistant_point', point=point))
        return bool(clean['points']), clean


official.Comment = SequenceComment


class Bridge:
    def __init__(self, owner):
        self.owner = owner
        self.chat = SimpleNamespace(completions=self)

    def create(self, *, messages, n=1, **kwargs):
        transported = copy.deepcopy(messages)
        for i, message in enumerate(transported):
            if i and message['role'] == 'system':
                message['role'] = 'assistant'
        choices = []
        for _ in range(n):
            try:
                text = self.owner.complete(transported)
            except Exception as exc:
                self.owner.inference_error = exc
                raise
            choices.append(SimpleNamespace(message=SimpleNamespace(content=text)))
        return SimpleNamespace(choices=choices, usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0))


class Assistant(official.Assistant):
    def __init__(self, owner):
        self.owner = owner
        bridge = Bridge(owner)
        parameter = SimpleNamespace(name='sequence', type=Type.continuous, get_bounds=lambda: (0, 1))
        experiment = SimpleNamespace(owner=owner, bridge=bridge, name=owner.task,
            reference=owner.reference, dim=256, description='Constrained antimicrobial peptide optimization',
            parameters=[parameter], keys=['sequence'], pbounds={}, constraint=None,
            type=Type.continuous, default_precision=15, domain='antimicrobial peptides',
            target=SimpleNamespace(name='negative APEX bacteria_0 mean MIC (higher is better)'))
        super().__init__('local-unused-no-network', experiment, np.zeros(1),
            str(owner.directory/'bora-comments.md'), llm_model='local-qwen', random_seed=owner.cfg['seed'])
        self._client = bridge
        self._chat_history = [dict(role='system', content=(
            'You are BORA, a Bayesian Optimization Research Assistant for antimicrobial peptides. '
            'Maximize negative APEX bacteria_0 mean MIC. Lower positive MIC is better. '
            'Every proposed point must be exactly {"sequence":"AMINOACIDS"}. '
            'Only the 20 canonical amino acids ACDEFGHIKLMNPQRSTVWY are allowed; length 1 to 30; '
            'at most one cysteine C; similarity 1-Levenshtein_distance(sequence,reference)/len(reference) >= 0.75. '
            'Avoid previously observed sequences. Return exactly one JSON object with "comment" and '
            '"hypotheses"; each hypothesis has "name", "rationale", "confidence", "points" '
            '(a list containing one sequence point). Do not invent measurements. '
            'Reference sequence: '+owner.reference))]

    def _init_log_path(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return path

    def _get_model_info(self):
        return dict(max_tokens=self.owner.cfg['max_input_tokens']+self.owner.cfg['max_new_tokens'],
            prompt_tokens_price_per_1k=0, completion_tokens_price_per_1k=0)

    def _chat_completion(self, *args, **kwargs):
        result = super()._chat_completion(*args, **kwargs)
        if self.owner.inference_error is not None:
            raise self.owner.inference_error
        return result

    def _add_assistant_message(self, text):
        if text is not None:
            super()._add_assistant_message(text)

    def _get_token_count(self, text):
        return len(self.owner.tokenizer.encode(text))

    def _constraint_violation(self, comment):
        return comment  # SequenceComment checked exact physical constraints.

    def _context(self, data, selection=None):
        rows = data.to_dict('records')
        history = [dict(iteration=int(r['iteration']), sequence=r['sequence'], score=float(r['score'])) for r in rows]
        def assemble(observations, summary=None):
            result = dict(task=self.owner.task, reference=self.owner.reference,
                objective='maximize negative MIC', evaluated_count=len(history), observations=observations)
            if summary is not None:
                result['history_summary'] = summary
            if selection is not None:
                result['allowed_candidates'] = [{'sequence':s} for s in selection]
            result['previous_hypotheses'] = [str(c) for c in self._comments[-1:]]
            return json.dumps(result, separators=(',', ':'))
        text = assemble(history)
        # Explicit sequence-domain summary replaces unsupported numeric correlations.
        # The released merge request appends three complete draft responses.
        # Reserve their worst-case lengths before deciding whether to summarize.
        draft_reserve = 3*self.owner.cfg['max_new_tokens']+1024
        if self.owner.prompt_tokens(self._chat_history+[dict(role='user', content=text)]) > self.owner.cfg['max_input_tokens']-draft_reserve:
            order = sorted(range(len(history)), key=lambda i: -history[i]['score'])
            keep = sorted(set(order[:40]) | set(range(max(0, len(history)-40), len(history))))
            scores = [r['score'] for r in history]
            summary = dict(policy='best40_recent40_plus_all_history_score_statistics',
                total=len(history), best=max(scores), worst=min(scores), mean=float(np.mean(scores)),
                omitted_rows=len(history)-len(keep), full_history_preserved_in_observation_ledger=True)
            text = assemble([history[i] for i in keep], summary)
            self.owner.record(dict(stage='history_summary', **summary))
        return text

    def _generate_comment_prompt(self, data):
        return (self._context(data)+'\nReflect on the observed data and previous hypotheses. '
            'Propose hypotheses for improving negative MIC; give one new feasible sequence per hypothesis. '
            'Use the exact JSON format from the system message.')

    def _generate_comment_selection_prompt(self, data, suggestions):
        return (self._context(data, self.owner.selection)+'\nGive two hypotheses, each containing one '
            'different sequence chosen exactly from allowed_candidates. Do not alter the offered sequences. '
            'Use the exact JSON format from the system message.')

    def ensure_overview(self):
        if self._experiment_overview is None:
            prompt = ('Describe the experimental objective, exact sequence constraints, and useful scientific '
                'considerations for this peptide task. There are 100 shared initial measurements. '
                'Do not propose or claim any measured outcomes yet.')
            self._experiment_overview = self._chat_completion(self._chat_history+[dict(role='user', content=prompt)])
            self._add_assistant_message(self._experiment_overview)

    def suggest(self, data, observed):
        self.ensure_overview()
        comment = self.comment_optimization(data, {(s,):True for s in observed})
        return [] if comment is None else [p['sequence'] for h in comment.hypotheses for p in h['points']]

    def select(self, data, candidates):
        self.ensure_overview()
        self.owner.selection = list(candidates)
        self._add_user_message(self._generate_comment_selection_prompt(data, None))
        try:
            for _ in range(3):
                comment = SequenceComment(self._chat_completion(self._chat_history, n=3), len(data)+1, self._experiment)
                if not comment.is_valid:
                    continue
                seen = set()
                for h in comment.hypotheses:
                    chosen = []
                    for point in h['points']:
                        s = point['sequence']
                        if s in candidates and s not in seen:
                            chosen.append(point)
                            seen.add(s)
                    h['points'] = chosen
                comment.hypotheses = [h for h in comment.hypotheses if h['points']]
                if comment.is_valid:
                    self._comments.append(comment)
                    self._update_comment_log_file()
                    return [p['sequence'] for h in comment.hypotheses for p in h['points']]
        finally:
            self._chat_history.pop()
        return []

    def snapshot(self):
        return dict(chat=self._chat_history, overview=self._experiment_overview, total_tokens=self._total_tokens,
            comments=[dict(iteration=c.iteration, comment=c.comment, hypotheses=c.hypotheses, type=c.type.name) for c in self._comments])

    def restore(self, state):
        self._chat_history = state['chat']
        self._experiment_overview = state['overview']
        self._total_tokens = state['total_tokens']
        self._comments = []
        for saved in state['comments']:
            c = SequenceComment(None, saved['iteration'], self._experiment)
            c._comment, c._hypotheses = saved['comment'], saved['hypotheses']
            c.type = official.CommentType[saved['type']]
            self._comments.append(c)
