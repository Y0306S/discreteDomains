#!/usr/bin/env python3
"""Audit the completed preference export with the real tokenizer and HPO parser."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'examples' / 'benchmark'))

from p5_response_format import CHAT_TEMPLATE_VERSION, parse_response, render_s1b_prompt
from regenerate_t2_rationales import canonical, read_jsonl, save_json, sha256, task_key, visible_configs


def audit(root, model):
    from transformers import AutoTokenizer
    from llamafactory.data.template import TEMPLATES
    from mindware.components.optimizers.pt4opt_llm_optimizer import parse_llm_json

    manifest = json.loads((root / 'manifest.json').read_text())
    source = read_jsonl(root / 'source.jsonl')
    rows = read_jsonl(root / 'p5_attr_t2_rationale.jsonl')
    training = read_jsonl(root / 'p5_attr_t2_rationale_train.jsonl')
    requests = {row['id']: row for row in read_jsonl(root / 'requests.jsonl')}
    teachers = {}
    weight_manifests = set()
    for path in root.glob('teacher_*.json'):
        teacher = json.loads(path.read_text())
        digest = hashlib.sha256(canonical(teacher).encode()).hexdigest()
        if path.name != f'teacher_{digest}.json' or teacher['request_sha256'] != manifest['request_sha256']:
            raise RuntimeError('Teacher provenance does not match this generation recipe')
        if teacher['generator_sha256'] != sha256(root / 'regenerate_t2_rationales.py'):
            raise RuntimeError('Generation did not use the frozen source snapshot')
        if teacher['model'] != str(model.resolve()):
            raise RuntimeError('Unexpected teacher model path')
        if teacher['adapter'] is not None or teacher['enable_thinking']:
            raise RuntimeError('Teacher settings differ from the approved recipe')
        teachers[digest] = teacher
        weight_manifests.add(canonical(teacher['files_sha256']))
    if len(weight_manifests) != 1:
        raise RuntimeError('Workers did not use one identical teacher checkpoint')
    train_tasks = {task_key(row['metadata']) for row in read_jsonl(root / 'train_manifest.jsonl')}
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    template = TEMPLATES['qwen3_5_nothink']
    counts = Counter()
    lengths = []
    review_sample = []
    sampled = set()
    if len(rows) != len(training) or len(rows) != manifest['eligible_pairs']:
        raise RuntimeError('Export count mismatch')
    for full, plain in zip(rows, training):
        meta = full['metadata']
        old = source[meta['source_index']]
        if task_key(meta) not in train_tasks:
            raise RuntimeError('Non-training task in export')
        if plain != {key: full[key] for key in ('conversations', 'chosen', 'rejected')}:
            raise RuntimeError('Training view differs from audited full export')
        if any(word in canonical(plain['conversations']) for word in ('No rationale.', 'Return only this JSON')):
            raise RuntimeError('Contradictory output instructions')
        system = full['conversations'][0]['value']
        user = full['conversations'][1]['value']
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
        rl_prompt = tokenizer(render_s1b_prompt(tokenizer, messages), add_special_tokens=False)['input_ids']
        prefixes = []
        visible = {canonical(config) for config in visible_configs(user)}
        for side in ('chosen', 'rejected'):
            if meta['rationale_generation'][side]['teacher_provenance'] not in teachers:
                raise RuntimeError('Missing teacher provenance for an exported answer')
            rationale, config = parse_response(full[side]['value'])
            actual, parsed_rationale, _ = parse_llm_json(full[side]['value'])
            if actual != config or canonical(config) != canonical(old['metadata'][side + '_config']):
                raise RuntimeError('Original configuration changed or live parser disagrees')
            if parsed_rationale != rationale:
                raise RuntimeError('Live parser did not recover the original rationale')
            if canonical(config) in visible:
                raise RuntimeError('Candidate outcome is visible in its training history')
            request = requests[meta['rationale_generation'][side]['request_id']]
            if canonical(request['config']) != canonical(config):
                raise RuntimeError('Teacher explained a different candidate')
            prefixes.append(request['messages'][1]['content'].partition('\n\nFIXED CANDIDATE:\n')[0])
            if request['messages'][0]['content'] != manifest['teacher_system']:
                raise RuntimeError('Teacher instructions differed between candidates')
            lf_prompt, lf_answer = template.encode_oneturn(
                tokenizer, [{'role': 'user', 'content': user},
                            {'role': 'assistant', 'content': full[side]['value']}], system=system)
            if lf_prompt != rl_prompt:
                raise RuntimeError('SFT/ORPO and RL prompt tokenization differs')
            answer = tokenizer(full[side]['value'], add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
            if lf_answer[:len(answer)] != answer or tokenizer.decode(lf_answer[len(answer):]) != '\n':
                raise RuntimeError('Answer tokens differ before the end-of-turn token')
            lengths.append(len(lf_prompt) + len(lf_answer))
            counts['answers_parsed_and_unchanged'] += 1
            counts['sft_orpo_rl_token_boundary_matches'] += 1
            group = (meta['task_type'], meta['algorithm'], side)
            if group not in sampled:
                sampled.add(group)
                review_sample.append({'source_index': meta['source_index'], 'task': task_key(meta),
                                      'side': side, 'config': config, 'rationale': rationale})
        if prefixes[0] != prefixes[1]:
            raise RuntimeError('Pair members received different historical context')
        counts['pairs'] += 1
        counts[meta['task_type'] + '_pairs'] += 1
    if max(lengths) > 6016:
        raise RuntimeError('An example exceeds the original ORPO token budget; refusing silent truncation')
    counts['training_tasks'] = len({task_key(row['metadata']) for row in rows})
    report = {
        'passed': True, 'checks': counts, 'chat_template_contract': CHAT_TEMPLATE_VERSION,
        'orpo_tokens_min': min(lengths), 'orpo_tokens_max': max(lengths),
        'orpo_tokens_mean': sum(lengths) / len(lengths), 'orpo_cutoff_checked': 6016,
        'source_sha256': sha256(root / 'source.jsonl'),
        'teacher_provenance_files': len(teachers),
        'training_file_sha256': sha256(root / 'p5_attr_t2_rationale_train.jsonl'),
        'audit_code_sha256': sha256(Path(__file__)),
        'semantic_review': 'stratified sample; automated checks do not prove every explanation factually correct',
        'training_launched': False,
    }
    save_json(root / 'validation.json', report)
    save_json(root / 'review_sample.json', review_sample)
    first_meta = rows[0]['metadata']
    save_json(root / 'prompt_example.json', {
        'note': 'Side labels below are for inspection only; the teacher receives only each messages list.',
        'messages_by_side': {
            side: requests[first_meta['rationale_generation'][side]['request_id']]['messages']
            for side in ('chosen', 'rejected')
        },
    })
    save_json(root / 'stage_inputs.json', {
        'scope': 'dataset and response-format settings; not a training launcher',
        'orpo': {'dataset_dir': str(root.resolve()), 'dataset': 'p5_attr_t2_rationale',
                 'template': 'qwen3_5_nothink', 'training_file_sha256': report['training_file_sha256']},
        'rl_collection': {'llm_response_format': 'rationale_json', 'qwen_enable_thinking': False},
        'rl_training': {'response_format': 'rationale_json', 'chat_template_contract': CHAT_TEMPLATE_VERSION},
        'training_launched': False,
    })
    code_dir = root / 'runtime_format_code'
    code_dir.mkdir(exist_ok=True)
    for name in ('p5_response_format.py', 'hf_qwen_client.py', 'train_p5_reinforce.py',
                 'run_p5_rl_bo_prefix_b.py', 'run_p5plus_rs_mix_prefix_rollouts.sh',
                 'audit_t2_rationale_export.py'):
        shutil.copyfile(REPO / 'examples' / 'benchmark' / name, code_dir / name)
    save_json(root / 'runtime_format_code_sha256.json',
              {path.name: sha256(path) for path in code_dir.iterdir() if path.is_file()})
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    audit(args.output, args.model)
