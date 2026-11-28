#!/usr/bin/env python3
"""Build the selected final-model inventory without copying or loading weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

REPO = Path(__file__).resolve().parents[1]
C = Path('/root/data/cys/baseline/campaigns')
L = Path('/root/data/logs')
B = Path('/root/data/cys/BOLT')
META = {'config.json', 'generation_config.json', 'processor_config.json',
        'preprocessor_config.json', 'video_preprocessor_config.json', 'tokenizer.json', 'tokenizer_config.json',
        'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt',
        'chat_template.jinja', 'adapter_config.json', 'model.safetensors.index.json',
        'LICENSE', 'LICENSE.txt', 'NOTICE', 'README.md'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def specifications():
    specs = []

    def add(name, domain, kind, path, note, depends=(), mode='file'):
        specs.append(dict(id=name, domain=domain, kind=kind, source=str(path),
                          note=note, dependencies=list(depends), mode=mode))

    base = 'qwen35-9b-exact-base'
    add(base, 'shared', 'full-model', '/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B',
        'Exact local base for adapters, BORA and pure Qwen comparisons.', mode='hf')
    for domain, date, step, orpostep in [('query','20261112',1224,'/checkpoint-103'),
                                       ('peptide','20261111',1500,'')]:
        root = C/f'{domain}_our_orpo_rl800_{date}'
        add(f'{domain}-warmstart', domain, 'full-model',
            C/f'{domain}_warmstart_full800_{date}/model/checkpoint-{step}',
            f'Exact selected warm-start checkpoint-{step}; optimizer and intermediate weights excluded.', mode='hf')
        add(f'{domain}-orpo', domain, 'adapter', str(root/'models/orpo')+orpostep,
            'Selected ORPO adapter. Query current evaluation has rl_enabled=false.', [base], 'hf')
        add(f'{domain}-orpo-rl', domain, 'adapter', root/'cycles/round1',
            'Final trained ORPO+RL variant; distinct from the ORPO-only evaluation.', [base], 'hf')
        add(f'{domain}-semanticopt', domain, 'adapter',
            C/f'semanticopt_discrete_20261109/{domain}/model/final_adapter',
            'Completed domain SemanticOPT SFT adapter, trained directly on the common base.', [base], 'hf')
        add(f'{domain}-retrieval-gp-means', domain, 'retrieval-assets', root/'assets/source_gp_mean',
            'Compact source GP means for retrieval; not a replacement for the RGPE covariance bank.', mode='tree')
    add('peptide-bolt-warmstart','peptide','full-model',
        '/root/data/cys/baseline/artifacts/bolt_peptide_top10_full_sft/model_adamw8bit',
        'Original completed BOLT top10 full SFT, step2500; preserve separately from our warm start.', mode='hf')
    add('query-vae80','query','vae',C/'query_our_fsbo_vae80_test200_20261117/training/vae/best.ckpt',
        'Selected VAE80 checkpoint; also used frozen by the newer BOLT30/LOLBO continuation.')
    add('query-fsbo-vae80','query','fsbo',C/'query_our_fsbo_vae80_test200_20261117/training/fsbo/weights',
        'Source FSBO rebuilt in VAE80 coordinates.', ['query-vae80'])
    add('query-author-vae','query','vae','/root/data/cys/logs/query_train800_author_fixed_20261004/native/vae/CEB_64_author.ckpt',
        'Frozen author/retrieval encoder; must remain distinct from VAE80.')
    add('query-frozen-fsbo','query','fsbo',L/'query_transfer_train800_20261106_v1/fsbo/weights',
        'Frozen source FSBO baseline.', ['query-author-vae'])
    for name, path in [('query-bolt-vae',B/'optimization/query_plans/query_plan_optimization/vae/CEB_64.ckpt'),
                       ('query-bolt-ft-vae',B/'optimization/query_plans/query_plan_optimization/vae/CEB_64_ft.ckpt'),
                       ('query-bayesqo-vae',Path('/root/data/cys/BayesQO/bayes_lqo/vae/CEB_64.ckpt'))]:
        add(name,'query','vae',path,'Historical native VAE variant; use only with its corresponding recipe.')
    add('peptide-author-vae','peptide','vae',B/'optimization/peptides/uniref_vae/saved_models/dim128_k1_kl0001_eff256_dff256_pious-sea-2_model_state_epoch_118_pretrained_authors.pkl',
        'Released author VAE used by frozen baselines.')
    add('peptide-adapted-vae','peptide','vae',L/'peptide_target_loss_gate_test10_20261119/vae/weights.pt',
        'Selected offline adapted VAE before online per-task updates.')
    add('peptide-adapted-fsbo','peptide','fsbo',L/'peptide_target_loss_gate_test10_20261119/fsbo/weights',
        'Source800 FSBO rebuilt for the offline adapted VAE.', ['peptide-adapted-vae'])
    add('peptide-frozen-fsbo','peptide','fsbo',L/'peptide_transfer_train800_20261027_v2/fsbo/weights',
        'Frozen baseline source weights.', ['peptide-author-vae'])
    add('peptide-apex-oracle','peptide','oracle',B/'optimization/peptides/apex_oracle/APEX_pathogen_models',
        'Eight pathogen MIC predictor weights; preprocessing source and init_data are separate dependencies.', mode='tree')
    for domain, path in [('query',L/'query_transfer_train800_20261106_v1/rgpe/models.sqlite'),
                         ('peptide',L/'peptide_transfer_train800_20261027_v2/rgpe/models.sqlite')]:
        add(f'{domain}-rgpe',domain,'rgpe-bank',path,
            'Immutable full source GP payloads; exported by task, reconstructed with original metadata and hashes.',
            [f'{domain}-author-vae'], 'sqlite')
    online = Path('/root/data/discreteDomains-final-models/online')
    for version in ['online-original', 'online-rollback', 'online-feasible-search']:
        if (online/version/'index.json').is_file():
            add(f'peptide-{version}', 'peptide', 'final-task-parameters', online/version,
                'Final parameters for ten completed tasks; VAE, source FSBO prior, normalization and constraint model. '
                'This is not a full resume checkpoint or a fitted target posterior; refit needs the matching observations.',
                ['peptide-adapted-vae', 'peptide-adapted-fsbo', 'peptide-apex-oracle'], 'tree')
    return specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO/'migration/inventory/final-models.json')
    parser.add_argument('--reuse-hashes', action='store_true', help='Reuse prior digest only when size and mtime match.')
    args = parser.parse_args()
    prior = {}
    if args.reuse_hashes and args.output.exists():
        for item in json.loads(args.output.read_text())['models']:
            for f in item.get('files', []):
                prior[f['source']] = f
    models = specifications()
    for item in models:
        p = Path(item['source'])
        if not p.exists():
            raise FileNotFoundError(p)
        item['files'] = []
        if item['mode'] == 'sqlite':
            with sqlite3.connect(p.as_uri()+'?mode=ro', uri=True) as conn:
                rows = conn.execute('SELECT task,metadata,sha256,length(payload) FROM models ORDER BY task').fetchall()
            item['rows'] = [dict(task=t, metadata=m, sha256=s, bytes=n, name=f'models/{t}.pt') for t,m,s,n in rows]
            item['bytes'] = sum(r['bytes'] for r in item['rows'])
            item['hash_status'] = 'Stored source-row SHA256; payload bytes verified during export/upload and restore.'
        else:
            paths = [p] if p.is_file() else sorted(x for x in p.rglob('*') if x.is_file())
            if item['mode'] == 'hf':
                paths = [x for x in paths if x.parent == p and (x.name in META or x.suffix == '.safetensors')]
                if not any(x.suffix == '.safetensors' for x in paths):
                    raise RuntimeError(f'No model weights: {p}')
            for f in paths:
                stat = f.stat()
                cached = prior.get(str(f), {})
                sha = cached.get('sha256') if (cached.get('bytes'), cached.get('mtime_ns')) == (stat.st_size, stat.st_mtime_ns) else None
                sha = sha or digest(f)
                record = dict(source=str(f), name=f.name if f == p else f.relative_to(p).as_posix(),
                              bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha)
                if str(f.resolve()) != str(f):
                    record['canonical_source'] = str(f.resolve())
                item['files'].append(record)
            item['bytes'] = sum(f['bytes'] for f in item['files'])
            item['hash_status'] = 'All listed file bytes SHA256 verified locally.'
        print(f"{item['id']}: {item['bytes']:,} bytes", flush=True)
    result = dict(schema_version=1, captured='2026-11-28', models=models,
                  total_payload_bytes=sum(m['bytes'] for m in models),
                  excluded='Intermediate checkpoints, optimizer/RNG/trainer state, live run journals and scientific datasets.',
                  online_models='Final per-task online VAE/FSBO exports have their own model entries and online-final-models.json provenance.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    online_index = Path('/root/data/discreteDomains-final-models/online/index.json')
    if online_index.exists():
        (args.output.parent/'online-final-models.json').write_text(online_index.read_text())


if __name__ == '__main__':
    main()
