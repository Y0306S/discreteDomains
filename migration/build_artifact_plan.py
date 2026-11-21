#!/usr/bin/env python3
"""List non-Git payload roots and exact dependencies for server migration."""
import json
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[1]


def main():
    inventory = REPO/'migration/inventory'
    sources = json.loads((inventory/'sources.json').read_text())
    roots = json.loads((inventory/'roots.json').read_text())
    payload = [r for r in roots if r.startswith(('/root/data/logs/', '/root/data/cys/logs/',
                '/root/data/cys/baseline/campaigns/'))]
    payload += [
        '/root/peptide_our_orpo_rl_train800_20261110',
        '/root/data/cys/BOLT', '/root/data/cys/APEXGo', '/root/data/cys/BayesQO',
        '/root/data/cys/baseline/repos/BOLT', '/root/data/cys/baseline/repos/BayesQO',
        '/root/data/cys/baseline/repos/FSBO', '/root/data/cys/baseline/repos/lolbo',
        '/root/data/cys/baseline/repos/bora-the-explorer',
        '/root/data/cys/baseline/repos/transfer-hpo-framework',
        '/root/data/cys/baseline/artifacts/bolt_peptide_top10_full_sft',
        '/root/data/cys/baseline/artifacts/peptide_our_orpo_rl_train800_20261110',
        '/root/data/cys/baseline/cache/bolt_peptide_sft',
        '/root/data/cys/llmoptimizer_benchmark/benchmark_data/bolt_peptide_top10_train800_v1',
        '/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B',
    ]
    links = json.loads((inventory/'symlinks.json').read_text())
    # Dereference scientific input symlinks, but do not import unrelated
    # relocated application logs or the entire google-research checkout.
    payload += [v for k,v in links.items() if Path(k).name in {
        'corpus','fsbo','vae','source_snapshot','reference','deps','runtime_deps','query_deps'}]
    references = set()
    path_re = re.compile(r'/root/[A-Za-z0-9_./+@=:-]+')
    for r in sources.values():
        if Path(r['snapshot']).suffix not in {'.py','.sh','.json','.yaml','.yml','.toml','.cfg'}:
            continue
        content = (REPO/r['snapshot']).read_text(errors='replace')
        references.update(s.rstrip('.:') for s in path_re.findall(content))
    exact_files = []
    model_extensions = {'.pt','.pth','.ckpt','.pkl','.safetensors','.npz','.npy','.sqlite','.db'}
    for ref in sorted(references):
        if ref in sources or any(p in ref for p in ['/.ssh/','/.codex/','/.aws/','/site-packages/']):
            continue
        p = Path(ref)
        if p.suffix in model_extensions and p.is_file():
            exact_files.append({'path': ref, 'bytes': p.stat().st_size})
            if not any(ref == r or ref.startswith(r+'/') for r in payload):
                payload.append(ref)
    payload = sorted(set(payload))
    payload = [r for r in payload if not any(r != other and r.startswith(other+'/') for other in payload)]
    (inventory/'artifact-roots.txt').write_text(''.join(r.lstrip('/')+'\n' for r in payload))
    (inventory/'source-excludes.txt').write_text(''.join(r+'\n' for r in sorted(sources)))
    (inventory/'exact-artifacts.json').write_text(json.dumps(exact_files,indent=2)+'\n')
    (inventory/'path-references.txt').write_text('\n'.join(sorted(references))+'\n')
    print(f'{len(payload)} payload roots, {len(exact_files)} existing model/data file references')


if __name__ == '__main__':
    main()
