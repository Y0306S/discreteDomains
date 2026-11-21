#!/usr/bin/env python3
"""Snapshot experiment sources without changing the original experiments."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[1]
SNAPSHOT = REPO / 'snapshot'
CYS = Path('/root/data/cys')
SKIP = {
    '.git', '.cache', '__pycache__', '.pytest_cache', '.ruff_cache', 'node_modules',
    'envs', '.venv', 'venv', 'site-packages', 'unsloth_compiled_cache', 'wandb',
    'runs', 'evaluation', 'rollouts', 'cycles', 'scratch', 'hf_cache', 'huggingface',
    'models', 'model', 'saved_models', 'checkpoints', 'checkpoint', 'outputs',
    'logs', 'monitor', 'workers', 'locks', 'catboost_info', 'history_data',
    'benchmark_data', 'datasets_bundle', 'databases', 'pools', 'services',
    'warm30', 'orpo_batch10', 'source_snapshot', 'corpus', 'deps', 'fsbo', 'vae',
    'cache', 'assets_cache', 'artifacts', 'runtime_deps', 'query_deps',
    'campaigns', 'repos', 'packs', 'histories', 'data', 'eval_runs',
    'experiment_runs', 'claims', 'validation_fullscale', 'smoke_eval',
    'smoke_eval_fullscale',
}
EXTENSIONS = {
    '.py', '.sh', '.bash', '.sql', '.yaml', '.yml', '.toml', '.ini', '.cfg',
    '.json', '.jsonl', '.csv', '.tsv', '.txt', '.md', '.rst', '.lock', '.ipynb',
    '.c', '.h', '.cpp', '.cu', '.hpp', '.r', '.R', '.jl', '.patch', '.diff',
}
NAMES = {'LICENSE', 'COPYING', 'NOTICE', 'Makefile', 'Dockerfile', '.gitignore',
         '.gitattributes', 'Snakefile', 'environment', 'requirements'}
SECRET_NAME = re.compile(r'(?:token|password|credential|secret|private.key)', re.I)
COPIED = {}
OMITTED = []
LINKS = {}


def sanitize(data, source):
    text = data.decode('utf-8')
    if source.name == 'model_registry.py':
        text = re.sub(r'(?m)^PKU_LAB_API_KEY\s*=\s*[\"\x27][^\"\x27]+[\"\x27]',
                      'PKU_LAB_API_KEY = os.environ.get("PKU_LAB_API_KEY", "")', text)
    if source.name in {'run_ml_evals_parallel.sh', 'run_all_evals_parallel.sh'}:
        text = re.sub(r'(?m)^export OPENAI_HK_API_KEY=.*$',
                      'export OPENAI_HK_API_KEY="${OPENAI_HK_API_KEY:?Set OPENAI_HK_API_KEY}"', text)
    if source.name == '__init__.py' and 'rag/utils' in str(source):
        text = re.sub(r'("api_key"\s*:\s*)"[a-fA-F0-9]{64}"',
                      r'\1__import__("os").environ["SERPAPI_API_KEY"]', text)
    if source.name == 'configs.py' and 'mindware/modules' in str(source):
        text = re.sub(r'(?m)^(\s*OPENAI_KEY\s*=\s*)"[^"]+"',
                      r'\1__import__("os").environ.get("OPENAI_API_KEY", "")', text)
    if source.name == 'run_all_evals_maxgpu.py':
        text = re.sub(r'("--api-key",\s*)"[^"]+"',
                      r'\1__import__("os").environ["OPENAI_HK_API_KEY"]', text)
    return text.encode('utf-8')


def roots():
    out = [CYS/'llmoptimizer_benchmark', CYS/'BOLT', CYS/'APEXGo', CYS/'BayesQO',
           CYS/'LLMOptimizer', CYS/'baseline', Path('/root/peptide_our_orpo_rl_train800_20261110'),
           Path('/root/data/env-packs')]
    vendor = CYS/'baseline/repos'
    for name in ['BOLT', 'BayesQO', 'FSBO', 'bora-the-explorer', 'lolbo', 'transfer-hpo-framework']:
        out.append(vendor/name)
    campaigns = CYS/'baseline/campaigns'
    out.extend(p for p in campaigns.iterdir() if p.is_dir() and
               p.name.startswith(('peptide_', 'query_', 'semanticopt_discrete_', 'semanticopt_eval200_', 'semanticopt_peptide_')))
    for parent in [Path('/root/data/logs'), CYS/'logs']:
        out.extend(p for p in parent.iterdir() if p.is_dir() and p.name.startswith(('peptide_', 'query_')))
    for p in Path('/root/reports').iterdir():
        if p.is_dir() and p.name.startswith(('peptide_', 'query_')):
            out.append(p)
    return sorted(set(out))


def copy_file(source):
    source = Path(source)
    key = str(source)
    if key in COPIED or not source.is_file():
        return
    if SECRET_NAME.search(source.name) and not source.name.startswith(('test_', 'check_')) and source.suffix not in {'.py', '.md'}:
        OMITTED.append({'path': key, 'reason': 'credential filename'})
        return
    if source.suffix not in EXTENSIONS and source.name not in NAMES:
        return
    size = source.stat().st_size
    limit = 5 * 1024**2 if source.suffix in {'.py', '.sh', '.sql', '.md', '.lock'} else 512 * 1024
    if size > limit:
        OMITTED.append({'path': key, 'reason': 'large input; transfer separately', 'bytes': size})
        return
    data = source.read_bytes()
    if b'\0' in data:
        return
    original_hash = hashlib.sha256(data).hexdigest()
    try:
        data = sanitize(data, source)
    except UnicodeError:
        return
    dest = SNAPSHOT / source.relative_to('/')
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    shutil.copymode(source, dest)
    COPIED[key] = {'snapshot': str(dest.relative_to(REPO)), 'bytes': len(data),
                   'sha256': hashlib.sha256(data).hexdigest(), 'original_sha256': original_hash,
                   'sanitized': original_hash != hashlib.sha256(data).hexdigest()}


def collect_tree(root):
    if not root.exists():
        return
    for directory, dirs, files in os.walk(root, followlinks=False):
        directory = Path(directory)
        for name in dirs + files:
            p = directory/name
            if p.is_symlink():
                LINKS[str(p)] = str(p.resolve())
        dirs[:] = sorted(d for d in dirs if d not in SKIP and
                         not d.startswith(('checkpoint-', '.')) and not (directory/d).is_symlink())
        for name in sorted(files):
            copy_file(directory/name)


def collect_package_sources(selected):
    # Package directories named model/models/data/vae are source modules in
    # several upstream projects, even though those names denote heavy outputs
    # in campaign directories. Preserve their code with a separate pass.
    packages = [CYS/'BOLT', CYS/'APEXGo', CYS/'BayesQO',
                CYS/'llmoptimizer_benchmark/mindware',
                CYS/'LLMOptimizer/LlamaFactory/src',
                CYS/'LLMOptimizer/LLMOptimizer/src']
    for root in selected:
        if root.parent == CYS/'baseline/repos':
            packages.append(root)
        for name in ['code', 'batch_code', 'native', 'services']:
            if (root/name).is_dir():packages.append(root/name)
    hard_skip = {'.git', '__pycache__', '.cache', 'envs', 'venv', '.venv',
                 'site-packages', 'wandb', 'node_modules', 'unsloth_compiled_cache'}
    for root in sorted(set(packages)):
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in hard_skip and not (Path(directory)/d).is_symlink()]
            for name in files:
                p = Path(directory)/name
                if p.suffix in {'.py','.sh','.sql','.c','.h','.cpp','.cu','.hpp'}:
                    copy_file(p)


def main():
    selected = roots()
    for root in selected:
        collect_tree(root)
    collect_package_sources(selected)
    # Preserve copies of locally modified native codecs and transitive source
    # dependencies named explicitly by the copied code/configs. No execution.
    path_re = re.compile(r'/root/[A-Za-z0-9_./+@=:-]+')
    for _ in range(4):
        previous = len(COPIED)
        for record in list(COPIED.values()):
            if Path(record['snapshot']).suffix not in {'.py', '.sh', '.json', '.yaml', '.yml', '.toml'}:
                continue
            text = (REPO/record['snapshot']).read_text(errors='replace')
            for match in path_re.findall(text):
                p = Path(match.rstrip('.:'))
                if p.suffix in {'.py', '.sh', '.sql', '.yaml', '.yml', '.toml', '.cfg'}:
                    if not any(x in p.parts for x in ['.ssh', '.codex', '.aws', 'site-packages', 'envs']):
                        copy_file(p)
        if len(COPIED) == previous:
            break
    inventory = REPO/'migration/inventory'
    inventory.mkdir(parents=True, exist_ok=True)
    (inventory/'sources.json').write_text(json.dumps(COPIED, indent=2, sort_keys=True)+'\n')
    (inventory/'symlinks.json').write_text(json.dumps(LINKS, indent=2, sort_keys=True)+'\n')
    (inventory/'omitted.json').write_text(json.dumps(OMITTED, indent=2)+'\n')
    (inventory/'roots.json').write_text(json.dumps(list(map(str, selected)), indent=2)+'\n')
    upstream = []
    for root in selected:
        if (root/'.git').exists():
            def git(*args):
                return subprocess.run(['git', '-C', str(root), *args], text=True, capture_output=True).stdout.strip()
            # Avoid exporting credential-bearing remotes.
            remote = git('remote', 'get-url', 'origin')
            remote = re.sub(r'(https?://)[^/@]+@', r'\1', remote)
            upstream.append({'path': str(root), 'head': git('rev-parse', 'HEAD'),
                             'origin': remote, 'status': git('status', '--porcelain', '-uno')})
    (inventory/'upstream.json').write_text(json.dumps(upstream, indent=2)+'\n')
    print(f'Copied {len(COPIED)} files ({sum(r["bytes"] for r in COPIED.values())/1024**2:.1f} MiB) from {len(selected)} roots')


if __name__ == '__main__':
    main()
