"""Independent full-history validation of generated pairs and visible causal history."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import tarfile
import time
from Levenshtein import distance
from peptide_prompt import atomic_json, file_sha

ROOT = Path(__file__).resolve().parents[1]


def audit_task(item):
    task, source, reference, sha = item
    with tarfile.open(Path(source)/'runs'/f'{task}.tar.gz', 'r|gz') as archive:
        for member in archive:
            if member.name.rsplit('/', 1)[-1] == 'history.json':
                raw = archive.extractfile(member).read()
                assert hashlib.sha256(raw).hexdigest() == sha
                history = json.loads(raw)
                break
        else:
            raise AssertionError('Missing source history')
    assert history['status'] == 'complete' and history['reference'] == reference
    assert history['identity']['task'] == task
    rows = history['observations']
    saved = json.loads((ROOT/'data/tasks'/f'{task}.json').read_text())
    pairs = saved['pairs']
    all_seen, valid_indices, snapshots = set(), [], {}
    cuts = {p['metadata']['prefix_end_exclusive'] for p in pairs}
    for i, r in enumerate(rows):
        if i in cuts:
            snapshots[i] = (set(all_seen), list(valid_indices))
        seq = r['sequence']
        sim = 1-distance(seq, reference)/len(reference)
        ok = (1<=len(seq)<=30 and set(seq)<=set('ACDEFGHIKLMNPQRSTVWY') and seq.count('C')<=1 and sim>=.75)
        assert bool(r['feasible']) == ok and math.isfinite(r['mic']) and math.isfinite(r['score'])
        assert abs(r['mic']+r['score'])<=1e-6
        if ok and seq not in all_seen:
            assert r['phase']=='initialization' or not r['cached']
            valid_indices.append(i)
        all_seen.add(seq)
    for pair in pairs:
        m = pair['metadata']
        cut, end = m['prefix_end_exclusive'], m['block_end_exclusive']
        seen, prefix = snapshots[cut]
        assert cut>0 and (rows[cut-1]['phase'],rows[cut-1]['round'])!=(rows[cut]['phase'],rows[cut]['round'])
        assert end==len(rows) or (rows[end-1]['phase'],rows[end-1]['round'])!=(rows[end]['phase'],rows[end]['round'])
        baseline = min(rows[i]['mic'] for i in prefix)
        assert baseline == m['current_best_mic']
        context = json.loads(pair['conversations'][1]['value'])
        assert context['reference_sequence']==reference and context['current_best_mic']==baseline
        assert len(context['rules'])==6 and all(r['example'] for r in context['rules'])
        assert context['valid_distinct_evaluations_so_far']==len(prefix)
        expected = dict(recent50=prefix[-50:],
                        good=sorted(prefix,key=lambda i:(rows[i]['mic'],i))[:5],
                        bad=sorted(prefix,key=lambda i:(-rows[i]['mic'],i))[:5])
        for key, indices in expected.items():
            visible = context[key]
            assert [r['observation_index'] for r in visible] == indices
            if key=='recent50':
                assert len(visible)==50 and len({r['sequence'] for r in visible})==50
            for shown, index in zip(visible,indices):
                assert shown['sequence']==rows[index]['sequence'] and shown['mic']==rows[index]['mic']
                assert shown['similarity']==1-distance(shown['sequence'],reference)/len(reference)
        prompt = '\n'.join(r['value'] for r in pair['conversations'])
        for side in ('chosen','rejected'):
            index = m[side+'_index']
            answer = json.loads(pair[side]['value'])
            assert set(answer)=={'sequence'} and answer['sequence']==rows[index]['sequence']
            seq = answer['sequence']
            assert seq not in seen and seq not in prompt and rows[index]['feasible']
            assert index>=cut and not rows[index]['cached']
            assert rows[index]['mic']==m[side+'_mic']
        assert cut<=m['chosen_index']<end
        assert m['chosen_mic']<baseline<=m['rejected_mic']
        assert pair['chosen']['value']!=pair['rejected']['value']
        positives=[i for i in valid_indices if cut<=i<end and rows[i]['mic']<baseline
                   and rows[i]['sequence'] not in seen and rows[i]['sequence'] not in prompt]
        assert m['chosen_index']==min(positives,key=lambda i:(rows[i]['mic'],i))
        chosen=rows[m['chosen_index']]
        pool=[i for i in valid_indices if i>=cut and rows[i]['sequence'] not in seen
              and rows[i]['mic']>=baseline and rows[i]['sequence']!=chosen['sequence'] and rows[i]['sequence'] not in prompt]
        gaps=[rows[i]['mic']-chosen['mic'] for i in pool]
        lo,hi=min(gaps),max(gaps)
        scored=[]
        for i,gap in zip(pool,gaps):
            seq=rows[i]['sequence']
            sim=1-distance(seq,chosen['sequence'])/max(len(seq),len(chosen['sequence']))
            norm=1. if hi<=lo+1e-15 else (gap-lo)/(hi-lo)
            scored.append((.5*sim+.5*norm,sim,norm,-i))
        # A positive weighted maximum must be Pareto-optimal; verify selected maximum independently.
        best=max(scored)
        assert m['rejected_index']==-best[3]
        assert m['negative_selection']['weighted']==best[0]
        assert m['negative_selection']['pool_size']==len(pool)
    return dict(task=task, pairs=len(pairs), observations=len(rows), passed=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    start=time.time()
    manifest=json.loads((ROOT/'manifest.json').read_text())
    assert manifest['tasks_scanned']==800 and manifest['test_tasks_used']==0 and not set(manifest['split']['train'])&set(manifest['split']['test'])
    original=json.loads(Path('/root/reports/peptide_train800_test200_audit_20261027/train800_audit.json').read_text())
    checks={r['task']:r for r in original['tasks']}
    recipe=json.loads((Path(manifest['source'])/'recipe.json').read_text())
    assert manifest['split']==dict(train=recipe['train'],test=recipe['test'])
    jobs=[(t,manifest['source'],recipe['references'][t],checks[t]['member_checksums']['history.json']) for t in manifest['split']['train']]
    results, errors=[],{}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        fs={ex.submit(audit_task,j):j[0] for j in jobs}
        for future in as_completed(fs):
            try:results.append(future.result())
            except Exception as error:errors[fs[future]]=repr(error)
            if (len(results)+len(errors))%50==0:
                print(json.dumps(dict(audited=len(results),errors=errors)),flush=True)
    # Verify the actual training file exactly matches audited per-task rows in source split order.
    count=0
    with (ROOT/'data/orpo_train.jsonl').open() as stream:
        for task in manifest['split']['train']:
            for expected in json.loads((ROOT/'data/tasks'/f'{task}.json').read_text())['pairs']:
                assert json.loads(next(stream))==expected
                count+=1
        assert stream.readline()==''
    assert count==manifest['pairs'] and file_sha(ROOT/'data/orpo_train.jsonl')==manifest['output_sha256']
    report=dict(passed=not errors,tasks_audited=len(results),pairs_audited=count,errors=errors,
                all_history_hashes_verified=True if not errors else False, full_prefix_novelty=True if not errors else False,
                prompt_answer_leakage_count=0 if not errors else None, causal_batch_prefixes=True if not errors else False,
                recent50_valid_distinct_chronological=True if not errors else False,
                baseline_improvement_labels=True if not errors else False, data_sha256=file_sha(ROOT/'data/orpo_train.jsonl'),
                code_sha256={f.name:file_sha(f) for f in (ROOT/'code').glob('*.py')},seconds=time.time()-start)
    atomic_json(ROOT/'reports/independent_audit.json',report)
    print(json.dumps(report),flush=True)
    if errors:raise RuntimeError(errors)


if __name__=='__main__':main()
