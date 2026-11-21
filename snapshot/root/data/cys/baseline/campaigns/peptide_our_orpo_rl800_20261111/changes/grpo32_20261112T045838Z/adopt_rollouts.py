"""Copy unchanged completed groups into the authorized 32-task scope.

Original ledgers stay immutable in this directory. Only the copied ledger's
campaign identity changes; every sampled token, measurement and export is kept.
"""
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

ARCHIVE=Path(__file__).resolve().parent
ROOT=ARCHIVE.parents[1]
sys.path.insert(0,str(ROOT/'code'))
from campaign_io import canonical,sha,verify_campaign
from peptide_context import prompt
from peptide_prompt import atomic_json,digest
from rl_common import select_training_members


def states(path):
    with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as db:
        return {k:json.loads(v) for k,v in db.execute('SELECT key,value FROM state')}


def main():
    signature=verify_campaign(ROOT)
    old_signature=__import__('hashlib').sha256(canonical(json.loads((ARCHIVE/'checksums.json').read_text())).encode()).hexdigest()
    settings=json.loads((ROOT/'settings.json').read_text())
    parent_sha=sha(Path(settings['parent_adapter'])/'adapter_model.safetensors')
    imported=[]
    for job in json.loads((ROOT/'assets/jobs.json').read_text()):
        relative=Path(job['phase'])/job['task_id']/('k'+str(job['prefix_k']))/'r0'
        source=ARCHIVE/'rollouts'/relative
        if not (source/'group.sqlite').exists():continue
        group=states(source/'group.sqlite')
        if 'complete' not in group:continue
        assert group['identity']==dict(campaign=old_signature,group=job['group_id'],adapter_sha256=parent_sha)
        output=source/('rollouts_'+job['task_id']+'.jsonl')
        assert sha(output)==group['complete']['output_sha256']
        rows=[json.loads(line) for line in output.read_text().splitlines()]
        messages,meta,_,_=prompt(job)
        assert group['prompt']==dict(messages=messages,meta=meta)
        assert len(rows)==group['complete']['rows']
        selection=select_training_members(rows,job['group_id'],settings['training']['seed'])
        assert selection==group['complete']['selection']
        for row in rows:
            assert all(row[k]==job[k] for k in ['group_id','task_id','phase','scenario','prefix_k'])
            assert row['messages']==messages and row['policy_sha256']==parent_sha
            assert row['train_member']==(row['sample_i'] in selection['train_members'])
            generation=group['generation_'+str(row['sample_i'])]
            assert all(row[k]==generation[k] for k in ['completion','prompt_ids','answer_ids'])
            sample=states(source/f"sample{row['sample_i']}.sqlite")
            assert sample['identity']['campaign']==old_signature
            assert sample['record']['evaluation_status']==row['evaluation_status']
            if row['evaluation_status']=='SUCCESS':
                assert sample['evaluation_result']==row['evaluation']
                assert row['loss']==sample['evaluation_result']['objective']
        checks={str(p.relative_to(source)):sha(p) for p in source.iterdir() if p.is_file()}
        destination=ROOT/'rollouts'/relative
        if destination.exists():raise RuntimeError('Import destination already exists: '+str(destination))
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(source,destination)
        provenance=dict(source=str(source),source_campaign=old_signature,destination_campaign=signature,
                        source_files=checks,at=time.time(),original_export_preserved=True)
        for path in destination.glob('*.sqlite'):
            with sqlite3.connect(path) as db:
                identity=json.loads(db.execute("SELECT value FROM state WHERE key='identity'").fetchone()[0])
                assert identity['campaign']==old_signature
                identity['campaign']=signature
                db.execute("UPDATE state SET value=? WHERE key='identity'",(canonical(identity),))
                db.execute("INSERT INTO state(key,value) VALUES('scope_import',?)",(canonical(provenance),))
        assert sha(destination/output.name)==sha(output)
        imported.append(dict(group_id=job['group_id'],samples=len(rows),export_sha256=sha(output),source=str(source)))
    report=dict(passed=True,at=time.time(),original_campaign_signature=old_signature,
                campaign_signature=signature,imported_groups=len(imported),groups=imported,
                original_ledgers_preserved=True,script_sha256=sha(Path(__file__)))
    atomic_json(ROOT/'evidence/grpo32_rollout_import.json',report)
    print(canonical(report))


if __name__=='__main__':main()
