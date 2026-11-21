"""Score a saved proposal queue on one CPU core and commit the results together."""
import math
import time
from protocol import canonical,value_hash


def score_and_commit(db,get,put,*,batch,start,total,validate,oracle,seen,runtime_sha,policy_sha):
    journal=get(db,'oracle_batch')
    if journal and journal['status']=='in_flight':
        raise RuntimeError('Unresolved peptide batch oracle call; requires audit before replay')
    if get(db,'pending') is not None:raise ValueError('Single-item oracle must finish before batching')
    offset=start-batch['start']
    if not 0<=offset<len(batch['items']):raise ValueError('Peptide oracle batch offset mismatch')
    items=batch['items'][offset:offset+min(50,total-start)]
    if not items:raise ValueError('Empty or exhausted peptide oracle batch')
    excluded=set(seen);configs=[]
    for item in items:
        config=item['answer']['next_configuration'];validate(config)
        key=canonical(config)
        if key in excluded:raise ValueError('Duplicate peptide in oracle batch')
        excluded.add(key);configs.append(config)
    if db.execute('select count(*) from records').fetchone()[0]!=start:
        raise ValueError('Peptide oracle batch cursor mismatch')
    journal=dict(status='in_flight',start=start,count=len(items),configurations=configs,
        configuration_sha256=value_hash(configs),started_at=time.time(),policy_sha256=policy_sha)
    with db:put(db,'oracle_batch',journal)
    results=oracle(configs)
    if not isinstance(results,list) or len(results)!=len(items):
        raise ValueError('Peptide oracle batch returned the wrong number of results')
    finished=time.time()
    for i,(result,item,config) in enumerate(zip(results,items,configs)):
        if result.get('configuration')!=config or not math.isfinite(result['objective']):
            raise ValueError('Peptide oracle batch result order/value mismatch')
        provenance=item['provenance'];number=start+i
        result.update(evaluation=number+1,at=finished,proposal_source=provenance['source'],
            proposal_provenance=provenance,runtime_policy_sha256=runtime_sha,
            batch_policy_sha256=batch['policy_sha256'],proposal_round=number//50,
            proposal_round_offset=number%50,oracle_execution='cpu_batch',
            oracle_batch_start=start,oracle_batch_size=len(items),
            oracle_batch_seconds=finished-journal['started_at'],peptide_oracle_policy_sha256=policy_sha)
    # All scores become durable in the same transaction as the completion flag.
    # No result can be committed twice or exposed without the remaining scores.
    with db:
        for i,result in enumerate(results):
            db.execute('INSERT INTO records VALUES (?,?)',(start+i,canonical(result)))
        put(db,'oracle_batch',dict(journal,status='committed',finished_at=finished))
    return results
