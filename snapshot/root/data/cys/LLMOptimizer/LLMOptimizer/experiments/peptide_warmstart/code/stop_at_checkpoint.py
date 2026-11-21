"""One-shot controller: retain a verified epoch checkpoint and stop this SFT run."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from train import ROOT, atomic, load, sha, checkpoint_files, verify_identity

POLICY = ROOT/'stop_policy.json'
STATE = ROOT/'services/stop_at_epoch3.json'

def process_identity(pid):
    p=Path('/proc')/str(pid)
    fields=(p/'stat').read_text().rsplit(') ',1)[1].split()
    return dict(pid=pid,start_ticks=fields[19],command=[s.decode() for s in (p/'cmdline').read_bytes().split(b'\0') if s])

def alive(expected):
    try:
        p=Path('/proc')/str(expected['pid'])
        if (p/'stat').read_text().rsplit(') ',1)[1].startswith('Z'):return False
        return process_identity(expected['pid'])==expected
    except (OSError,ValueError):return False

def send(expected,sig):
    if alive(expected):os.kill(expected['pid'],sig)

def readiness(root,policy):
    target=policy['stop_step'];checkpoint=root/'model'/f'checkpoint-{target}'
    statuses=[load(root/f'rank{rank}_status.json') for rank in range(2)]
    for rank,s in enumerate(statuses):
        if s['rank']!=rank or s.get('step',0)>target+1:
            raise RuntimeError('Stop target was missed or rank identity changed; no automatic fallback')
    if not all(s.get('step',0)>=target and s.get('latest_checkpoint')==str(checkpoint)
               and s.get('saved_weight_samples_match') and s.get('first_update_verified') for s in statuses):
        return None,statuses
    checkpoint_files(checkpoint,target)
    state=load(checkpoint/'trainer_state.json')
    if abs(state['epoch']-policy['stop_epoch'])>1e-6:
        raise ValueError('Checkpoint epoch does not match stop policy')
    return checkpoint,statuses

def attach(root,policy):
    queue=process_identity(load(root/'status.json')['pid'])
    launch=process_identity(load(root/'services/sft_process.json')['pid'])
    ranks=[process_identity(load(root/f'rank{rank}_status.json')['pid']) for rank in range(2)]
    if str(root/'code/queue_training.py') not in queue['command']:
        raise ValueError('Wrong queue process')
    for p in [launch]+ranks:
        if str(root/'code/train.py') not in p['command'] or 'train' not in p['command']:
            raise ValueError('Wrong SFT trainer process')
    return dict(queue=queue,launcher=launch,ranks=ranks)

def publish_model(checkpoint,policy,statuses):
    """Link immutable BF16 model shards; keep the full restart checkpoint intact."""
    output=ROOT/'model';shards=checkpoint_files(checkpoint,policy['stop_step'])
    for source in checkpoint.iterdir():
        if not source.is_file() or source.name in ('trainer_state.json','training_args.bin','latest') or source.name.startswith('rng_state_'):
            continue
        if source.suffix not in ('.json','.safetensors','.jinja','.txt','.model'):continue
        target=output/source.name
        if target.exists():
            if sha(target)!=sha(source):raise ValueError('Final export already exists with different content: '+str(target))
            continue
        temp=target.with_name(target.name+'.epoch3.tmp');temp.unlink(missing_ok=True)
        os.link(source,temp);os.replace(temp,target)
    hashes={name:sha(output/name) for name in sorted(shards)}
    proof=dict(passed=True,at=time.time(),shards=hashes,optimizer_steps=policy['stop_step'],epochs=policy['stop_epoch'],
        model_config_sha256=sha(output/'config.json'),source_checkpoint=str(checkpoint),
        saved_weight_samples_match=True,verification='Both original training rank callbacks verified checkpoint samples against live gathered weights',
        original_scheduler_steps=policy['original_scheduler_steps'],stop_policy_sha256=sha(POLICY))
    atomic(ROOT/'evidence/final_weights.json',proof)
    for rank,s in enumerate(statuses):
        value=dict(s,status='completed',at=time.time(),step=policy['stop_step'],epoch=policy['stop_epoch'],
            max_steps=policy['stop_step'],original_scheduler_steps=policy['original_scheduler_steps'],
            completion_reason='user_requested_stop_at_verified_epoch_checkpoint',
            final_checkpoint=str(checkpoint),model_dir=str(output),final_weights_evidence=str(ROOT/'evidence/final_weights.json'))
        atomic(ROOT/f'rank{rank}_status.json',value)
        if rank==0:atomic(ROOT/'training_status.json',value)
    atomic(ROOT/'status.json',dict(at=time.time(),pid=os.getpid(),status='completed',stage='finished',
        optimizer_steps=policy['stop_step'],epochs=policy['stop_epoch'],model_dir=str(output),
        final_checkpoint=str(checkpoint),final_weights_evidence=str(ROOT/'evidence/final_weights.json'),
        completion_reason='user_requested_stop_at_verified_epoch_checkpoint'))

def finish(policy,processes,checkpoint,statuses):
    frozen=[]
    try:
        # The old queue expects 2500 steps. Hold it before stopping its children,
        # then replace its finalization with the explicit audited epoch stop.
        for p in [processes['queue'],processes['launcher']]+processes['ranks']:
            if not alive(p):raise RuntimeError('Owned SFT process exited before checkpoint stop')
            send(p,signal.SIGSTOP);frozen.append(p)
        actual,_=readiness(ROOT,policy)
        if actual!=checkpoint:raise RuntimeError('Checkpoint readiness changed')
        atomic(STATE,dict(status='stopping_at_checkpoint',at=time.time(),checkpoint=str(checkpoint)))
        # Freeze and kill only exact recorded processes; optimizer/model/RNG
        # have already been saved and checked by both ranks.
        for p in frozen:send(p,signal.SIGKILL)
        deadline=time.monotonic()+30
        while any(alive(p) for p in frozen) and time.monotonic()<deadline:time.sleep(.25)
        if any(alive(p) for p in frozen):raise RuntimeError('An SFT process did not terminate')
        publish_model(checkpoint,policy,statuses)
        from queue_training import scheduling
        scheduling(remote_reserved=[],local_reserved=[])
        atomic(STATE,dict(status='completed',at=time.time(),checkpoint=str(checkpoint),
            stop_step=policy['stop_step'],stop_epoch=policy['stop_epoch'],model_dir=str(ROOT/'model')))
    finally:
        for p in frozen:send(p,signal.SIGCONT)

def main(check_only=False):
    policy=load(POLICY);verify_identity()
    if policy['stop_step']!=1500 or policy['stop_epoch']!=3 or policy['original_scheduler_steps']!=2500:
        raise ValueError('Unexpected stop policy')
    guard=(ROOT/'services/stop_at_epoch3.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    processes=attach(ROOT,policy)
    checkpoint,statuses=readiness(ROOT,policy)
    if check_only:
        print(json.dumps(dict(valid=True,attached=processes,current_steps=[s['step'] for s in statuses],target=1500,checkpoint_ready=bool(checkpoint))));return
    atomic(ROOT/'services/stop_at_epoch3_process.json',dict(**process_identity(os.getpid()),started_at=time.time()))
    while True:
        if not all(alive(p) for p in [processes['queue'],processes['launcher']]+processes['ranks']):
            raise RuntimeError('Training process exited before the target checkpoint; inspect training logs')
        checkpoint,statuses=readiness(ROOT,policy)
        atomic(STATE,dict(status='armed',at=time.time(),pid=os.getpid(),stop_step=1500,stop_epoch=3,
            current_steps=[s['step'] for s in statuses],checkpoint_ready=bool(checkpoint),poll_seconds=1,
            stop_policy_sha256=sha(POLICY),mechanism='detached one-shot checkpoint stop controller'))
        if checkpoint:
            finish(policy,processes,checkpoint,statuses);return
        time.sleep(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    try:main(args.check)
    except Exception as exc:
        atomic(STATE,dict(status='error',at=time.time(),error=repr(exc)))
        raise
