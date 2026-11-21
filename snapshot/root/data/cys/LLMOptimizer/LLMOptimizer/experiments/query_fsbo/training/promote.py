"""Validate retrained checkpoints, import BOLT30 unchanged, and start the rerun."""
import collections,fcntl,json,os,shlex,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/root/data/cys/baseline/campaigns/query_our_fsbo_test200_20261116')
sys.path[:0]=[str(ROOT/'batch_code'),str(ROOT/'operations/recovery'),str(ROOT/'training')]
from train import load,atomic,digest
# train.py's preparation imports the parent implementation. Promotion must
# resolve every runtime module from this isolated campaign instead.
sys.path[:0]=[str(ROOT/'batch_code'),str(ROOT/'operations/recovery'),str(ROOT/'training')]
PY='/root/data/cys/envs/bolt_clean_venv/bin/python'
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONHASHSEED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
def command(args,name,timeout=900,cwd=None):
    print('START',name,flush=True)
    env=dict(ENV)
    if name=='preflight':
        spec=next(w for w in load(ROOT/'resources.json')['workers'] if w['host']=='LLMOptimizer2')
        env.update(DB_HOST='127.0.0.1',DB_PORT=str(spec['db_port']),DB_NAME='imdb',DB_USER='imdb',DB_PASSWORD='imdb')
    with (ROOT/'training'/f'promotion.{name}.log').open('ab') as log:
        result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=cwd,timeout=timeout)
    if result.returncode:raise RuntimeError(f'{name} exited {result.returncode}; inspect training/promotion.{name}.log')
    print('PASSED',name,flush=True)
def configurations():
    cfg=load(ROOT/'training/config.json');vs=load(ROOT/'training/vae/status.json');fs=load(ROOT/'training/fsbo/status.json');corpus=load(ROOT/'training/corpus/manifest.json')
    assert vs['status']=='complete' and vs['reload_verified'] and vs['weights_changed']
    assert fs['status']=='complete' and fs['reload_verified'] and fs['step']==100000
    assert corpus['vae_sha256']==digest(ROOT/'training/vae/best.ckpt')==vs['checkpoint_sha256']
    assert fs['identity']['corpus_sha256']==digest(ROOT/'training/corpus/manifest.json')
    assert set(corpus['source_tasks'])==set(cfg['source_tasks']) and not set(cfg['source_tasks'])&set(cfg['excluded_test_tasks'])
    fr=load(ROOT/'fsbo_recipe.json');fr.update(transfer=str(ROOT/'training'),representation=corpus['representation'],
        fsbo_weights=str(ROOT/'training/fsbo/weights'),fsbo_weights_sha256=digest(ROOT/'training/fsbo/weights'),
        vae_checkpoint=str(ROOT/'training/vae/best.ckpt'),vae_sha256=vs['checkpoint_sha256'],lower=corpus['lower'],upper=corpus['upper'],
        encoder_float32_matmul_precision='highest',latent_corpus_manifest_sha256=digest(ROOT/'training/corpus/manifest.json'),
        batch_size=10,initialization=30,proposals=770)
    for p in [ROOT/'training/vae/best.ckpt',ROOT/'training/fsbo/weights',ROOT/'training/corpus/manifest.json',ROOT/'training/config.json',ROOT/'training/train.py',ROOT/'training/gpu_reembed.py']:
        fr['pinned_files'][str(p)]=digest(p)
    atomic(ROOT/'fsbo_recipe.json',fr)
    recipe=load(ROOT/'batch_recipe.json');recipe['name']='query_our_fsbo_vae80_test200_20261117_batch10_v2'
    recipe['representation_retraining']=dict(training_tasks=cfg['requested_tasks'],fit_tasks=len(cfg['fsbo_train_tasks']),validation_tasks=len(cfg['fsbo_validation_tasks']),
        config_sha256=digest(ROOT/'training/config.json'),vae_sha256=vs['checkpoint_sha256'],fsbo_sha256=fr['fsbo_weights_sha256'],
        parent_campaign=str(OLD),optimization_history_reused=False,bolt_measurements_reused=True)
    atomic(ROOT/'batch_recipe.json',recipe)
    base=load(OLD/'batch_recipe.json');changed=[k for k in base if base[k]!=recipe.get(k)]
    assert set(changed)<= {'name','fsbo_recipe'}
    assert load(ROOT/'batch_policy.json')==load(OLD/'batch_policy.json')
    assert load(ROOT/'resources.json')==load(OLD/'resources.json')
    atomic(ROOT/'evidence/rerun_configuration_comparison.json',dict(passed=True,changed_recipe_fields=changed,
        added_fields=['representation_retraining'],batch_policy_identical=True,resource_topology_identical=True,models_identical=recipe['models']==base['models'],
        budget_evaluations_per_task=800,total_tasks=200,warm_measurements_reused=6000,at=time.time()))
def import_warm():
    from run import identity,freeze
    from method import validate_pool
    cfg=load(ROOT/'batch_recipe.json');sources={};phases=collections.Counter()
    old_cfg=load(OLD/'batch_recipe.json');assert cfg['tasks']==old_cfg['tasks']
    for task in cfg['tasks']:
        p=OLD/'warm30/runs'/task/'complete.json';record=load(p)
        assert record['recipe_sha256']==digest(OLD/'batch_recipe.json') and record['policy_sha256']==digest(OLD/'batch_policy.json')
        validate_pool(task,record['rows'],30)
        imported=dict(record,**identity('warm30',task))
        imported['reuse_provenance']=dict(campaign=str(OLD),source_complete_sha256=digest(p),measurement_rows_unchanged=True,
            original_recipe_sha256=record['recipe_sha256'],original_policy_sha256=record['policy_sha256'])
        target=ROOT/'warm30/runs'/task/'complete.json'
        if target.exists():assert load(target)==imported
        else:atomic(target,imported)
        progress=load(p.with_name('progress.json'))
        assert progress['rows']==30 and progress['attempts']==record['generation_attempts']
        atomic(target.with_name('progress.json'),dict(progress,reused=30))
        assert load(target)['rows']==record['rows']
        phases.update(r['source_phase'] for r in record['rows']);sources[task]=digest(p)
    freeze()
    from run import status
    imported_status=status()
    assert imported_status['stages']['warm30']['measurements']==6000
    assert imported_status['counted_measurements']==6000
    atomic(ROOT/'evidence/bolt30_reuse_audit.json',dict(passed=True,tasks=200,measurements=6000,all_plans_strictly_valid=True,
        sources=sources,source_phases=dict(phases),plans_runtimes_and_provenance_unchanged=True,old_optimization_histories_reused=False,
        frozen_pool_sha256=digest(ROOT/'pools/batch10_frozen_pool.json'),at=time.time()))
def services():
    import recovery,resources
    command([PY,str(ROOT/'batch_code/resources.py'),'host_prepare','--host','LLMOptimizer2'],'db_local')
    command(['ssh','-o','BatchMode=yes','LLMOptimizer4',shlex.join([PY,str(ROOT/'batch_code/resources.py'),'host_prepare','--host','LLMOptimizer4'])],'db_remote')
    resources.vae_services()
    tunnel_record=ROOT/'services/resources_tunnel.process.json'
    if not tunnel_record.exists() or not resources.record_alive(load(tunnel_record)):resources.tunnel()
    resources.inference_stage('orpo_batch10')
    deadline=time.monotonic()+180
    while True:
        try:resources.health();break
        except Exception:
            if time.monotonic()>deadline:raise
            time.sleep(2)
def migrate_recovery():
    # Keep the existing remote 600-second scheduler and restricted SSH key.
    # Its original entry point dispatches only tick to this authorized rerun.
    oldops=OLD/'operations/recovery';newops=ROOT/'operations/recovery'
    with (oldops/'tick.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        script=oldops/'recovery.py';backup=ROOT/'operations/parent_recovery.before.py'
        if not backup.exists():shutil.copy2(script,backup)
        s=script.read_text()
        marker=" value=tick() if a.action=='tick'"
        if 'active_target.json' not in s:
            assert marker in s
            s=s.replace(marker," if a.action=='tick' and (OPS/'active_target.json').exists():\n  destination=Path(load(OPS/'active_target.json')['recovery_script'])\n  assert destination.is_absolute() and destination.name=='recovery.py' and destination.is_file()\n  os.execv(PYTHON,[PYTHON,'-u',str(destination),'tick'])\n"+marker)
            script.write_text(s)
        atomic(oldops/'active_target.json',dict(campaign=str(ROOT),recovery_script=str(newops/'recovery.py'),reason='Authorized retrained VAE+FSBO rerun',at=time.time()))
        # A new campaign starts its own rate series; do not subtract its count
        # from the previous experiment's accumulated evaluations.
        latest=oldops/'latest.json'
        if latest.exists() and load(latest).get('campaign')!=str(ROOT):
            backup=ROOT/'operations/parent_watchdog_latest.before.json'
            if not backup.exists():shutil.copy2(latest,backup)
            latest.unlink()
    monitor=Path('/root/.codex/loops/query-training-hourly-20261112');mp=monitor/'monitor.py'
    backup=ROOT/'operations/monitor.before.py'
    if not backup.exists():shutil.copy2(mp,backup)
    s=mp.read_text().replace(OLD.name,ROOT.name);mp.write_text(s)
    state=load(monitor/'state.json')
    import resources
    if resources.record_alive(state):
        args=Path(f"/proc/{state['pid']}/cmdline").read_bytes().split(b'\0')
        assert str(mp).encode() in args
        os.kill(state['pid'],signal.SIGTERM)
        deadline=time.monotonic()+30
        while resources.record_alive(state):
            if time.monotonic()>deadline:raise RuntimeError('Old read-only monitor failed to stop')
            time.sleep(.2)
    from recovery import ensure_monitor
    ensure_monitor()
    atomic(ROOT/'operations/recovery_handoff.json',dict(cadence_seconds=600,new_target=str(ROOT),
        retained_scheduler=str(oldops/'watchdog.py'),retained_scheduler_state=str(oldops/'state.json'),new_recovery=str(newops/'recovery.py'),
        new_watchdog_created=False,at=time.time()))
def main():
    lock=(ROOT/'training/promotion.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import resources,recovery
    assert resources.ROOT==ROOT and recovery.ROOT==ROOT, 'Wrong campaign runtime import'
    controller=ROOT/'services/batch_controller.process.json'
    if controller.exists() and resources.record_alive(load(controller)):
        print('Rerun controller already alive; refusing a duplicate launch',flush=True);return
    assert (OLD/'PAUSE').exists() and (ROOT/'PAUSE').exists()
    configurations();import_warm()
    if '--resume-validated' in sys.argv:
        # A bookkeeping-only startup repair may reuse completed checks only
        # while every tested runtime/model/configuration pin still matches.
        recovery.verify_pins()
        for name in ['batch_tests','batch_preflight','batch_context_smoke','resources_remote_smoke','batch_launch_readiness']:
            assert load(ROOT/'evidence'/f'{name}.json')['passed'], name
        preflight=load(ROOT/'evidence/batch_preflight.json')
        assert preflight['recipe_sha256']==digest(ROOT/'batch_recipe.json')
        assert preflight['policy_sha256']==digest(ROOT/'batch_policy.json')
        assert preflight['fsbo_final_fit']['checkpoint_sha256']==digest(ROOT/'training/fsbo/weights')
        assert load(ROOT/'evidence/resources_remote_smoke.json')['fsbo_fit']['checkpoint_sha256']==digest(ROOT/'training/fsbo/weights')
        resources.health()
        print('Reusing completed live checks with unchanged verified pins',flush=True)
    else:
        command([PY,'-m','unittest','test_batch','test_fsbo_batch','test_service_recovery','test_retraining'],'unit_tests',cwd=ROOT/'batch_code')
        command([PY,'-m','unittest','test_recovery'],'recovery_tests',cwd=ROOT/'operations/recovery')
        atomic(ROOT/'evidence/batch_tests.json',dict(passed=True,command='unittest test_batch test_fsbo_batch test_service_recovery; unittest test_recovery',
            logs=['training/promotion.unit_tests.log','training/promotion.recovery_tests.log'],at=time.time()))
        services()
        command([PY,'-u',str(ROOT/'batch_code/preflight.py')],'preflight',timeout=1200)
        command([PY,'-u',str(ROOT/'batch_code/context_smoke.py')],'context',timeout=600)
        command([PY,'-u',str(ROOT/'batch_code/resource_smoke.py')],'remote_smoke',timeout=900)
    command([PY,'-u',str(ROOT/'batch_code/validate_launch.py')],'validate_launch',timeout=600)
    probes=[recovery.probe('LLMOptimizer2'),recovery.host_call('LLMOptimizer4','probe')]
    assert not any(p['unrecorded_workers'] or any(w['alive'] for w in p['workers']) for p in probes)
    assert all(d['ready'] and not d['active'] for p in probes for d in p['databases'])
    migrate_recovery()
    (ROOT/'PAUSE').unlink()
    pid=resources.start([PY,'-u',str(ROOT/'batch_code/run.py'),'controller'],'batch_controller')
    deadline=time.monotonic()+180
    while True:
        assert resources.record_alive(load(controller)), 'Rerun controller exited'
        s=load(ROOT/'status.json')
        if s.get('status')=='running' and s.get('worker_count')==21:break
        if time.monotonic()>deadline:raise RuntimeError('Rerun worker startup timed out')
        time.sleep(2)
    atomic(ROOT/'evidence/retrained_rerun_started.json',dict(passed=True,controller_pid=pid,status=s,
        vae_checkpoint_sha256=digest(ROOT/'training/vae/best.ckpt'),fsbo_weights_sha256=digest(ROOT/'training/fsbo/weights'),at=time.time()))
    print('RERUN STARTED',pid,'workers',s['worker_count'],flush=True)
if __name__=='__main__':main()
