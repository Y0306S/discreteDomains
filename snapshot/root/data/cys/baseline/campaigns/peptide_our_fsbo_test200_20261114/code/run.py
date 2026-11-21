"""Prepare, inspect, and explicitly run individual peptide tasks. No queue command."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path('/root/data/cys/llmoptimizer_benchmark/tools')
TRAIN = Path('/root/data/cys/baseline/campaigns/peptide_our_orpo_rl800_20261111')
WARM = Path('/root/data/cys/baseline/campaigns/peptide_warmstart_full800_20261111')
BASE = Path('/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B')
PARENT = Path('/root/data/logs/peptide_transfer_test200_20261028_v1/fsbo/recipe.json')
sys.path.insert(0, str(TOOLS))
from peptide_prompt import atomic_json, file_sha
from qwen_inference_transport import model_identity, digest
from method import Journal, warm_start, optimize, observed, verify_completed_stage


def prepare():
    if (ROOT/'recipe.json').exists():
        raise FileExistsError('Recipe exists; validate it without replacing its identity')
    import peptide_transfer_eval as transfer
    import peptide_local_policy
    backend = transfer.verify(PARENT.parent)
    warm_status = json.loads((WARM/'training_status.json').read_text())
    if warm_status['status'] != 'completed' or warm_status['step'] != 1500:
        raise ValueError('Warm-start training is incomplete')
    shards = json.loads((WARM/'evidence/final_weights.json').read_text())
    if not shards['passed']:
        raise ValueError('Warm-start weights were not verified at training completion')
    completion = json.loads((TRAIN/'cycles/round1/complete.json').read_text())
    models = {}
    for stage, base, adapter in [('warm_start', WARM/'model/checkpoint-1500', None),
                                  ('orpo', BASE, TRAIN/'models/orpo'),
                                  ('orpo_rl', BASE, TRAIN/'cycles/round1')]:
        models[stage] = dict(identity=model_identity(base, adapter), checkpoint=str(adapter or base))
    if models['orpo_rl']['identity']['adapter_sha256'] != completion['adapter_sha256']:
        raise ValueError('RL adapter differs from completed cycle')
    orpo_status = json.loads((TRAIN/'orpo_status.json').read_text())
    if (orpo_status['status'] != 'completed' or
            models['orpo']['identity']['adapter_sha256'] != orpo_status['adapter_sha256']):
        raise ValueError('ORPO adapter differs from completed training')
    models['warm_start']['weight_shards_sha256'] = shards['shards']
    models['warm_start']['weight_shard_stat'] = {
        name:dict(size=(WARM/'model/checkpoint-1500'/name).stat().st_size,
                  mtime_ns=(WARM/'model/checkpoint-1500'/name).stat().st_mtime_ns)
        for name in shards['shards']}
    recipe = dict(format='peptide-our-warm50-fsbo-v1', name=ROOT.name, prepared_at=time.time(),
                  execution='prepared_only_not_queued', queue_armed=False, seed=1,
                  tasks=backend['test'], train_tasks=backend['train'],
                  references={t:backend['references'][t] for t in backend['test']},
                  stages=['warm_start','orpo','orpo_rl'], models=models,
                  constraints=dict(alphabet='ACDEFGHIKLMNPQRSTVWY', maximum_length=30,
                     minimum_length=1, maximum_cysteine=1, minimum_reference_similarity=.75,
                     similarity='1-Levenshtein(candidate,reference)/len(reference)', unchanged_reference_allowed=False),
                  sampling=dict(temperature=1., top_p=.9, top_k=50, max_new_tokens=128),
                  warm_start=dict(count=50, maximum_samples=500, output='one_sequence_per_response',
                      acceptance='first50_feasible_unique_in_generation_order', random_fallback=False,
                      underfilled='stop_and_report_incomplete', score='real_CPU_APEX_bacteria_0',
                      freeze='one_shared_measured_pool_for_both_optimization_stages'),
                  optimization=dict(attempts=80, selected_candidates_per_attempt=1,
                      initial_llm_attempts=8, llm_generation_limit=5, duplicate_or_invalid_fallback='FSBO',
                      fsbo_decode_limit=5, exhausted_fsbo='NO_VALID_PROPOSAL_consumes_attempt_no_fabricated_score',
                      random_fallback=False, infrastructure_failure='pause_same_task; no_attempt_consumed',
                      selector='PUCB_arm_best', pucb_c2=1.4, llm_prior_start=.9, llm_prior_end=.3,
                      fsbo_prior_start=.3, fsbo_prior_end=.9, prior_weights_normalized=False,
                      schedule_budget=100, schedule_index='attempt-9', schedule_denominator=91,
                      cooldown='attempts10..20: FSBO if previous selected_arm was LLM, including fallback',
                      warm_starts_update_selector=False,
                      orpo_prompt='exact_ORPO_training_peptide_prompt: good5_bad5_recent50',
                      rl_prompt='training-compatible retrieval attempts1..10; normal afterwards; no region summaries'),
                  fsbo=dict(parent_recipe=str(PARENT), parent_recipe_sha256=file_sha(PARENT),
                      backend_recipe=backend, checkpoint=backend['fsbo_weights'],
                      checkpoint_sha256=backend['fsbo_weights_sha256'], target_scaling='minmax_of_negative_MIC',
                      maximum_target_updates=backend['fsbo_target_epochs'], target_patience=16,
                      min_delta=.0001, target_learning_rate=.001, candidate_pool=5000,
                      acquisition='native_constrained_Thompson_sampling; single_candidate',
                      vae='frozen_author; canonical256D_posterior_mean',
                      local=dict(peptide_local_policy.DEFAULTS),
                      region_policy='reset_local_region_at_current_best_each_FSBO_selection',
                      recenter=False, vae_training=False, constraint_fit='20_epochs_all_observations_each_FSBO_selection'),
                  comparison=dict(same_pool=True, second_stage_continues_first_history=False,
                      second_stage_waits_for='all200_ORPO_tasks_committed_at80_attempts',
                      initialization_measurements=10000, optimization_attempts_each_stage=16000,
                      maximum_total_measurements=42000),
                  launch=dict(armed=False, automatic_handoff=False, reserved_gpus=0,
                      reserved_cpus=0, cpu_threads_per_task=1, inference_endpoints='assigned_only_at_manual_launch',
                      pending_live_preflight=['warm_model_shard_rehash','real_LLM_VAE_APEX_FSBO_smoke_on_train_task',
                                              'model_service_memory_and_prompt_cache_parity']),
                  differences_from_historical_peptide_FSBO=['50 new BOLT-generated initializations instead of100 mutation-pool initializations',
                    '80 single-candidate hybrid attempts instead of5000 batch50 proposals',
                    'Fixed local region centered at current best; no native recenter/e2e; separately named method'],
                  input_files={})
    assert len(set(recipe['tasks'])) == 200 and len(set(recipe['train_tasks'])) == 800
    assert not set(recipe['tasks']) & set(recipe['train_tasks'])
    files = [PARENT, PARENT.with_name('target_fit_amendment.json'),
             WARM/'evidence/final_weights.json', TRAIN/'assets/source_cards.json',
             TRAIN/'assets/source_gp_inventory.json', TRAIN/'cycles/round1/complete.json',
             Path(backend['corpus']), TOOLS/'bolt_peptide_sft.py']
    files += [Path(p) for p in backend['code']]
    files += list((ROOT/'code').glob('*.py'))
    recipe['input_files'] = {str(p):file_sha(p) for p in sorted(set(files))}
    atomic_json(ROOT/'recipe.json', recipe)
    atomic_json(ROOT/'status.json', dict(status='prepared_not_queued', queue_armed=False,
        started_tasks=0, started_workers=0, generated_initializations=0, oracle_calls=0,
        stages=recipe['stages'], recipe_sha256=file_sha(ROOT/'recipe.json')))
    for stage in recipe['stages']:
        atomic_json(ROOT/'stages'/f'{stage}.json', dict(stage=stage, queued=False,
            tasks=200, model=models[stage],
            attempts_per_task=50 if stage=='warm_start' else 80,
            prerequisite={'warm_start':'manual_start', 'orpo':'frozen_all200_shared_pool',
                          'orpo_rl':'all200_orpo_tasks_complete_at80'}[stage]))
        service = dict(name=ROOT.name+'_'+stage, base=models[stage]['identity']['base'],
            adapter=models[stage]['identity']['adapter'], identities=[models[stage]['identity']],
            gpu=None, cpus=[], port=None, memory_gib=21, max_input_tokens=15360, max_new_tokens=128,
            prompt_cache=dict(enabled=True, max_bytes=1610612736, min_tokens=128),
            status='template_only_no_gpu_or_port_assigned')
        atomic_json(ROOT/'service_templates'/f'{stage}.json', service)
    return validate()


def validate(check_weights=False):
    recipe = json.loads((ROOT/'recipe.json').read_text())
    if recipe['queue_armed'] or recipe['launch']['armed']:
        raise ValueError('This setup is intended to remain unqueued')
    for name, expected in recipe['input_files'].items():
        if file_sha(name) != expected:
            raise ValueError('Pinned dependency changed: '+name)
    for stage, model in recipe['models'].items():
        identity = model['identity']
        if model_identity(identity['base'], identity['adapter']) != identity:
            raise ValueError('Model identity changed: '+stage)
    for name, expected in recipe['models']['warm_start']['weight_shards_sha256'].items():
        p=Path(recipe['models']['warm_start']['checkpoint'])/name
        stat=p.stat(); saved=recipe['models']['warm_start']['weight_shard_stat'][name]
        if stat.st_size != saved['size'] or stat.st_mtime_ns != saved['mtime_ns']:
            raise ValueError('Warm checkpoint shard changed: '+name)
        if check_weights and file_sha(p) != expected:
            raise ValueError('Warm weight shard hash mismatch: '+name)
    assert len(recipe['tasks']) == 200 and len(recipe['train_tasks']) == 800
    assert not set(recipe['tasks']) & set(recipe['train_tasks'])
    report=dict(passed=True, recipe_sha256=file_sha(ROOT/'recipe.json'), tasks=200,
                initialization_count=50, optimization_attempts_per_stage=80,
                queue_armed=False, production_tasks_started=0,
                warm_weights='rehash_passed' if check_weights else 'training_verified_hashes_pinned; current_size_and_mtime_match',
                real_end_to_end_preflight='not_run; no_GPU_services_or_oracle_calls_started', at=time.time())
    atomic_json(ROOT/'evidence/config_validation.json', report)
    if check_weights:
        atomic_json(ROOT/'evidence/warm_weight_rehash.json', report)
    return report


def freeze_pool(recipe):
    key=file_sha(ROOT/'recipe.json'); pools={}
    for task in recipe['tasks']:
        path=ROOT/'warm_start/runs'/task/'complete.json'
        value=json.loads(path.read_text())
        if value['recipe_sha256'] != key or value['task'] != task or value['stage'] != 'warm_start':
            raise ValueError('Warm pool identity mismatch: '+task)
        rows=value['rows']
        if len(rows)!=50 or len({r['sequence'] for r in rows})!=50:
            raise ValueError('Incomplete/nonunique initial pool: '+task)
        for row in rows:
            observed(row['sequence'],row['mic'],recipe['references'][task],row['observation_index'],'initialization')
        pools[task]=rows
    result=dict(format='frozen_BOLT_initial50_v1',recipe_sha256=key,tasks=pools,
                observations=10000,shared_across=['orpo','orpo_rl'])
    path=ROOT/'pools/frozen_pool.json'
    if path.exists() and json.loads(path.read_text())!=result:
        raise ValueError('Frozen pool may not be replaced')
    atomic_json(path,result)
    atomic_json(ROOT/'pools/frozen_pool.identity.json',dict(sha256=file_sha(path),recipe_sha256=key))
    return dict(frozen=True,tasks=200,measurements=10000,sha256=file_sha(path))


def run_task(args, recipe):
    if not args.execute:
        raise ValueError('Individual execution requires --execute; preparation never runs tasks')
    if args.task not in recipe['tasks']:
        raise ValueError('Task outside heldout200')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Task workers require CUDA_VISIBLE_DEVICES=""; GPU services are separate')
    key=file_sha(ROOT/'recipe.json');pool=None; pool_sha=None
    if args.stage!='warm_start':
        path=ROOT/'pools/frozen_pool.json';record=json.loads(path.read_text())
        pool_sha=file_sha(path)
        if (record['recipe_sha256']!=key or set(record['tasks'])!=set(recipe['tasks'])
                or json.loads((ROOT/'pools/frozen_pool.identity.json').read_text())['sha256']!=pool_sha):
            raise ValueError('Frozen pool changed')
        pool=record['tasks'][args.task]
    if args.stage=='orpo_rl':
        verify_completed_stage(ROOT,'orpo',key,recipe['tasks'],80)
    evidence=ROOT/'evidence/warm_weight_rehash.json'
    if not evidence.exists() or json.loads(evidence.read_text())['recipe_sha256']!=key:
        raise RuntimeError('Run validate --weights before loading the warm-start checkpoint')
    os.sched_setaffinity(0,{args.cpu})
    directory=ROOT/args.stage/'runs'/args.task
    directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'task.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity=dict(recipe_sha256=key,stage=args.stage,task=args.task,pool_sha256=pool_sha)
    journal=Journal(directory/'runtime.sqlite',identity)
    fsbo=None
    try:
        from backends import LLM,apex,FSBO,Retrieval
        generate=LLM(recipe,args.stage,args.llm_url)
        # Lazy oracle initialization: all invalid/duplicate generations remain unscored.
        evaluate=lambda sequence:apex()(sequence)
        if args.stage=='warm_start':
            state=warm_start(journal,args.task,recipe['references'][args.task],recipe,generate,evaluate)
            result=dict(identity,attempts=50,rows=state['rows'],generation_attempts=state['attempts'],
                        rejected=state['rejected'],status='complete')
        else:
            if not args.vae_url:
                raise ValueError('FSBO requires the dedicated frozen-VAE service URL')
            fsbo=FSBO(recipe,args.task,args.vae_url,key)
            retrieval=Retrieval(recipe,args.task,fsbo.client)
            state=optimize(journal,args.stage,args.task,recipe['references'][args.task],recipe,pool,
                           generate,evaluate,fsbo,retrieval)
            result=dict(identity,attempts=len(state['events']),rows=state['rows'],events=state['events'],
                        initial_best_mic=min(r['mic'] for r in pool),
                        best_mic=min(r['mic'] for r in state['rows']),status='complete')
        atomic_json(directory/'complete.json',result)
        return dict(stage=args.stage,task=args.task,status='complete',attempts=result['attempts'])
    finally:
        journal.close()
        if fsbo is not None:fsbo.close()
        lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare')
    check=sub.add_parser('validate');check.add_argument('--weights',action='store_true')
    sub.add_parser('show-config');sub.add_parser('freeze-pool')
    task=sub.add_parser('run-task')
    task.add_argument('--stage',choices=['warm_start','orpo','orpo_rl'],required=True)
    task.add_argument('--task',required=True);task.add_argument('--llm-url',required=True)
    task.add_argument('--vae-url');task.add_argument('--cpu',type=int,required=True)
    task.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':result=prepare()
    elif args.command=='validate':result=validate(args.weights)
    else:
        recipe=json.loads((ROOT/'recipe.json').read_text())
        if args.command=='show-config':result=recipe
        else:
            validate()
            result=freeze_pool(recipe) if args.command=='freeze-pool' else run_task(args,recipe)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
