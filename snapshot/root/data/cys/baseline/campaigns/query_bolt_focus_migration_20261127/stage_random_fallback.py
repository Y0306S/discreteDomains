"""Stage complete validated proposal batches without running any SQL."""
import fcntl
import sys
from common import *
from prepare_random_fallback import OUT,TASKS,measurements
from random_feasible_fallback import next_batch

def main():
    require_host('LLMOptimizer5');os.sched_setaffinity(0,{60})
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0,str(BOLT));sys.path.insert(0,str(OPS));import run_lolbo as runner
    runner.validate_pins()
    config_path=ROOT/'random_feasible_fallback_20261128.json';config=load(config_path)
    baseline=load(OUT/'baseline.json');placement=load(ROOT/'placement_cpu5.json');results={}
    for task in TASKS:
        folder=BOLT/'runs'/task
        handle=(folder/'task.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert (BOLT/'task_pauses'/f'{task}.json').exists()
        journal=runner.Journal(folder/'runtime.sqlite',runner.identity(task))
        try:
            assert measurements(journal.db)==baseline['tasks'][task]['completed_measurements']
            state=journal.get('lolbo_state');number=len(state['rounds'])+1
            policy=dict(config['tasks'][task],policy_path=str(config_path),
                policy_sha256=sha(config_path),max_random_draws=config['max_random_draws'])
            vae=runner.QueryVAEBackend(runner.recipe()['vae_pool'],task,runner.valid_fallback(task,state['rows']),
                seed=runner.recipe()['seed'])
            vae.pool.endpoints=placement['vae_endpoints'].copy()
            batch=journal.once(f'proposal/{number}',dict(task=task,round=number,batch_size=10),
                lambda:next_batch(runner,task,state,vae,number,policy))
            keys={runner.plan_key(row['plan']) for row in state['rows']}
            assert len(batch['items'])==10
            for item in batch['items']:
                runner.canonicalize(task,item['plan'],strict=True)
                assert runner.plan_key(item['plan']) not in keys
                keys.add(runner.plan_key(item['plan']))
                assert item['proposal_source']=='random_feasible_fallback'
            assert measurements(journal.db)==baseline['tasks'][task]['completed_measurements']
            assert len(journal.get('lolbo_state')['rows'])==baseline['tasks'][task]['saved_rows']
            results[task]=dict(round=number,staged_random_plans=10,
                random_draws=batch['fit']['random_draws'],measurements_unchanged=True,at=time.time())
            print(json.dumps(dict(task=task,**results[task])),flush=True)
        finally:journal.close();handle.close()
    atomic(OUT/'staged.json',dict(tasks=results,policy_sha256=sha(config_path),sql_calls=0,at=time.time()))

if __name__=='__main__':main()
