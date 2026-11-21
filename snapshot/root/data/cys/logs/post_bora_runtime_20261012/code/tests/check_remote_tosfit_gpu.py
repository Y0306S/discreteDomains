"""Short component check using real Qwen weights; never writes benchmark results."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mindware.components.optimizers.runtime_rpc import server
from mindware.components.optimizers.tosfit_remote import DurableToSFiTActor, RemoteToSFiTRuntime
from mindware.components.optimizers.tosfit_runtime import ToSFiTRuntime, published_recipe


def weights_digest(runtime):
    import torch
    digest = hashlib.sha256()
    for name, tensor in runtime.model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    recipe = published_recipe()
    recipe.update(batch_size=2, max_new_tokens=16, n_marginal_likelihood_warmup_steps=2)
    factory = lambda: ToSFiTRuntime('cuda:0', 'cuda:1', recipe=recipe)
    actor = DurableToSFiTActor(args.out, 'qwen9b-component-check', factory,
                              checkpoint_lock=args.out/'checkpoint.lock')
    httpd = server(actor, 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    config = {'url':'http://127.0.0.1:%d'%httpd.server_address[1],
              'identity':'qwen9b-component-check','timeout_seconds':900}
    session = args.out/'task'
    client = RemoteToSFiTRuntime(config, session, 'component-check', 19)
    prompt = ('Recommend a random forest configuration as JSON. The search space is '
              'n_estimators: integer 100 to 500; max_depth: integer 2 to 12. '
              'The objective is to maximize validation accuracy.')
    records = []
    unload_allocated = []
    restored = False
    try:
        for step, reward in enumerate((-3., -1., -2., 0., .5)):
            started = time.monotonic()
            sample = client.propose(prompt, step)
            client.observe(sample['features'], reward)
            if sample['gradient_step'] != (step >= 2):
                raise AssertionError('Wrong warmup/update boundary')
            records.append({'step':step, 'policy_version':sample['policy_version'],
                            'seconds':time.monotonic()-started, 'gpu_peak_reserved_gib':client.gpu_peaks})
            print(json.dumps(records[-1]), flush=True)
            if step==2:
                checkpoint = client.snapshot(session)
                before = weights_digest(actor.runtime)
                client.detach()
                import torch
                unload_allocated = [torch.cuda.memory_allocated(i) for i in range(2)]
                if max(unload_allocated) > 128 * 2**20:
                    raise AssertionError('Detached slot retained more than 128 MiB of tensors')
                client = RemoteToSFiTRuntime(config, session, 'component-check', 19)
                client.restore(checkpoint)
                with actor.lock:
                    actor._load()
                if weights_digest(actor.runtime) != before:
                    raise AssertionError('Full Qwen weight reload differs')
                restored = True
        if client.policy_version != 3:
            raise AssertionError('Expected three gradient rounds')
        gp_observations = actor.runtime.gp.n_observations
        client.weights_retired = True
        client.snapshot(session)
        client.detach()
        report = {'passed':True, 'base':'Qwen3.5-9B', 'diagnostic_recipe':recipe,
                  'production_recipe_unchanged':True, 'full_weight_hash_reload_verified':restored,
                  'gradient_rounds':3, 'gp_observations':gp_observations, 'rounds':records,
                  'unload_allocated_bytes':unload_allocated,
                  'ml_evaluator_calls':0, 'gpt_calls':0}
        (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()
        actor._drop()


if __name__=='__main__':
    main()
