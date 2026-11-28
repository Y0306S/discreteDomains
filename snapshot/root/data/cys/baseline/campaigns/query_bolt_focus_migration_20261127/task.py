"""Operational placement wrapper; scientific recipe and runner stay unchanged."""
import argparse
import sys
from common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('task');args=p.parse_args()
    policy=load(Path(os.environ.get('BOLT_PLACEMENT',str(ROOT/'placement.json'))))
    if sha(BOLT/'recipe.json')!=policy['recipe_sha256']:
        raise RuntimeError('Scientific BOLT recipe changed')
    sys.path.insert(0,str(BOLT));sys.path.insert(0,str(OPS))
    import run_lolbo
    original=run_lolbo.QueryVAEBackend
    endpoints=policy['vae_endpoints']
    if len(endpoints)!=16 or len(set(endpoints))!=16:
        raise RuntimeError('Expected sixteen independent forwarded VAE endpoints')

    class PlacedVAE(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            # Replace the old eight endpoints, so no fallback URL can route via Optimizer2.
            self.pool.endpoints=endpoints.copy()
    run_lolbo.QueryVAEBackend=PlacedVAE
    from bolt_proposal_recovery import install
    install(run_lolbo,args.task,OPS)
    from random_feasible_fallback import install as install_random_fallback
    install_random_fallback(run_lolbo,args.task,ROOT)
    run_lolbo.run_task(args.task)


if __name__=='__main__':main()
