"""Add one resource slot without altering the recipe/checkpoint identity."""
import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('existing_bolt_controller',ROOT/'control/run.py')
controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
original_load=controller.load
def load(path,default=None):
    cfg=original_load(path,default)
    if Path(path)==ROOT/'recipe.json':
        cfg['workers']=cfg['workers']+[original_load(ROOT/'control/extra_worker.json')]
    return cfg
controller.load=load
if __name__=='__main__':
    ready=json.loads((ROOT/'evidence/worker07/database.ready.json').read_text())
    assert ready['ready'] and ready['spec']['slot']==7
    controller.slot(7)
