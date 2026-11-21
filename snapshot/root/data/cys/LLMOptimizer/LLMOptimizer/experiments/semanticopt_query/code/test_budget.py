import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import common
import evaluate
from evaluate import connect,get,put
from peptide_batch import score_and_commit
from protocol import atomic,digest

class BudgetChecks(unittest.TestCase):
    def setUp(self):
        self.recipe=json.loads((common.ROOT/'recipe.json').read_text())
        self.recipe['batch_policy']={'size':50}
        self.policy=dict(base_recipe_sha256=digest(common.ROOT/'recipe.json'),batch_size=50,rounds=dict(peptide=20,query=20))
    def test_reduced_budget_preserves_original_and_query(self):
        before=copy.deepcopy(self.recipe)
        result=common.apply_budget_policy(self.recipe,self.policy,'budget')
        self.assertEqual(result['peptide']['proposals'],1000)
        self.assertEqual(result['query']['proposals'],1000)
        self.assertEqual(result['peptide']['original_proposals'],5000)
        self.assertEqual(result['batch_policy'],before['batch_policy'])
        self.assertEqual(result['peptide']['test'],before['peptide']['test'])
        self.assertEqual(result['peptide']['adapter_sha256'],before['peptide']['adapter_sha256'])
    def test_invalid_policy_rejected(self):
        for rounds in [0,-1,True,20.5,101]:
            p=copy.deepcopy(self.policy);p['rounds']['peptide']=rounds
            with self.assertRaises(ValueError):common.apply_budget_policy(copy.deepcopy(self.recipe),p,'budget')
        for change in [dict(batch_size=10),dict(base_recipe_sha256='wrong'),dict(rounds={'peptide':20})]:
            with self.assertRaises(ValueError):common.apply_budget_policy(copy.deepcopy(self.recipe),dict(self.policy,**change),'budget')
    def test_completion_accepts_1000_only_with_matching_policy(self):
        recipe=common.apply_budget_policy(self.recipe,self.policy,'budget');recipe['peptide']['test']=['p']
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);atomic(root/'recipe.json',{})
            data=dict(status='complete',evaluations=1000,recipe_sha256=digest(root/'recipe.json'),
                adapter_sha256=recipe['peptide']['adapter_sha256'],batch_policy_sha256=None,
                runtime_policy_sha256=None,budget_policy_sha256='budget')
            path=root/'peptide/runs/p/status.json';atomic(path,data)
            with patch.object(common,'ROOT',root):
                self.assertTrue(common.all_complete('peptide',recipe))
                atomic(path,dict(data,budget_policy_sha256=None));self.assertFalse(common.all_complete('peptide',recipe))
                atomic(path,dict(data,evaluations=999));self.assertFalse(common.all_complete('peptide',recipe))
                atomic(path,dict(data,evaluations=5000));self.assertFalse(common.all_complete('peptide',recipe))
    def test_saved_queue_cannot_overshoot_new_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            db=connect(Path(tmp))
            with db:
                db.executemany('insert into records values (?,?)',[(i,json.dumps({'evaluation':i+1})) for i in range(995)])
            calls=[]
            def oracle(configs):
                calls.append(len(configs));return [dict(configuration=c,objective=1.) for c in configs]
            items=[dict(answer=dict(next_configuration={'sequence':str(i)}),provenance={'source':'llm'}) for i in range(50)]
            result=score_and_commit(db,get,put,batch=dict(start=950,items=items,policy_sha256='batch'),
                start=995,total=1000,validate=lambda c:None,oracle=oracle,seen=set(),runtime_sha='r',policy_sha='p')
            self.assertEqual(calls,[5]);self.assertEqual(result[-1]['evaluation'],1000)
            self.assertEqual(db.execute('select count(*) from records').fetchone()[0],1000)
            with self.assertRaises(ValueError):
                score_and_commit(db,get,put,batch=dict(start=950,items=items,policy_sha256='batch'),
                    start=1000,total=1000,validate=lambda c:None,oracle=oracle,seen=set(),runtime_sha='r',policy_sha='p')
            self.assertEqual(calls,[5]);db.close()
    def test_resumed_worker_stops_at_round20_without_regeneration(self):
        recipe=common.apply_budget_policy(self.recipe,self.policy,'budget')
        recipe['peptide']['test']=['p']
        recipe.update(runtime_policy_sha256='runtime',batch_policy_sha256='batch',
            peptide_oracle_policy_sha256='oracle',peptide_oracle_policy={'enabled':True})
        initial=[dict(configuration={'sequence':'initial'},objective=10.,status='success',censored=False)]
        calls=[]
        def oracle(configs):
            calls.append(len(configs))
            return [dict(configuration=c,objective=1.,status='success',censored=False) for c in configs]
        def single(c):raise AssertionError('Expected batched oracle')
        single.batch=oracle
        class Client:
            def __init__(self,*args):self.state={}
            def import_state(self,state):pass
            def ensure_loaded(self):pass
            def complete_messages(self,*args,**kwargs):raise AssertionError('Saved batch must not regenerate')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);atomic(root/'recipe.json',{})
            folder=root/'peptide/runs/p';db=connect(folder)
            with db:
                for i in range(950):
                    db.execute('insert into records values (?,?)',(i,json.dumps(dict(configuration={'sequence':str(i)},objective=2.,status='success',censored=False))))
                put(db,'proposal_batch',dict(start=950,policy_sha256='batch',items=[
                    dict(answer=dict(next_configuration={'sequence':str(i)}),provenance={'source':'llm'}) for i in range(950,1000)]))
            db.close()
            with patch.object(evaluate,'ROOT',root),patch.object(evaluate,'verify',return_value=recipe),\
                 patch.object(evaluate,'peptide_setup',return_value=({'task_id':'p'},initial,lambda c:None,single)),\
                 patch('qwen_inference_transport.SharedQwenClient',Client),\
                 patch('transformers.AutoTokenizer.from_pretrained',return_value=object()):
                evaluate.run('peptide','p',0);evaluate.run('peptide','p',0)
            status=json.loads((folder/'status.json').read_text())
            self.assertEqual(calls,[50]);self.assertEqual(status['status'],'complete')
            self.assertEqual(status['evaluations'],1000);self.assertEqual(status['budget'],1000)
            self.assertEqual(status['completed_rounds'],20);self.assertEqual(status['budget_policy_sha256'],'budget')

if __name__=='__main__':unittest.main()
