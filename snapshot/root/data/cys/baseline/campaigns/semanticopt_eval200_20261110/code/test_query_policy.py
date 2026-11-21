import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import common,evaluate
from batch import batch_messages
from protocol import atomic,canonical

class QueryPolicyChecks(unittest.TestCase):
    def test_query_only_resolution_and_completion(self):
        recipe=common.verify();before=copy.deepcopy(recipe)
        q=common.domain_recipe(recipe,'query');p=common.domain_recipe(recipe,'peptide')
        self.assertEqual((q['batch_policy']['size'],q['query']['rounds'],q['query']['proposals']),(10,20,200))
        self.assertEqual((p['batch_policy']['size'],p['peptide']['rounds'],p['peptide']['proposals']),(50,20,1000))
        self.assertEqual(p['batch_policy_sha256'],common.digest(common.ROOT/'batch_policy.json'))
        self.assertEqual(p['budget_policy_sha256'],common.digest(common.ROOT/'budget_policy.json'))
        self.assertEqual(q['batch_policy_sha256'],recipe['query_policy_sha256'])
        self.assertEqual(recipe,before)
        q['query']=dict(q['query'],test=['q'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);atomic(root/'recipe.json',{})
            status=dict(status='complete',evaluations=200,recipe_sha256=common.digest(root/'recipe.json'),
                batch_policy_sha256=q['batch_policy_sha256'],budget_policy_sha256=q['budget_policy_sha256'],
                runtime_policy_sha256=q['runtime_policy_sha256'])
            atomic(root/'query/runs/q/status.json',status)
            with patch.object(common,'ROOT',root):self.assertTrue(common.all_complete('query',q))
    def test_progress_reports_query_policy_while_queued(self):
        from query_progress import update_progress
        recipe=common.verify();recipe['query']=dict(recipe['query'],test=['q'])
        with tempfile.TemporaryDirectory() as tmp:
            progress=update_progress(Path(tmp),recipe,now=100,force=True)
            self.assertEqual((progress['batch_size'],progress['rounds'],progress['proposals_per_task']),(10,20,200))
            self.assertEqual(progress['status'],'queued')
            self.assertIsNone(progress['windows']['60']['remaining_hours'])

    def test_resume_partial_round_to_20_rounds_preserves_history_and_bao(self):
        recipe=common.domain_recipe(common.verify(),'query');recipe['query']=dict(recipe['query'],test=['q'])
        initial=[dict(configuration={'plan':[i]},objective=10.,status='success',censored=False,feasible=True) for i in range(49)]
        history=[dict(configuration={'plan':[i+100]},objective=9.,status='success',censored=False,feasible=True) for i in range(15)]
        generations=[];calls=[]
        class Client:
            def __init__(self,*a):self.state={};self.last_response=None
            def import_state(self,state):self.state=state or {}
            def ensure_loaded(self):pass
            def complete_messages(self,messages,**kw):
                prompt=json.loads(messages[1]['content']);count=prompt['requested_configurations'];start=prompt['completed_proposals']
                generations.append(count)
                self_test.assertEqual(prompt['initial_observation_count'],49)
                self_test.assertEqual(len(prompt['bao_initial_observations']),49)
                self_test.assertEqual(prompt['batch_size'],10)
                self_test.assertEqual(prompt['proposal_budget'],200)
                return canonical({str(i+1):[start+i+100] for i in range(count)})
        class Tokenizer:
            def apply_chat_template(self,*a,**kw):return 'prompt'
            def __call__(self,*a,**kw):return {'input_ids':[1]}
        def oracle(config):
            calls.append(config);return dict(configuration=config,objective=8.,status='success',censored=False,feasible=True)
        self_test=self
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);atomic(root/'recipe.json',{})
            folder=root/'query/runs/q';db=evaluate.connect(folder)
            with db:
                for i,row in enumerate(history):db.execute('INSERT INTO records VALUES(?,?)',(i,canonical(row)))
                evaluate.put(db,'initial',initial)
            before=db.execute('SELECT * FROM records ORDER BY n').fetchall();db.close()
            with patch.object(evaluate,'ROOT',root),patch.object(evaluate,'verify',return_value=recipe),\
                 patch.object(evaluate,'query_setup',return_value=({'task_id':'q'},initial,lambda c:None,oracle)),\
                 patch('qwen_inference_transport.SharedQwenClient',Client),\
                 patch('transformers.AutoTokenizer.from_pretrained',return_value=Tokenizer()):
                real_status=evaluate.status
                def interrupt(path,**values):
                    real_status(path,**values)
                    if values.get('evaluations')==20:raise InterruptedError('test resume')
                with patch.object(evaluate,'status',interrupt):
                    with self.assertRaises(InterruptedError):evaluate.run('query','q',0)
                evaluate.run('query','q',0);evaluate.run('query','q',0)
            self.assertEqual(generations,[5]+[10]*18)
            self.assertEqual(len(calls),185)
            self.assertEqual(len({canonical(c) for c in calls}),185)
            db=evaluate.connect(folder)
            self.assertEqual(db.execute('SELECT * FROM records WHERE n<15 ORDER BY n').fetchall(),before)
            self.assertEqual(evaluate.get(db,'initial'),initial);db.close()
            status=common.load(folder/'status.json')
            self.assertEqual((status['evaluations'],status['completed_rounds'],status['batch_size']),(200,20,10))

if __name__=='__main__':unittest.main()
