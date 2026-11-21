import json,tempfile,unittest
from pathlib import Path
from evaluate import connect,get,put
from peptide_batch import score_and_commit
from protocol import canonical

class OracleBatchChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.db=connect(Path(self.temp.name));self.calls=[]
        self.items=[dict(answer=dict(next_configuration=dict(sequence=str(i))),provenance=dict(source='llm')) for i in range(50)]
    def tearDown(self):self.db.close();self.temp.cleanup()
    def oracle(self,configs):
        self.calls.append(configs)
        return [dict(configuration=c,objective=float(i),status='success',censored=False,feasible=True) for i,c in enumerate(configs)]
    def invoke(self,**kwargs):
        args=dict(batch=dict(start=0,items=self.items,policy_sha256='b'),start=0,total=5000,
            validate=lambda c:None,oracle=self.oracle,seen=set(),runtime_sha='r',policy_sha='p')
        args.update(kwargs);return score_and_commit(self.db,get,put,**args)
    def test_all_50_one_call_one_commit(self):
        results=self.invoke();self.assertEqual([len(x) for x in self.calls],[50]);self.assertEqual(len(results),50)
        self.assertEqual(self.db.execute('select count(*) from records').fetchone()[0],50)
        self.assertEqual(get(self.db,'oracle_batch')['status'],'committed')
        self.assertEqual(results[-1]['evaluation'],50)
        with self.assertRaisesRegex(ValueError,'cursor mismatch'):self.invoke()
        self.assertEqual(len(self.calls),1)
    def test_duplicate_never_scored(self):
        self.items[-1]=self.items[0]
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.invoke()
        self.assertEqual(self.calls,[]);self.assertIsNone(get(self.db,'oracle_batch'))
    def test_inflight_blocks_replay(self):
        def interrupted(configs):self.calls.append(configs);raise InterruptedError('power loss')
        with self.assertRaises(InterruptedError):self.invoke(oracle=interrupted)
        with self.assertRaisesRegex(RuntimeError,'requires audit'):self.invoke()
        self.assertEqual(len(self.calls),1);self.assertEqual(self.db.execute('select count(*) from records').fetchone()[0],0)
    def test_commit_failure_rolls_back_all_results(self):
        def failing_put(db,key,value):
            if key=='oracle_batch' and value['status']=='committed':raise InterruptedError('commit interrupted')
            put(db,key,value)
        args=dict(batch=dict(start=0,items=self.items,policy_sha256='b'),start=0,total=5000,
            validate=lambda c:None,oracle=self.oracle,seen=set(),runtime_sha='r',policy_sha='p')
        with self.assertRaises(InterruptedError):score_and_commit(self.db,get,failing_put,**args)
        self.assertEqual(self.db.execute('select count(*) from records').fetchone()[0],0)
        self.assertEqual(get(self.db,'oracle_batch')['status'],'in_flight')
    def test_wrong_order_and_nonfinite_scores_rejected(self):
        def wrong(configs):
            v=self.oracle(configs);v[0]['objective']=float('nan');return v
        with self.assertRaisesRegex(ValueError,'order/value'):self.invoke(oracle=wrong)
        self.assertEqual(self.db.execute('select count(*) from records').fetchone()[0],0)
    def test_budget_tail(self):
        results=self.invoke(total=3);self.assertEqual(len(results),3);self.assertEqual(len(self.calls[0]),3)

if __name__=='__main__':unittest.main()
