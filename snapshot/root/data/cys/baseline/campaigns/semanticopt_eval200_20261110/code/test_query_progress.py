import json,sqlite3,tempfile,unittest
from pathlib import Path
from query_progress import update_progress

class QueryProgressChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.recipe=dict(query={'test':['q1','q2'],'proposals':1000},batch_policy={'size':50},budget_policy_sha256='budget')
        self.directory=self.root/'query/runs/q1';self.directory.mkdir(parents=True)
        self.path=self.directory/'observations.sqlite'
        with sqlite3.connect(self.path) as c:c.execute('CREATE TABLE records(n INTEGER PRIMARY KEY,value TEXT)')
    def tearDown(self):self.temp.cleanup()
    def add(self,start,count,at):
        with sqlite3.connect(self.path) as c:
            c.executemany('INSERT INTO records VALUES(?,?)',[(i,json.dumps(dict(at=at,oracle={'elapsed_seconds':2},proposal_source='llm',status='success'))) for i in range(start,start+count)])
    def test_queued_excludes_historical_timings(self):
        self.add(0,15,10)
        v=update_progress(self.root,self.recipe,now=100)
        self.assertEqual(v['status'],'queued');self.assertEqual(v['evaluations'],15)
        self.assertEqual(v['last_hour_commits'],0);self.assertIsNone(v['windows']['60']['remaining_hours'])
    def test_measured_rate_uses_commits_after_activation_and_not_repeated_reads(self):
        self.add(0,15,10);update_progress(self.root,self.recipe,now=100)
        update_progress(self.root,self.recipe,now=200,activate=True)
        self.add(15,50,400)
        v=update_progress(self.root,self.recipe,now=800,activate=True)
        self.assertEqual(v['last_hour_commits'],50);self.assertEqual(v['windows']['60']['evaluations_per_hour'],300)
        self.assertAlmostEqual(v['windows']['60']['remaining_hours'],1935/300)
        self.assertEqual(v['recent_oracle_mean_seconds'],2)
        v=update_progress(self.root,self.recipe,now=900,activate=True)
        self.assertEqual(v['last_hour_commits'],50)
    def test_paused_and_failed_reads_never_claim_completion(self):
        update_progress(self.root,self.recipe,now=100,activate=True)
        self.add(0,50,200);(self.root/'PAUSE').touch()
        v=update_progress(self.root,self.recipe,now=700)
        self.assertEqual(v['status'],'paused');self.assertIsNone(v['windows']['60']['remaining_hours'])
        self.path.write_bytes(b'not a database')
        v=update_progress(self.root,self.recipe,now=800)
        self.assertEqual(v['status'],'unverified');self.assertTrue(v['read_errors']);self.assertEqual(v['evaluations'],50)
    def test_no_recent_progress_has_no_eta(self):
        update_progress(self.root,self.recipe,now=100,activate=True)
        self.add(0,50,200);v=update_progress(self.root,self.recipe,now=4000)
        self.assertEqual(v['last_hour_commits'],0);self.assertIsNone(v['windows']['60']['remaining_hours'])
    def test_completion_and_throttling(self):
        self.recipe['query']['test']=['q1'];self.recipe['query']['proposals']=50
        self.add(0,50,50)
        v=update_progress(self.root,self.recipe,now=100,activate=True)
        self.assertEqual(v['status'],'complete');self.assertEqual(v['windows']['60']['remaining_hours'],0)
        self.assertIsNone(update_progress(self.root,self.recipe,now=101))

if __name__=='__main__':unittest.main()
