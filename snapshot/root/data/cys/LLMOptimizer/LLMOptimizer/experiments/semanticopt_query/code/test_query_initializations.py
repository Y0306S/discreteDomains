import copy
import json
import sqlite3
import sys
import unittest

from common import ROOT, QUERY_SOURCE
from evaluate import query_initializations
from protocol import validate_query_plan

sys.path.insert(0, str(QUERY_SOURCE/'code'))
import run_query_ceb_baselines as query
query.native_path(QUERY_SOURCE/'native')
from codec.codec import AliasesCodec
from workload.workloads import get_workload_set


class QueryInitializationResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task='CEB_11B163'
        native=QUERY_SOURCE/'native'
        cls.definition=get_workload_set('CEB_3K').extract_query(
            query.sql_path(native,cls.task).read_text(),cls.task)
        cls.codec=AliasesCodec(cls.definition.all_tables)
        cls.raw=query.load_initializations(
            native/'initialization_data/ceb_3k_bao_initialization.csv',[cls.task])[cls.task]
        path=ROOT/'query/runs'/cls.task/'observations.sqlite'
        with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True) as db:
            cls.saved=json.loads(db.execute("SELECT value FROM state WHERE key='initial'").fetchone()[0])

    def test_real_checkpoint_is_preserved_exactly(self):
        before=copy.deepcopy(self.saved)
        result=query_initializations(self.raw,self.definition,self.codec,self.saved)
        self.assertEqual(result,before)
        self.assertEqual(self.saved,before)

    def test_equivalent_independent_branch_order_is_accepted(self):
        saved=copy.deepcopy(self.saved)
        # This real Bao plan starts with two independent joins. Exchanging
        # their evaluation order leaves every branch, side and operator intact.
        plan=saved[13]['configuration']['plan']
        changed=plan[5:10]+plan[:5]+plan[10:]
        left=self.codec.decode(self.definition.query_tables,plan.copy())
        right=self.codec.decode(self.definition.query_tables,changed.copy())
        self.assertTrue(left.equal(right))
        saved[13]['configuration']['plan']=changed
        self.assertEqual(query_initializations(self.raw,self.definition,self.codec,saved),saved)
        validate_query_plan(plan,self.definition,self.codec)
        validate_query_plan(changed,self.definition,self.codec)

    def test_candidate_decoder_repairs_are_rejected(self):
        plan=self.saved[13]['configuration']['plan'].copy()
        plan[5:10]=plan[:5]
        with self.assertRaisesRegex(ValueError,'implicit codec repair'):
            validate_query_plan(plan,self.definition,self.codec)
        plan=self.saved[0]['configuration']['plan'].copy()
        plan[4]=4
        with self.assertRaisesRegex(ValueError,'Unsupported join operator'):
            validate_query_plan(plan,self.definition,self.codec)

    def test_changed_measurements_and_counts_are_rejected(self):
        for field,value in [('objective',999.),('censored',True),('status','timeout'),('feasible',False)]:
            with self.subTest(field=field):
                saved=copy.deepcopy(self.saved);saved[0][field]=value
                with self.assertRaisesRegex(ValueError,'measurement changed'):
                    query_initializations(self.raw,self.definition,self.codec,saved)
        with self.assertRaisesRegex(ValueError,'count changed'):
            query_initializations(self.raw,self.definition,self.codec,self.saved[:-1])

    def test_changed_tree_or_invalid_plan_is_rejected(self):
        saved=copy.deepcopy(self.saved)
        saved[0]['configuration']['plan'][4]=2
        with self.assertRaisesRegex(ValueError,'join tree changed'):
            query_initializations(self.raw,self.definition,self.codec,saved)
        saved=copy.deepcopy(self.saved)
        saved[0]['configuration']['plan'][0]=999
        with self.assertRaisesRegex(ValueError,'outside this query'):
            query_initializations(self.raw,self.definition,self.codec,saved)


if __name__=='__main__':
    unittest.main()
