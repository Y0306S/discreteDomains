"""Amendment identity checks and dispatch safeguards, without launching jobs."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from peptide_rgpe_merged_policy import digest, verify
import peptide_rgpe_merged_dispatch as dispatch


class AmendmentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.recipe = self.root/'recipe.json'
        self.recipe.write_text(json.dumps(dict(method='rgpe', vae_policy='frozen_author', e2e_every=0)))
        self.code = {str(ROOT/'tools'/name):digest(ROOT/'tools'/name) for name in
            ('peptide_rgpe_merged.py', 'peptide_rgpe_merged_policy.py',
             'peptide_rgpe_merged_launch.py', 'peptide_rgpe_merged_worker.py')}
        self.report = self.root/'fixture_validation.json'
        self.report.write_text(json.dumps(dict(passed=True, code_sha256=self.code,
            production_scale_passed=True, recovery_passed=True, purpose='unit_test_fixture_only')))
        self.amendment = dict(format='rgpe-merged-acquisition-v1', campaign=str(self.root),
            recipe_sha256=digest(self.recipe), code=self.code,
            validation=dict(path=str(self.report),sha256=digest(self.report)))
        self.path = self.root/'amendment.json'
    def tearDown(self): self.tmp.cleanup()
    def save(self): self.path.write_text(json.dumps(self.amendment))
    def test_valid_amendment_preserves_recipe_bytes(self):
        self.save(); before = self.recipe.read_bytes()
        self.assertEqual(verify(self.path)['code'],self.code)
        self.assertEqual(self.recipe.read_bytes(),before)
    def test_changed_code_rejected(self):
        self.amendment['code'][next(iter(self.code))] = 'bad'
        self.save()
        with self.assertRaises(ValueError): verify(self.path)
    def test_changed_recipe_rejected(self):
        self.save(); self.recipe.write_text('{}')
        with self.assertRaises(ValueError): verify(self.path)
    def test_changed_report_rejected(self):
        self.save(); self.report.write_text('{}')
        with self.assertRaises(ValueError): verify(self.path)
    def test_unvalidated_recovery_or_production_rejected(self):
        for key in ('passed','production_scale_passed','recovery_passed'):
            report=dict(passed=True,code_sha256=self.code,production_scale_passed=True,recovery_passed=True)
            report[key]=False; self.report.write_text(json.dumps(report))
            self.amendment['validation']['sha256']=digest(self.report)
            self.save()
            with self.assertRaises(ValueError): verify(self.path)
    def test_missing_dependency_rejected(self):
        self.amendment['code']={}; self.save()
        with self.assertRaises(ValueError): verify(self.path)


class DispatchTest(unittest.TestCase):
    def test_quota_and_memory_growth_guards(self):
        self.assertEqual(dispatch.admission_limit(40,3,30,20),7)
        self.assertEqual(dispatch.admission_limit(40,3,41,20),0)
        self.assertFalse(dispatch.memory_admissible(256,40,16,[1]*28,8))
        self.assertTrue(dispatch.memory_admissible(256,40,16,[1]*27,8))
    def test_prior_cpu_owner_and_children_block_admission(self):
        rows={1:dict(pid=1,ppid=0,args=['bora'],cpus=[12]),
              2:dict(pid=2,ppid=1,args=['oracle'],cpus=[12])}
        self.assertIn(12,dispatch.blockers(rows,[12],set(),{}))
        del rows[1]
        self.assertIn(12,dispatch.blockers(rows,[12],set(),{}))
        own={v['pid'] for v in dispatch.descendants(rows,2)}
        self.assertEqual(dispatch.blockers(rows,[12],own,{}),{})
    def test_exclusive_task_claim(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'claim.lock'; first=dispatch.claim(path)
            self.assertIsNone(dispatch.claim(path)); first.close()
            second=dispatch.claim(path); self.assertIsNotNone(second); second.close()


if __name__ == '__main__': unittest.main(verbosity=2)
