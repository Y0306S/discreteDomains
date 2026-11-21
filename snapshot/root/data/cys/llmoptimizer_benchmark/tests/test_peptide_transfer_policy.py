import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'peptide_transfer_policy.py'
spec = importlib.util.spec_from_file_location('transfer_policy', SCRIPT)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class PolicyTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.recipe_path = self.root / 'recipe.json'
        self.loader = self.root / 'peptide_transfer_policy.py'
        self.helper = self.root / 'fsbo_target_refit.py'
        self.changed = self.root / 'peptide_transfer_models.py'
        for path in (self.loader, self.helper, self.changed):
            path.write_text('new code')
        self.recipe = dict(method='fsbo', fsbo_target_epochs=10000,
            fsbo_target_patience=16, code={str(self.changed): 'previous-code'})
        self.recipe_path.write_text(json.dumps(self.recipe))
        code = {str(path): dict(previous_sha256=self.recipe['code'].get(str(path)),
            sha256=policy.digest(path)) for path in (self.loader, self.helper, self.changed)}
        self.report = self.root / 'validation.json'
        settings = dict(maximum_updates=1000, patience=16, min_delta=0.0001,
            likelihood='deterministic_seeded_likelihood')
        self.report.write_text(json.dumps(dict(passed=True, settings=settings,
            code_sha256={name: entry['sha256'] for name, entry in code.items()})))
        self.amendment = dict(format='fsbo-target-fit-amendment-v1',
            recipe_sha256=policy.digest(self.recipe_path), code=code,
            validation=dict(path=str(self.report), sha256=policy.digest(self.report)),
            settings=settings)
        self.amendment_path = self.root / 'target_fit_amendment.json'

    def load(self):
        self.amendment_path.write_text(json.dumps(self.amendment))
        with patch.object(policy, '__file__', str(self.loader)):
            return policy.effective_recipe(self.recipe_path)

    def test_original_recipe_without_amendment(self):
        self.assertEqual(policy.effective_recipe(self.recipe_path), self.recipe)

    def test_amendment_preserves_recovery_recipe(self):
        before = self.recipe_path.read_bytes()
        value = self.load()
        self.assertEqual(self.recipe_path.read_bytes(), before)
        self.assertEqual(value['code'][str(self.changed)], policy.digest(self.changed))
        self.assertEqual(value['fsbo_target_stopping']['patience'], 16)
        self.assertEqual(value['fsbo_target_epochs'], 1000)
        self.assertEqual(value['fsbo_original_target_epochs'], 10000)

    def test_wrong_recipe_identity_rejected(self):
        self.amendment['recipe_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'recipe identity'):
            self.load()

    def test_modified_helper_rejected(self):
        self.helper.write_text('changed after validation')
        with self.assertRaisesRegex(ValueError, 'code changed'):
            self.load()

    def test_modified_report_rejected(self):
        self.report.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'report changed'):
            self.load()

    def test_unrelated_code_change_rejected(self):
        self.amendment['code'][str(self.root / 'oracle.py')] = dict(previous_sha256=None, sha256='fake')
        with self.assertRaisesRegex(ValueError, 'Unrelated code'):
            self.load()

    def test_expanded_update_cap_rejected(self):
        self.amendment['settings']['maximum_updates'] = 10001
        with self.assertRaisesRegex(ValueError, 'settings'):
            self.load()


if __name__ == '__main__':
    unittest.main()
