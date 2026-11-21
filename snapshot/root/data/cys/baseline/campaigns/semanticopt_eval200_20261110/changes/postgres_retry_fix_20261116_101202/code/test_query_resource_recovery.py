import unittest

import psycopg

from query_resource_recovery import evaluate_query


class Oracle:
    worst_runtime = 1000

    def __init__(self, error=None, result=None):
        self.error, self.result = error, result
        self.calls = 0

    def __call__(self, plan, timeout):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


class ResourceRecoveryTest(unittest.TestCase):
    def test_resource_limit_is_censored_without_retry(self):
        oracle = Oracle(psycopg.errors.ConfigurationLimitExceeded(
            'temporary file size exceeds temp_file_limit (8388608kB)'))
        result = evaluate_query(oracle, [1], 120)
        self.assertEqual(oracle.calls, 1)
        self.assertEqual(result['runtime'], 240)
        self.assertEqual(result['sqlstate'], '53400')
        self.assertEqual(result['status'], 'query_resource_error')
        self.assertTrue(result['censored'])
        self.assertTrue(result['runtime_is_penalty'])

    def test_success_is_unchanged(self):
        result = dict(runtime=.12, censored=0, status='completed')
        self.assertIs(evaluate_query(Oracle(result=result), [], 120), result)

    def test_unrelated_failures_propagate(self):
        for error in [psycopg.errors.ConfigurationLimitExceeded('other limit'),
                      psycopg.OperationalError('connection lost'),
                      psycopg.errors.UndefinedTable('bad SQL')]:
            with self.subTest(error=error), self.assertRaises(type(error)):
                evaluate_query(Oracle(error), [], 120)


if __name__ == '__main__':
    unittest.main()
