import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import psycopg
from query_resource_recovery import evaluate_query, QueryInfrastructureRetryExhausted
from single_method import Journal, best, observed


DSA = 'invalid DSA memory alloc request size 1811939328\nCONTEXT:  parallel worker'


class QueryResourceTests(unittest.TestCase):
    def oracle(self, errors):
        return Mock(worst_runtime=80, side_effect=errors)

    def test_parallel_dsa_is_censored_and_durably_replayed(self):
        oracle = self.oracle([psycopg.errors.InternalError_(DSA)])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'runtime.sqlite'
            journal = Journal(path, {'task': 'test'})
            result = journal.once('measurement/4/4', {'plan': [1]},
                                  lambda: evaluate_query(oracle, [1], 50), measurement=True)
            journal.close()
            journal = Journal(path, {'task': 'test'})
            saved = journal.once('measurement/4/4', {'plan': [1]},
                                 lambda: self.fail('repeated completed SQL'), measurement=True)
            journal.close()
        self.assertEqual(result, saved)
        self.assertEqual(result['runtime'], 80)
        self.assertEqual(result['resource_limit'], 'parallel_dsa_allocation')
        self.assertEqual(result['status'], 'query_resource_error')
        self.assertTrue(result['runtime_is_penalty'])
        self.assertTrue(result['censored'])
        self.assertEqual(result['error'], DSA)
        self.assertIsNone(best([observed([1], result, 0, 'fsbo')]))
        oracle.assert_called_once_with([1], 50)

    def test_unrelated_internal_and_configuration_errors_raise(self):
        for error in [psycopg.errors.InternalError_('unexpected internal error'),
                      psycopg.errors.InternalError_('invalid DSA memory alloc request size 42'),
                      psycopg.errors.ConfigurationLimitExceeded('unrecognized limit')]:
            oracle = self.oracle([error])
            with self.assertRaises(type(error)):
                evaluate_query(oracle, [1], 50)
            self.assertEqual(oracle.call_count, 1)

    def test_temp_file_policy_preserved(self):
        oracle = self.oracle([psycopg.errors.ConfigurationLimitExceeded('temporary file size exceeds temp_file_limit')])
        result = evaluate_query(oracle, [1], 20)
        self.assertEqual(result['resource_limit'], 'temp_file_limit')
        self.assertEqual(result['runtime'], 40)
        self.assertTrue(result['censored'])

    def test_transport_retry_then_resource_failure_keeps_provenance(self):
        oracle = self.oracle([psycopg.errors.ConnectionTimeout('connection timeout'),
                              psycopg.errors.InternalError_(DSA)])
        with patch('query_resource_recovery.time.sleep'):
            result = evaluate_query(oracle, [1], 50)
        self.assertEqual(result['infrastructure_retries'], 1)
        self.assertEqual(result['infrastructure_failures'][0]['kind'], 'connection_timeout')
        self.assertEqual(result['resource_limit'], 'parallel_dsa_allocation')

    def test_transport_exhaustion_remains_unscored(self):
        oracle = self.oracle([psycopg.errors.ConnectionTimeout('timeout')] * 4)
        with patch('query_resource_recovery.time.sleep'):
            with self.assertRaises(QueryInfrastructureRetryExhausted):
                evaluate_query(oracle, [1], 50)
        self.assertEqual(oracle.call_count, 4)


if __name__ == '__main__':
    unittest.main()
