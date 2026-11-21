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

    def test_transient_relation_read_retries_same_plan_and_keeps_real_score(self):
        error = psycopg.errors.IoError('could not read block 81710 in file "base/16385/16475": Input/output error')
        measured = dict(runtime=1.25, censored=0, status='completed')
        oracle = self.oracle([error, measured])
        plan = [1, 2, 3]
        with patch('query_resource_recovery.time.sleep'):
            result = evaluate_query(oracle, plan, 50)
        self.assertEqual(result['runtime'], 1.25)
        self.assertFalse(result['censored'])
        self.assertEqual(result['infrastructure_failures'][0]['kind'], 'relation_read_io_error')
        self.assertEqual(result['infrastructure_failures'][0]['error'], str(error))
        self.assertEqual(oracle.call_count, 2)
        self.assertTrue(all(c.args == (plan, 50) for c in oracle.call_args_list))

    def test_persistent_relation_read_stays_unscored_in_journal(self):
        error = psycopg.errors.IoError('could not read block 1 in file "base/16385/16475.1": Input/output error')
        oracle = self.oracle([error] * 4)
        with tempfile.TemporaryDirectory() as folder:
            journal = Journal(Path(folder) / 'runtime.sqlite', {'task': 'test'})
            with patch('query_resource_recovery.time.sleep'):
                with self.assertRaises(QueryInfrastructureRetryExhausted):
                    journal.once('measurement/4/4', {'plan': [1]},
                                 lambda: evaluate_query(oracle, [1], 50), measurement=True)
            entry = journal.get('measurement/4/4')
            journal.close()
        self.assertEqual(entry['status'], 'safe_failed')
        self.assertNotIn('result', entry)
        self.assertEqual(len(entry['failures']), 4)

    def test_other_storage_errors_are_not_retried(self):
        for message in ['could not write block 1: Input/output error',
                        'invalid page in block 1 of relation base/16385/16475',
                        'could not read block 1 in file "base/16385/16475": Permission denied']:
            oracle = self.oracle([psycopg.errors.IoError(message)])
            with self.assertRaises(psycopg.errors.IoError):
                evaluate_query(oracle, [1], 50)
            self.assertEqual(oracle.call_count, 1)

    def test_terminal_temporary_file_read_retries_without_penalty(self):
        error = psycopg.errors.IoError('could not read file "base/pgsql_tmp/pgsql_tmp752781.35": Input/output error\nCONTEXT:  parallel worker')
        oracle = self.oracle([error, dict(runtime=3.5, censored=0, status='completed')])
        with patch('query_resource_recovery.time.sleep'):
            result = evaluate_query(oracle, [1], 50)
        self.assertEqual(result['runtime'], 3.5)
        self.assertFalse(result['censored'])
        self.assertEqual(result['infrastructure_failures'][0]['kind'], 'temporary_file_read_io_error')
        self.assertEqual(oracle.call_count, 2)

    def test_parallel_tablespace_spill_read_retries_same_proposal(self):
        error = psycopg.errors.IoError('could not read file "pg_tblspc/32775/PG_16_202307071/pgsql_tmp/pgsql_tmp65404.1.fileset/i9of32.p0.0": Input/output error\nCONTEXT:  parallel worker')
        oracle = self.oracle([error, dict(runtime=2.5, censored=0, status='completed')])
        plan = [1, 2, 3]
        with patch('query_resource_recovery.time.sleep'):
            result = evaluate_query(oracle, plan, 50)
        self.assertEqual(result['runtime'], 2.5)
        self.assertFalse(result['censored'])
        self.assertEqual(result['infrastructure_failures'][0]['kind'], 'temporary_file_read_io_error')
        self.assertTrue(all(call.args == (plan, 50) for call in oracle.call_args_list))

    def test_non_temporary_tablespace_io_errors_are_not_retried(self):
        for message in ['could not read file "pg_tblspc/32775/PG_16_202307071/16385/16475": Input/output error',
                        'could not write file "pg_tblspc/32775/PG_16_202307071/pgsql_tmp/pgsql_tmp1.2": Input/output error',
                        'could not read file "pg_tblspc/32775/PG_16_202307071/pgsql_tmp/pgsql_tmp1.2": Permission denied']:
            oracle = self.oracle([psycopg.errors.IoError(message)])
            with self.assertRaises(psycopg.errors.IoError):
                evaluate_query(oracle, [1], 50)
            self.assertEqual(oracle.call_count, 1)


if __name__ == '__main__':
    unittest.main()
