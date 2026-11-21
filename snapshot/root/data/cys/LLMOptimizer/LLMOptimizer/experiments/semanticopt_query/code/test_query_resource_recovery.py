import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import psycopg

from query_resource_recovery import evaluate_query, QueryInfrastructureRetryExhausted


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
                      psycopg.errors.InternalError_('could not open temporary file "x": No space left on device'),
                      psycopg.errors.InternalError_('could not open file "x": Interrupted system call'),
                      psycopg.errors.UndefinedTable('bad SQL')]:
            with self.subTest(error=error), patch('query_resource_recovery.time.sleep') as sleep:
                oracle = Oracle(error)
                with self.assertRaises(type(error)):
                    evaluate_query(oracle, [], 120)
                self.assertEqual(oracle.calls, 1)
                sleep.assert_not_called()

    def test_known_terminal_errors_retry_same_proposal_and_keep_real_score(self):
        errors = [psycopg.errors.ConnectionTimeout('connection timeout expired'),
                  psycopg.errors.InternalError_(
                      'could not open temporary file "base/pgsql_tmp/x": Interrupted system call')]
        for error in errors:
            with self.subTest(error=error), patch('query_resource_recovery.time.sleep') as sleep:
                measured = dict(runtime=.12, censored=0, status='completed', elapsed_seconds=.12)
                calls = []
                def oracle(plan, timeout):
                    calls.append((list(plan), timeout))
                    if len(calls) < 3:
                        raise error
                    return measured
                result = evaluate_query(oracle, [1, 2, 3], 50)
                self.assertEqual(calls, [([1, 2, 3], 50)] * 3)
                self.assertEqual(result['runtime'], measured['runtime'])
                self.assertEqual(result['elapsed_seconds'], measured['elapsed_seconds'])
                self.assertEqual(result['status'], 'completed')
                self.assertEqual(result['infrastructure_retries'], 2)
                self.assertEqual(len(result['infrastructure_failures']), 2)
                self.assertEqual([x.args[0] for x in sleep.call_args_list], [1, 2])

    def test_persistent_failure_is_bounded_without_fabricated_score(self):
        oracle = Oracle(psycopg.errors.ConnectionTimeout('connection timeout expired'))
        with patch('query_resource_recovery.time.sleep') as sleep:
            with self.assertRaises(QueryInfrastructureRetryExhausted) as error:
                evaluate_query(oracle, [1], 50)
        self.assertEqual(oracle.calls, 4)
        self.assertEqual(len(error.exception.failures), 4)
        self.assertEqual([x.args[0] for x in sleep.call_args_list], [1, 2, 4])

    def test_retry_exhaustion_persists_selected_proposal_for_resume(self):
        from evaluate import connect, get, measure_pending, put
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            db = connect(folder)
            pending = dict(status='selected', number=0, answer={'next_configuration': {'plan': [1]}})
            with db:
                put(db, 'pending', pending)
            failing = Oracle(psycopg.errors.ConnectionTimeout('connection timeout expired'))
            def call(config):
                return evaluate_query(failing, config['plan'], 50)
            with patch('query_resource_recovery.time.sleep'):
                with self.assertRaises(QueryInfrastructureRetryExhausted):
                    measure_pending(db, pending, call)
            db.close()
            db = connect(folder)
            restored = get(db, 'pending')
            self.assertEqual(restored['status'], 'selected')
            self.assertEqual(restored['answer'], pending['answer'])
            self.assertEqual(db.execute('SELECT count(*) FROM records').fetchone()[0], 0)
            audits = db.execute("SELECT value FROM state WHERE key LIKE 'infrastructure_failure_%'").fetchall()
            self.assertEqual(len(audits), 1)
            self.assertEqual(len(json.loads(audits[0][0])['failures']), 4)
            calls = []
            def recovered(config):
                calls.append(config)
                return {'runtime': .5, 'censored': 0}
            result = measure_pending(db, restored, recovered)
            self.assertEqual(result['runtime'], .5)
            self.assertEqual(calls, [pending['answer']['next_configuration']])
            db.close()

    def test_ambiguous_connection_loss_remains_in_flight(self):
        from evaluate import connect, get, measure_pending
        with tempfile.TemporaryDirectory() as temp:
            db = connect(Path(temp))
            pending = dict(status='selected', number=0, answer={'next_configuration': {'plan': [1]}})
            def ambiguous(config):
                raise psycopg.OperationalError('connection lost during execution')
            with self.assertRaises(psycopg.OperationalError):
                measure_pending(db, pending, ambiguous)
            self.assertEqual(get(db, 'pending')['status'], 'in_flight')
            self.assertEqual(db.execute('SELECT count(*) FROM records').fetchone()[0], 0)
            db.close()


if __name__ == '__main__':
    unittest.main()
