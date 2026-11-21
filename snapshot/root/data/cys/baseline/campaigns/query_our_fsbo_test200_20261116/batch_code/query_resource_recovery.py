"""Bounded recovery for known, terminal errors in read-only query evaluation."""
import sys
import time

import psycopg


class QueryInfrastructureRetryExhausted(RuntimeError):
    """The last call definitively failed; its proposal can safely remain selected."""

    def __init__(self, failures):
        self.failures = failures
        super().__init__(f'Query infrastructure retry limit reached after {len(failures)} calls: '
                         + failures[-1]['error'])


def retryable_error(exc):
    # RuntimeOracle opens a new connection for every read-only SELECT. A failed
    # connect sent no query; a reported temp-file EINTR terminated the statement.
    # A lost connection during execution remains ambiguous and is never retried.
    if isinstance(exc, psycopg.errors.ConnectionTimeout):
        return 'connection_timeout'
    message = str(exc).partition('\n')[0]
    if (isinstance(exc, psycopg.errors.InternalError_) and
            message.startswith('could not open temporary file "') and
            message.endswith(': Interrupted system call')):
        return 'temporary_file_interrupted'
    return None


def evaluate_query(oracle, plan, timeout):
    started = time.perf_counter()
    failures = []
    for attempt in range(4):
        try:
            result = oracle(plan, timeout)
        except psycopg.errors.ConfigurationLimitExceeded as exc:
            if 'temporary file size exceeds temp_file_limit' not in str(exc):
                raise
            result = dict(runtime=min(oracle.worst_runtime, timeout * 2), censored=1,
                          status='query_resource_error', sqlstate=exc.sqlstate,
                          error=str(exc), elapsed_seconds=time.perf_counter() - started,
                          runtime_is_penalty=True, resource_limit='temp_file_limit')
        except psycopg.Error as exc:
            kind = retryable_error(exc)
            if kind is None:
                raise
            failures.append(dict(kind=kind, sqlstate=exc.sqlstate, error=str(exc), at=time.time()))
            print(f'Query infrastructure call {attempt + 1}/4 failed ({kind}); '
                  + ('pausing without a score' if attempt == 3 else 'retrying the same proposal'),
                  file=sys.stderr, flush=True)
            if attempt == 3:
                raise QueryInfrastructureRetryExhausted(failures) from exc
            time.sleep(2 ** attempt)
            continue
        if failures:
            result = dict(result, infrastructure_retries=len(failures),
                          infrastructure_failures=failures,
                          elapsed_seconds_including_recovery=time.perf_counter() - started)
        return result
