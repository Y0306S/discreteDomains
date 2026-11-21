"""Bounded recovery for known, terminal errors in read-only query evaluation."""
import re
import sys
import time

import psycopg


class QueryInfrastructureRetryExhausted(RuntimeError):
    """The last call definitively failed; its proposal can safely remain selected."""

    def __init__(self, failures):
        self.failures = failures
        super().__init__(f'Query infrastructure retry limit reached after {len(failures)} calls: '
                         + failures[-1]['error'])


def parallel_dsa_allocation_error(message):
    """Recognize the observed terminal parallel-query allocation failure only."""
    lines = message.strip().splitlines()
    return (len(lines) == 2 and
            re.fullmatch(r'invalid DSA memory alloc request size [0-9]+', lines[0]) is not None and
            lines[1].strip() == 'CONTEXT:  parallel worker')


def resource_limit_error(exc):
    if (isinstance(exc, psycopg.errors.ConfigurationLimitExceeded) and
            'temporary file size exceeds temp_file_limit' in str(exc)):
        return 'temp_file_limit'
    if (isinstance(exc, psycopg.errors.InternalError_) and
            parallel_dsa_allocation_error(str(exc))):
        return 'parallel_dsa_allocation'
    return None


def retryable_error(exc):
    # RuntimeOracle opens a new connection for every read-only SELECT. A failed
    # connect sent no query; a reported temp-file EINTR terminated the statement.
    # A lost connection during execution remains ambiguous and is never retried.
    if isinstance(exc, psycopg.errors.ConnectionTimeout):
        return 'connection_timeout'
    message = str(exc).partition('\n')[0]
    # An explicit relation-read I/O error terminates the read-only statement.
    # The shared filesystem has twice failed such reads transiently; exact
    # page reads and complete relation scans subsequently succeeded. Retry
    # the same proposal, never turn a storage failure into a runtime target.
    if (isinstance(exc, psycopg.errors.IoError) and
            re.fullmatch(r'could not read block [0-9]+ in file "base/[0-9]+/[0-9]+(?:\.[0-9]+)?": Input/output error', message)):
        return 'relation_read_io_error'
    # PostgreSQL also reports terminal reads from its per-query spill files.
    # These are discarded when the failed statement's connection closes.
    if (isinstance(exc, psycopg.errors.IoError) and
            re.fullmatch(r'could not read file "base/pgsql_tmp/pgsql_tmp[0-9]+\.[0-9]+": Input/output error', message)):
        return 'temporary_file_read_io_error'
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
        except psycopg.Error as exc:
            limit = resource_limit_error(exc)
            if limit is not None:
                # Follow RuntimeOracle's existing resource-failure penalty.
                # This is censored, never an exact runtime or incumbent update.
                result = dict(runtime=min(oracle.worst_runtime, timeout * 2), censored=1,
                              status='query_resource_error', sqlstate=exc.sqlstate,
                              error=str(exc), elapsed_seconds=time.perf_counter() - started,
                              runtime_is_penalty=True, resource_limit=limit)
                break
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
        break
    if failures:
        result = dict(result, infrastructure_retries=len(failures),
                      infrastructure_failures=failures,
                      elapsed_seconds_including_recovery=time.perf_counter() - started)
    return result
