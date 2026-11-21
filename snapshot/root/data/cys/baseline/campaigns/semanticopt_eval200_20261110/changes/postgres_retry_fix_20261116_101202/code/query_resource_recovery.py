"""Treat PostgreSQL's bounded temporary-file exhaustion as a failed plan."""
import time

import psycopg


def evaluate_query(oracle, plan, timeout):
    started = time.perf_counter()
    try:
        return oracle(plan, timeout)
    except psycopg.errors.ConfigurationLimitExceeded as exc:
        # Other configuration errors remain infrastructure failures. Never turn
        # connection, SQL, or extension failures into invented measurements.
        if 'temporary file size exceeds temp_file_limit' not in str(exc):
            raise
        return dict(runtime=min(oracle.worst_runtime, timeout * 2), censored=1,
                    status='query_resource_error', sqlstate=exc.sqlstate,
                    error=str(exc), elapsed_seconds=time.perf_counter() - started,
                    runtime_is_penalty=True, resource_limit='temp_file_limit')
