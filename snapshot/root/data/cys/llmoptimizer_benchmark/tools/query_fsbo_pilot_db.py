#!/usr/bin/env python3
"""Bounded read-only timing helper for the query FSBO capacity pilot."""
import json
from pathlib import Path
import sys
import time


def main():
    import psycopg
    request = json.load(sys.stdin)
    results = []
    with psycopg.connect(host='127.0.0.1', port=request['port'], dbname='imdb',
                         user='imdb', password='imdb', autocommit=True,
                         connect_timeout=5) as conn:
        conn.execute("LOAD 'pg_hint_plan'")
        conn.execute('SET pg_hint_plan.enable_hint=on')
        conn.execute('SET join_collapse_limit=1')
        conn.execute('SET default_transaction_read_only=on')
        conn.execute("SET statement_timeout='10s'")
        # Bound temporary-file growth during this capacity probe only.
        conn.execute("SET temp_file_limit='256MB'")
        settings = {key: conn.execute('SHOW ' + key).fetchone()[0] for key in
                    ['shared_buffers', 'work_mem', 'max_parallel_workers_per_gather',
                     'max_parallel_workers', 'max_worker_processes']}
        for repeat in [-1, 0, 1]:
            started = time.monotonic()
            result = conn.execute('EXPLAIN (ANALYZE, FORMAT JSON, TIMING OFF) ' + request['sql']).fetchone()[0][0]
            nodes = []
            def walk(node):
                nodes.append(node)
                for child in node.get('Plans', []):
                    walk(child)
            walk(result['Plan'])
            results.append(dict(repeat=repeat, wall_seconds=time.monotonic()-started,
                                execution_seconds=result['Execution Time']/1000,
                                workers_planned=max(n.get('Workers Planned', 0) for n in nodes),
                                workers_launched=max(n.get('Workers Launched', 0) for n in nodes)))
    print(json.dumps(dict(status='passed', results=results, settings=settings)))


if __name__ == '__main__':
    main()
