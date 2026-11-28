import collections
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).parent
sources = json.loads(Path('/root/reports/query_convergence_fsbo_tpe_lolbo_20261109/audit.json').read_text())['sources']

def read(source):
    path = Path(source['source'])
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == source['sha256'], str(path)
    rows = json.loads(raw)['observations']
    assert len(rows) == 1049
    c = collections.Counter()
    for i, row in enumerate(rows[49:]):
        e = row['extra_info']
        assert e['proposal_index'] == i
        p = e['proposal']
        assert 'fallback' in p
        fresh = bool(e['database_call'])
        ok = not bool(e['censored'])
        fallback = bool(p['fallback'])
        c['proposals'] += 1
        c['decoder_fallback' if fallback else 'nonfallback'] += 1
        c['fresh' if fresh else 'cache_hits'] += 1
        c['uncensored' if ok else 'censored'] += 1
        if not fallback and ok:
            c['nonfallback_uncensored'] += 1
            if fresh:
                c['fresh_nonfallback_uncensored'] += 1
        c['status:'+e['status']] += 1
        c['decode_retries'] += p['decode_retries']
        if fresh:
            c['fresh_uncensored' if ok else 'fresh_censored'] += 1
            c['fresh_fallback' if fallback else 'fresh_nonfallback'] += 1
            if ok and float(e['runtime']) <= 50:
                c['fresh_completed_within_50'] += 1
        elif fallback:
            c['fallback_cache_hits'] += 1
    return dict(method=source['method'], task=source['task'], counts=c)

totals = collections.defaultdict(collections.Counter)
tasks = []
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for i, row in enumerate(pool.map(read, sources), 1):
        totals[row['method']].update(row['counts'])
        tasks.append(row)
        if i % 100 == 0:
            print('Read histories', i, flush=True)
summary = {}
for method, c in totals.items():
    assert c['proposals'] == 200000
    summary[method] = dict(counts=dict(c),
        decoder_fallback_percent=100*c['decoder_fallback']/c['proposals'],
        cache_hit_percent=100*c['cache_hits']/c['proposals'],
        fresh_sql_uncensored_percent=100*c['fresh_uncensored']/c['fresh'],
        fallback_as_failure_sql_success_percent=100*c['nonfallback_uncensored']/c['proposals'],
        fallback_and_cache_as_failure_sql_success_percent=100*c['fresh_nonfallback_uncensored']/c['proposals'],
        nonfallback_decode_percent=100*c['nonfallback']/c['proposals'],
        all_proposals_uncensored_percent=100*c['uncensored']/c['proposals'],
        fresh_nonfallback_per_proposal_percent=100*c['fresh_nonfallback']/c['proposals'])
result = dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              summary=summary, tasks=tasks,
              notes=['49 Bao initialization observations excluded.',
                     'Each historical campaign completed 200 tasks and 1000 charged proposals per task.',
                     'Decoder fallback is an imported Bao plan after failed native decoding; cached duplicates are allowed.',
                     'Fresh SQL rate excludes cache hits. Native caps: FSBO 50 seconds, TPE 100 seconds, LOLBO adaptive.',
                     'Generation metrics are per charged proposal, not per individual internal decoder retry.',
                     'Original history hashes verified against the existing audited source manifest.'])
(OUT/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(summary, indent=2), flush=True)
