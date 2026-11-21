#!/usr/bin/env python3
"""Run bounded peptide diagnostics through the real CLI in separate processes."""

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

RELEASE = Path('/root/data/logs/peptide_bo_release_20261003')
sys.path.insert(0, str(RELEASE))
sys.path.insert(0, str(RELEASE / 'tools'))
import peptide_bo_campaign as campaign
from mindware.domains.protein.verified_bo import atomic_json, digest
import torch


def main(args):
    recipe = campaign.verify_recipe(args.campaign)
    if recipe['purpose'] != 'component_check' or (
        recipe['initialization'], recipe['proposals'], recipe['batch_size'], recipe['e2e_every']
    ) != (100, 501, 50, 10):
        raise ValueError('Expected a separate diagnostic recipe: init 100, proposals 501, batch 50, E2E 10')
    sys.path.insert(0, recipe['upstream'])
    report_path = args.campaign / 'extended_report.json'
    report = dict(passed=False, recipe_sha256=digest(args.campaign / 'recipe.json'),
                  cases=[], blocked_checks={}, gpu=args.gpu, cpus=args.cpus)
    atomic_json(report_path, report)

    for task in recipe['blocked_tasks']:
        try:
            campaign.run_task(SimpleNamespace(campaign=args.campaign, task=task, launch=False))
        except ValueError as exc:
            if recipe['blocked_tasks'][task] not in str(exc):
                raise
            report['blocked_checks'][task] = str(exc)
        else:
            raise AssertionError('Impossible task was allowed to run')
    atomic_json(report_path, report)

    def run(task, suffix, stop=None):
        command = [sys.executable, str(RELEASE / 'tools/peptide_bo_campaign.py'), 'run-task',
            '--campaign', str(args.campaign), '--task', task, '--gpu', str(args.gpu), '--cpus', args.cpus]
        if stop is not None:
            command += ['--stop-after-rounds', str(stop)]
        started = time.monotonic()
        for attempt in range(1, 5):
            path = args.campaign / (task + '.' + suffix + '.retry%d.log' % attempt)
            with path.open('w') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         env=dict(os.environ, PYTHONUNBUFFERED='1'))
                try:
                    code = child.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    child.send_signal(signal.SIGTERM)
                    try:
                        child.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    raise RuntimeError('Diagnostic exceeded ten-minute limit: ' + task)
            refused = any(message in path.read_text() for message in (
                'or its SMT sibling is busy', 'is occupied; existing work will not be interrupted'))
            if not code or not refused:
                break
            time.sleep(2)
        duration = time.monotonic() - started
        if code:
            raise RuntimeError('Diagnostic failed: %s (exit %d), see %s' % (task, code, path))
        return duration

    for task in ('pep_003', 'pep_001', 'pep_014', 'pep_018', 'pep_050'):
        case = dict(task=task, passed=False)
        report['cases'].append(case)
        try:
            prefix = None
            wall_seconds = 0
            archive = args.campaign / 'runs' / (task + '.tar.gz')
            if task == 'pep_003' and not archive.exists():
                checkpoint = args.campaign / 'runs' / task / 'checkpoint.pt'
                prior = torch.load(checkpoint, map_location='cpu', weights_only=False) if checkpoint.exists() else None
                if prior is None or prior['progress']['rounds'] < 10:
                    wall_seconds += run(task, 'before_resume', stop=10)
                del prior
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                assert saved['progress']['rounds'] == 10
                assert saved['progress']['proposals'] == 500
                assert saved['state']['tot_num_e2e_updates'] == 1
                prefix = saved['rows']
                case['checkpoint_sha256'] = digest(checkpoint)
                case['resumed_in_new_process'] = True
                del saved
            reused = archive.exists()
            if not reused:
                wall_seconds += run(task, 'finish')
            history = campaign.verify_archive(archive)
            observations = history['observations']
            assert len(observations) == 601
            assert history['rounds'] == 11 and history['proposals'] == 501
            assert history['e2e_updates'] == 1
            assert sum(row['round'] == 11 for row in observations) == 1
            assert history['fresh_evaluations'] + history['cache_hits'] == 501
            if prefix is not None:
                assert observations[:len(prefix)] == prefix
            initial = observations[:100]
            proposed = observations[100:]
            best = min(row['mic'] for row in observations if row['feasible'])
            assert math.isclose(best, history['best_feasible_mic'], rel_tol=1e-6, abs_tol=1e-6)
            case.update(passed=True, archive=str(archive), archive_sha256=digest(archive),
                reference=history['reference'], proposals=501, rounds=11, e2e_updates=1,
                fresh_evaluations=history['fresh_evaluations'], cache_hits=history['cache_hits'],
                feasible_proposals=sum(row['feasible'] for row in proposed),
                distinct_proposals=len({row['sequence'] for row in proposed}),
                initial_best_mic=min(row['mic'] for row in initial), best_feasible_mic=best,
                reported_compute_seconds=history['seconds'],
                subprocess_wall_seconds=None if reused else wall_seconds,
                reused_completed_diagnostic=reused,
                archive_bytes=archive.stat().st_size)
            if prefix is not None:
                archive_hash = digest(archive)
                run(task, 'already_complete')
                assert digest(archive) == archive_hash
                case['completed_rerun_unchanged'] = True
        except Exception as exc:
            case['error'] = type(exc).__name__ + ': ' + str(exc)
            case['traceback'] = traceback.format_exc()
        atomic_json(report_path, report)
        print(json.dumps(case), flush=True)
    report['passed'] = all(case['passed'] for case in report['cases'])
    atomic_json(report_path, report)
    print(json.dumps(dict(passed=report['passed'], report=str(report_path))), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--cpus', required=True)
    raise SystemExit(main(parser.parse_args()))
