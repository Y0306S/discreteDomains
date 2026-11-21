#!/usr/bin/env python3
"""Infrastructure-only wrapper around the immutable test200 TPE protocol."""

import collections
import contextlib
import functools
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

import peptide_test200 as adapter
from mindware.domains.protein import peptide_rpc
from peptide_tpe_rpc import PersistentRemoteObjective

PROFILE_PATH = Path('/root/data/logs/peptide_test200_baselines_20261027_v2/workers/tpe_profile.sqlite')
active_profile = None


class Profile:
    def __init__(self, task):
        self.task = task
        self.started = time.time()
        self.stages = collections.defaultdict(lambda: dict(calls=0, seconds=0.0))
        self.errors = 0
        self.tells = 0

    def add(self, stage, seconds):
        value = self.stages[stage]
        value['calls'] += 1
        value['seconds'] += seconds

    def flush(self):
        started = time.monotonic()
        try:
            with sqlite3.connect(str(PROFILE_PATH), timeout=10) as db:
                db.execute('CREATE TABLE IF NOT EXISTS profiles (task TEXT PRIMARY KEY, value TEXT NOT NULL)')
                db.execute('INSERT OR REPLACE INTO profiles VALUES (?,?)', (self.task, json.dumps(dict(
                    task=self.task, started=self.started, updated=time.time(), stages=dict(self.stages),
                    tell_calls=self.tells, profile_errors=self.errors, pid=os.getpid(),
                    infrastructure_only=True, stage_timings_can_overlap=True))))
        except sqlite3.Error:
            self.errors += 1
        finally:
            self.add('profile_persistence', time.monotonic() - started)


def timing(stage, seconds):
    if active_profile is not None:
        active_profile.add(stage, seconds)


class ProfiledConnection(sqlite3.Connection):
    def __exit__(self, *args):
        started = time.monotonic()
        try:
            return super().__exit__(*args)
        finally:
            timing('checkpoint_commit', time.monotonic() - started)


def profiled_connect(path):
    db = sqlite3.connect(str(path), timeout=60, factory=ProfiledConnection)
    db.execute('PRAGMA journal_mode=DELETE')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value BLOB NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS trials (number INTEGER PRIMARY KEY, value BLOB NOT NULL)')
    db.commit()
    return db


class ProfiledRemoteObjective(PersistentRemoteObjective):
    def __init__(self, objective, url, identity):
        super().__init__(objective, url, identity, timing=timing)


def measured(stage, function):
    @functools.wraps(function)
    def call(*args, **kwargs):
        started = time.monotonic()
        try:
            return function(*args, **kwargs)
        finally:
            timing(stage, time.monotonic() - started)
    return call


def install():
    import optuna
    original = adapter.run_tpe
    original_tell = optuna.study.Study.tell

    def run_tpe(args, recipe):
        global active_profile
        PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        active_profile = Profile(args.task)
        try:
            return original(args, recipe)
        finally:
            active_profile.flush()
            active_profile = None

    def tell(*args, **kwargs):
        started = time.monotonic()
        try:
            return original_tell(*args, **kwargs)
        finally:
            timing('sampler_tell', time.monotonic() - started)
            if active_profile is not None:
                active_profile.tells += 1
                if active_profile.tells % 25 == 0:
                    active_profile.flush()

    adapter.run_tpe = run_tpe
    adapter.connect = profiled_connect
    for name in ['get', 'put', 'trials', 'add_trial']:
        setattr(adapter, name, measured('checkpoint_' + name, getattr(adapter, name)))
    optuna.study.Study.ask = measured('sampler_ask', optuna.study.Study.ask)
    optuna.study.Study.tell = tell
    peptide_rpc.RemoteObjective = ProfiledRemoteObjective
    adapter.worker = worker


def worker(args):
    recipe = adapter.verify(args.campaign)
    adapter.preflight(args.campaign)
    if recipe['method'] != 'tpe':
        raise ValueError('Optimized runtime is restricted to TPE')
    tasks = recipe['test']
    assigned = tasks[args.index::args.workers]
    ordered = assigned + [task for task in tasks if task not in assigned]
    child, stopping = None, False

    def stop(_sig, _frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for task in ordered:
        if stopping:
            return
        if (args.campaign / 'runs' / (task + '.tar.gz')).exists():
            continue
        command = [sys.executable, str(Path(__file__).resolve()), 'run-task', '--campaign', str(args.campaign),
                   '--task', task, '--cpu', str(args.cpu), '--server', args.server, '--launch']
        child = subprocess.Popen(command, env=os.environ.copy())
        result = child.wait()
        if result not in (0, 73):
            adapter.bo.atomic_json(args.campaign / 'worker_errors' / ('%d.json' % args.index),
                dict(task=task, returncode=result, time=time.time()))
            raise RuntimeError('TPE worker stopped after a task error')


if __name__ == '__main__':
    install()
    adapter.main()
