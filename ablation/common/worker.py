"""Idempotent job queue: start any number of workers, any number of times.

A worker scans the fixed job list, skips finished ones, takes whatever is
free, and exits once every job is finished or has failed too often. Nothing
needs to be told where to resume, so a launcher only has to restart workers.
"""

import os
import signal
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

from .resumable import JobStore


class Preempted(Exception):
    """Raised after the state is saved, when the worker was asked to stop."""


@dataclass(frozen=True)
class Job:
    key: str
    config: dict


class StopFlag:
    """Set by SIGTERM/SIGINT, checked by the training loop between steps."""

    def __init__(self):
        self.requested = False
        for signal_number in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signal_number, self._request)

    def _request(self, signal_number, frame):
        print(f'[worker] received signal {signal_number}, stopping after this step',
              flush=True)
        self.requested = True


def job_status(root, jobs, max_failures, stale_seconds):
    counts = Counter()
    for job in jobs:
        store = JobStore(Path(root) / job.key)
        if store.is_done():
            counts['done'] += 1
        elif store.failure_count() >= max_failures:
            counts['failed'] += 1
        elif store.is_claimed(stale_seconds):
            counts['running'] += 1
        elif store.has_state():
            counts['interrupted'] += 1
        else:
            counts['not_started'] += 1
    return counts


def print_status(root, jobs, max_failures, stale_seconds):
    counts = job_status(root, jobs, max_failures, stale_seconds)
    print(f'total={len(jobs)} ' + ' '.join(
        f'{name}={counts[name]}'
        for name in ('done', 'running', 'interrupted', 'not_started', 'failed')))
    for job in jobs:
        store = JobStore(Path(root) / job.key)
        if not store.is_done() and store.failure_count() >= max_failures:
            print(f'FAILED {job.key}: see {store.failures_path}')


def run_worker(root, jobs, execute, max_failures=3, stale_seconds=600,
               wait_seconds=60, wait_for_others=True, stop=None):
    """Run jobs until none remain. Return 0 if all are done, 1 otherwise."""
    import setproctitle
    setproctitle.setproctitle("AI Auto Research Pipeline")
    root = Path(root)
    stop = stop or StopFlag()
    while True:
        unfinished = False
        ran_a_job = False
        failed = 0
        for job in jobs:
            if stop.requested:
                return 1
            store = JobStore(root / job.key)
            if store.is_done():
                continue
            if store.failure_count() >= max_failures:
                failed += 1
                continue
            unfinished = True
            if not store.claim(stale_seconds):
                continue
            try:
                # Another worker may have finished between the check and the claim.
                if store.is_done():
                    continue
                ran_a_job = True
                store.ensure_config(job.config)
                print(f'[worker] start {job.key}', flush=True)
                result = execute(job, store, stop)
                store.write_done(result)
                unresolved = (result.get('stop_reason') in {'step_cap', 'epoch_cap'}
                              or result.get('needs_convergence_extension', False))
                if not unresolved:
                    store.clear_state()
                print(f'[worker] done {job.key}', flush=True)
            except Preempted:
                print(f'[worker] preempted during {job.key}', flush=True)
                return 1
            except Exception as error:  # One bad job must not stop the queue.
                message = traceback.format_exc()
                print(f'[worker] failure in {job.key}:\n{message}', flush=True)
                store.record_failure(message)
                campaign_task = os.environ.get('GNNM_CAMPAIGN_TASK')
                if campaign_task and isinstance(error, FloatingPointError):
                    store.write_done(dict(scientific_status='numerical_failure',
                                          error=str(error), error_type=type(error).__name__,
                                          config=job.config, campaign_task=campaign_task,
                                          state_preserved=store.has_state()))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            finally:
                store.release()
        if not unfinished:
            if failed:
                print(f'[worker] finished with {failed} failed jobs', flush=True)
                return 1
            print('[worker] all jobs are done', flush=True)
            return 0
        if not ran_a_job:
            # The rest are claimed. Stay until they finish or their owners die,
            # otherwise a job whose worker just crashed would be left behind.
            if not wait_for_others:
                return 1
            time.sleep(wait_seconds)
