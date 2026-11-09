"""Crash-safe per-job storage for runs on an unstable cluster.

Every job owns one directory:

    config.json    the job definition; a restart with a different one is refused
    state.pt       latest training state (model, optimizer, best model, RNG)
    state.prev.pt  the state before it, used if state.pt is unreadable
    claim          present while a worker owns the job, kept fresh by heartbeat
    failures.log   one line per Python exception raised by the job
    done.json      the result; its presence is the only meaning of "finished"

A job killed at any moment is continued from its last state by whichever
worker picks it up next.
"""

import hashlib
import json
import os
import random
import socket
import time
import uuid
from pathlib import Path

import numpy as np
import torch


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _temporary_path(path):
    return path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open('wb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(path.parent)


def atomic_write_json(value, path):
    path = Path(path)
    temporary = _temporary_path(path)
    try:
        with temporary.open('w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(path.parent)


def capture_rng(device):
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if device.type == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state(device).cpu()
    return state


def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'].cpu(), device)


def source_fingerprint(paths):
    """SHA-256 over source files, stored with each result for provenance."""
    digest = hashlib.sha256()
    for path in sorted(Path(p) for p in paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class JobStore:
    def __init__(self, job_dir):
        self.dir = Path(job_dir)
        self.config_path = self.dir / 'config.json'
        self.state_path = self.dir / 'state.pt'
        self.previous_state_path = self.dir / 'state.prev.pt'
        self.claim_path = self.dir / 'claim'
        self.failures_path = self.dir / 'failures.log'
        self.done_path = self.dir / 'done.json'

    def ensure_config(self, config):
        """Bind the directory to one job definition."""
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.config_path.exists():
            existing = json.loads(self.config_path.read_text())
            if existing != config:
                raise ValueError(
                    f'{self.dir} belongs to a different job: {existing} != {config}. '
                    'Use a new --root for a changed protocol.'
                )
            return
        atomic_write_json(config, self.config_path)

    def is_done(self):
        return self.done_path.exists()

    def read_done(self):
        return json.loads(self.done_path.read_text())

    def write_done(self, result):
        self.dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(result, self.done_path)

    def save_state(self, state):
        self.dir.mkdir(parents=True, exist_ok=True)
        temporary = _temporary_path(self.state_path)
        try:
            with temporary.open('wb') as stream:
                torch.save(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            if self.state_path.exists():
                os.replace(self.state_path, self.previous_state_path)
            os.replace(temporary, self.state_path)
        finally:
            temporary.unlink(missing_ok=True)
        _fsync_directory(self.dir)

    def load_state(self):
        """Return the newest readable state, or None for a fresh start."""
        for path in (self.state_path, self.previous_state_path):
            if not path.exists():
                continue
            try:
                # These files are written only by this module.
                return torch.load(path, map_location='cpu', weights_only=False)
            except Exception as error:  # A truncated file can raise many types.
                print(f'[resume] ignoring unreadable {path}: {error!r}', flush=True)
        return None

    def clear_state(self):
        self.state_path.unlink(missing_ok=True)
        self.previous_state_path.unlink(missing_ok=True)

    def has_state(self):
        return self.state_path.exists() or self.previous_state_path.exists()

    def claim(self, stale_seconds):
        """Take exclusive ownership; False if another live worker has the job.

        Two workers can still overlap if one stalls past ``stale_seconds`` and
        then wakes up. That wastes GPU time only: state and done files are
        replaced atomically and either run is a valid result.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        owner = f'{socket.gethostname()} pid={os.getpid()} time={time.time()}\n'
        for _ in range(2):
            try:
                descriptor = os.open(self.claim_path,
                                     os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                if not self.claim_is_stale(stale_seconds):
                    return False
                aside = self.dir / f'.claim.stale.{uuid.uuid4().hex}'
                try:
                    os.replace(self.claim_path, aside)
                except FileNotFoundError:
                    return False
                aside.unlink(missing_ok=True)
                continue
            with os.fdopen(descriptor, 'w') as stream:
                stream.write(owner)
            return True
        return False

    def claim_is_stale(self, stale_seconds):
        try:
            modified = self.claim_path.stat().st_mtime
        except FileNotFoundError:
            return True
        return time.time() - modified > stale_seconds

    def is_claimed(self, stale_seconds):
        return self.claim_path.exists() and not self.claim_is_stale(stale_seconds)

    def heartbeat(self):
        try:
            os.utime(self.claim_path)
        except FileNotFoundError:
            pass

    def release(self):
        self.claim_path.unlink(missing_ok=True)

    def record_failure(self, message):
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.failures_path.open('a') as stream:
            stream.write(' '.join(message.split()) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def failure_count(self):
        if not self.failures_path.exists():
            return 0
        with self.failures_path.open() as stream:
            return sum(1 for _ in stream)
