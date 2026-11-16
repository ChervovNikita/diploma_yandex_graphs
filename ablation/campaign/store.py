"""Atomic state and kernel-owned locks on the shared campaign filesystem.

Lock files retain their inode throughout the campaign. A child inherits its
task lock descriptor, so killing its supervisor cannot allow a second process
to mutate the same fit while the original child remains alive. The filesystem
must provide working cross-host POSIX advisory locks.
"""

import fcntl
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .schema import canonical_hash, relative_path, validate_plan


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def acquire_lock(path, blocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    # Retry nonblocking requests locally. Blocking NFSv4 flock requests can
    # remain queued for a lease-retry interval after short locks are released.
    while True:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return descriptor
        except BlockingIOError:
            if not blocking:
                os.close(descriptor)
                return None
            time.sleep(0.05)
        except BaseException:
            os.close(descriptor)
            raise


def release_lock(descriptor):
    # Closing preserves locks held by an inherited descriptor in a live child.
    os.close(descriptor)


@contextmanager
def locked(path):
    descriptor = acquire_lock(path, blocking=True)
    try:
        yield descriptor
    finally:
        release_lock(descriptor)


class CampaignStore:
    def __init__(self, root, plan):
        self.root = Path(root).resolve()
        self.plan = plan
        self.tasks = validate_plan(plan)
        self.plan_hash = canonical_hash(plan)
        self._states = {task_id: {"status": "pending", "attempts": 0,
                                 "failed_attempts": 0, "task_id": task_id}
                        for task_id in self.tasks}
        self._event_offset = 0

    def prepare(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with locked(self.root / ".prepare.lock"):
            path = self.root / "plan.json"
            if path.exists():
                existing = json.loads(path.read_text())
                if canonical_hash(existing) != self.plan_hash:
                    raise ValueError("Campaign root already belongs to a different immutable plan")
            else:
                atomic_json(path, self.plan)
            atomic_json(self.root / "plan_identity.json", {
                "schema": "gnnm-campaign-identity-v1",
                "campaign_id": self.plan["campaign_id"],
                "sha256": self.plan_hash,
            })

    def task_dir(self, task_id):
        if task_id not in self.tasks:
            raise KeyError(task_id)
        return self.root / "tasks" / relative_path(task_id)

    def _read_events(self):
        path = self.root / "events.jsonl"
        if not path.exists():
            return
        with path.open("rb") as stream:
            stream.seek(self._event_offset)
            while True:
                position = stream.tell()
                line = stream.readline()
                if not line or not line.endswith(b"\n"):
                    self._event_offset = position
                    return
                state = json.loads(line)
                if state.get("plan_sha256") != self.plan_hash:
                    raise ValueError("Status journal belongs to another plan")
                if state["task_id"] not in self.tasks:
                    raise ValueError("Status journal names an unknown task")
                self._states[state["task_id"]] = state
                self._event_offset = stream.tell()

    def read_state(self, task_id):
        self._read_events()
        return dict(self._states[task_id])

    def write_state(self, task_id, **fields):
        state = {
            "schema": "gnnm-campaign-task-state-v1",
            "task_id": task_id,
            "plan_sha256": self.plan_hash,
            "updated": time.time(),
            **fields,
        }
        # The append-only journal is canonical. Per-task state.json is its
        # inspectable projection. Incremental readers avoid a full NFS tree scan.
        with locked(self.root / ".events.lock"):
            path = self.root / "events.jsonl"
            with path.open("a+b") as stream:
                stream.seek(0, os.SEEK_END)
                end = stream.tell()
                if end:
                    stream.seek(end - 1)
                    if stream.read(1) != b"\n":
                        # Preserve complete events after a torn final append.
                        position = end
                        while position > 0:
                            start = max(0, position - 65536)
                            stream.seek(start)
                            chunk = stream.read(position - start)
                            newline = chunk.rfind(b"\n")
                            if newline >= 0:
                                stream.truncate(start + newline + 1)
                                break
                            position = start
                        else:
                            stream.truncate(0)
                stream.seek(0, os.SEEK_END)
                stream.write(json.dumps(state, sort_keys=True).encode() + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
        atomic_json(self.task_dir(task_id) / "state.json", state)
        self._read_events()
        return state

    def task_lock(self, task_id):
        return acquire_lock(self.task_dir(task_id) / ".ownership.lock")

    def snapshot(self):
        self._read_events()
        return dict(self._states)

    def required_outputs_exist(self, task):
        directory = self.task_dir(task["id"])
        return all((directory / relative_path(name)).exists()
                   for name in task.get("required_outputs", []))
