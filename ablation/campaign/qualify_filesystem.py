"""Check shared cross-host locks with CPU-only contenders under the campaign repo.

Run the same command from two Slurm nodes against the same previously unused
--root. No CUDA initialization or training occurs. Every participant waits for
the declared participant count, then checks that flock excludes other writers.
"""

import argparse
import json
import os
from pathlib import Path
import socket
import time
import uuid

from .store import acquire_lock, atomic_json, release_lock


def wait_for_count(directory, count, deadline):
    while len(list(directory.glob("*.json"))) < count:
        if time.monotonic() > deadline:
            raise TimeoutError(f"Expected {count} participants in {directory}")
        time.sleep(0.05)


def main(args):
    if args.participants < 2 or args.rounds < 1 or args.hold_seconds <= 0:
        raise ValueError("Use at least two participants, positive rounds and hold duration")
    root = args.root.resolve()
    repository = Path(__file__).resolve().parents[2]
    if not root.is_relative_to(repository):
        raise ValueError("Filesystem qualification root must be inside the repository")
    ready = root / "ready"
    results = root / "results"
    ready.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    identity = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    deadline = time.monotonic() + args.timeout_seconds
    atomic_json(ready / f"{identity}.json", {"host": socket.gethostname(), "pid": os.getpid()})
    wait_for_count(ready, args.participants, deadline)
    violations = []
    contended = 0
    completed = 0
    for index in range(args.rounds):
        descriptor = None
        while descriptor is None:
            descriptor = acquire_lock(root / "shared.lock")
            if descriptor is None:
                contended += 1
                if time.monotonic() > deadline:
                    raise TimeoutError("Lock qualification timed out")
                time.sleep(0.01)
        sentinel = root / "exclusive-writer"
        owns_sentinel = False
        try:
            try:
                exclusive = os.open(sentinel, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                os.close(exclusive)
                owns_sentinel = True
            except FileExistsError:
                violations.append({"round": index, "reason": "overlapping_lock_owners"})
            time.sleep(args.hold_seconds)
            completed += 1
        finally:
            if owns_sentinel:
                sentinel.unlink()
            release_lock(descriptor)
        time.sleep(0.01)
    atomic_json(results / f"{identity}.json", {
        "identity": identity, "host": socket.gethostname(),
        "completed_rounds": completed, "contention_count": contended,
        "violations": violations,
    })
    wait_for_count(results, args.participants, deadline)
    records = [json.loads(path.read_text()) for path in sorted(results.glob("*.json"))]
    passed = len(records) == args.participants and not any(record["violations"] for record in records)
    report = {
        "passed": passed, "participants": len(records),
        "hosts": sorted({record["host"] for record in records}),
        "contention_count": sum(record["contention_count"] for record in records),
        "records": records,
    }
    atomic_json(root / "qualification.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, required=True)
    result.add_argument("--participants", type=int, required=True)
    result.add_argument("--rounds", type=int, default=8)
    result.add_argument("--hold-seconds", type=float, default=0.1)
    result.add_argument("--timeout-seconds", type=float, default=120)
    return result


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
