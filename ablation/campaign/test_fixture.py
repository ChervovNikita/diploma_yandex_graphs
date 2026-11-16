"""CPU-only subprocess fixture for scheduler qualification."""

import argparse
import json
import os
from pathlib import Path
import signal
import time

from .store import atomic_json, locked


def update_registry(root, task_id, exclusive, entering):
    root.mkdir(parents=True, exist_ok=True)
    with locked(root / ".fixture.lock"):
        path = root / "registry.json"
        registry = json.loads(path.read_text()) if path.exists() else {"active": {}, "events": [], "violations": [], "peak": 0}
        active = registry["active"]
        if entering:
            if task_id in active:
                registry["violations"].append("duplicate:" + task_id)
            if active and (exclusive or any(value["exclusive"] for value in active.values())):
                registry["violations"].append("exclusive_overlap:" + task_id)
            active[task_id] = {"exclusive": exclusive, "pid": os.getpid()}
            registry["peak"] = max(registry["peak"], len(active))
        else:
            active.pop(task_id, None)
        registry["events"].append({"task": task_id, "entering": entering, "time": time.time()})
        atomic_json(path, registry)


def main(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        return 0
    attempt = int(os.environ.get("GNNM_CAMPAIGN_ATTEMPT", "1"))
    if args.fail_first and attempt == 1:
        return 7
    if args.fail:
        return 9
    if args.gate is not None:
        atomic_json(output / "decision.json", {"expand": args.gate == "true"})
    interrupted = False

    def stop(number, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, stop)
    progress_path = output / "progress.json"
    start = json.loads(progress_path.read_text())["completed"] if progress_path.exists() else 0
    task_id = os.environ.get("GNNM_CAMPAIGN_TASK", output.parent.name)
    update_registry(args.registry, task_id, args.exclusive, True)
    try:
        for step in range(start, args.steps):
            time.sleep(args.duration / args.steps)
            atomic_json(progress_path, {"completed": step + 1})
            if interrupted:
                return 130
        atomic_json(output / "result.json", {"completed": args.steps, "attempt": attempt, "resumed_at": start})
        return 0
    finally:
        update_registry(args.registry, task_id, args.exclusive, False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--exclusive", action="store_true")
    parser.add_argument("--fail", action="store_true")
    parser.add_argument("--fail-first", action="store_true")
    parser.add_argument("--gate", choices=("true", "false"))
    raise SystemExit(main(parser.parse_args()))
