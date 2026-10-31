"""Bound the whole authorized research process group, including inner supervisors.

The existing authorization and logging programs remain unchanged. An early
termination budget leaves time for TERM/KILL within the declared whole cap.
Partial inner evidence is retained; only a normal zero exit is completion.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
PHASE = Path(__file__).resolve().parents[1]
REPO = PHASE.parents[1] if PHASE.parent.name == "experiments_iclr" else PHASE.parent
DESTINATION = "anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru"


def confined(value):
    path = Path(os.path.abspath(value))
    if not path.is_relative_to(PHASE) or path == PHASE:
        raise ValueError("Path must be strictly inside this project phase")
    part = PHASE
    for name in path.relative_to(PHASE).parts:
        part /= name
        if part.is_symlink():
            raise ValueError("Symlink evidence/output paths are forbidden")
    return path.resolve()


def sha(path):
    with Path(path).open("rb") as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
        return h.hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def group_signal(pid, sig):
    try:
        os.killpg(pid, sig)
        return True
    except ProcessLookupError:
        return False


def execute_group(command, env, stream, cap, grace):
    """POSIX child session; no shell. Reserve grace inside the whole deadline."""
    begin = time.monotonic()
    child = subprocess.Popen(command, env=env, stdout=stream,
                             stderr=subprocess.STDOUT, start_new_session=True)
    interrupted = None
    timeout = False
    term_sent = kill_sent = False
    try:
        child.wait(timeout=max(0.001, cap - grace - (time.monotonic() - begin)))
    except subprocess.TimeoutExpired:
        timeout = True
    except BaseException as exc:
        interrupted = type(exc).__name__
    if timeout or interrupted:
        term_sent = group_signal(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=max(0.001, cap - (time.monotonic() - begin)))
        except subprocess.TimeoutExpired:
            pass
        # Kill surviving descendants even if the immediate supervisor exited.
        kill_sent = group_signal(child.pid, signal.SIGKILL)
        child.wait()
    else:
        # No background descendants are admitted as a successful completion.
        surviving = group_signal(child.pid, 0)
        if surviving:
            interrupted = "surviving_process_group_after_parent_exit"
            term_sent = group_signal(child.pid, signal.SIGTERM)
            kill_sent = group_signal(child.pid, signal.SIGKILL)
    return {"pid_and_pgid": child.pid, "child_exit_code": child.returncode,
            "timed_out": timeout, "interruption": interrupted,
            "TERM_sent": term_sent, "KILL_sent": kill_sent,
            "child_group_seconds": time.monotonic() - begin,
            "complete": not timeout and interrupted is None and child.returncode == 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--cap-seconds", type=float, required=True)
    parser.add_argument("--grace-seconds", type=float, default=5.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    cap, grace = args.cap_seconds, args.grace_seconds
    if not (math.isfinite(cap) and math.isfinite(grace) and 0 < grace < cap <= 28800):
        raise ValueError("Require a finite bounded cap with termination grace inside it")
    if Path.cwd().resolve() != REPO or os.environ.get("GNNM_SSH_DESTINATION") != DESTINATION:
        raise ValueError("Require exact authorized repository cwd and corrected route")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or os.environ.get("GNNM_PHASE_ROOT") != str(PHASE):
        raise ValueError("Require the existing repository cache/temp wrapper")
    interpreter = Path(os.path.abspath(command[0])) if command else None
    if len(command) < 7 or interpreter != REPO / ".venv/bin/python":
        raise ValueError("Use the exact repository venv interpreter")
    if command[1] != str(PHASE / "protocols/run_authorized_v2.py") or command[2] != "--output":
        raise ValueError("Bounded command must enter the corrected authorization supervisor")
    inner = confined(command[3])
    if "--" not in command[4:]:
        raise ValueError("Require an explicit authorization child argument boundary")
    boundary = command.index("--", 4)
    child = command[boundary + 1:]
    if len(child) < 2 or Path(os.path.abspath(child[0])) != interpreter:
        raise ValueError("Require the same exact repository child interpreter")
    script = confined(child[1])
    output, request = confined(args.output), confined(args.request)
    if output.exists() or inner.exists() or inner.is_relative_to(output) or output.is_relative_to(inner):
        raise ValueError("Outer and inner outputs must be new, separate phase directories")
    request_sha = sha(request)
    # Export/preflight children use --request. Scientific fits consume their
    # frozen whole-study root admission as --binding. Prefer the actual export
    # request when both flags are present and bind its exact bytes here.
    request_flag = "--request" if "--request" in child else "--binding"
    if child.count(request_flag) != 1 or Path(child[child.index(request_flag) + 1]).resolve() != request:
        raise ValueError("Child must explicitly consume this same root request")
    if child.count("--supervisor") != 1 or Path(child[child.index("--supervisor") + 1]).resolve() != inner:
        raise ValueError("Child must explicitly consume this same inner supervisor")
    output.mkdir(parents=True)
    begin = time.monotonic()
    start = {"schema": "gnnm-whole-process-bound-start-v1",
             "start_UTC": datetime.now(timezone.utc).isoformat(),
             "cwd": str(REPO), "phase": str(PHASE), "ssh_destination": DESTINATION,
             "outer_output": str(output), "inner_supervisor_directory": str(inner),
             "root_request": {"path": str(request.relative_to(PHASE)), "sha256": request_sha},
             "root_request_child_flag": request_flag,
             "whole_cap_seconds": cap, "termination_grace_seconds": grace,
             "normal_child_budget_seconds": cap - grace,
             "argv": command, "child_argv": child, "shell": False,
             "start_new_session": True, "source_sha256": sha(Path(__file__)),
             "authorization_source_sha256": sha(PHASE / "protocols/run_authorized_v2.py"),
             "logging_source_sha256": sha(PHASE / "protocols/run_logged.py"),
             "research_script_sha256": sha(script),
             "wrapper_sha256": sha(PHASE / "protocols/repo_env.sh")}
    write(output / "START.json", start)
    env = os.environ.copy()
    env["GNNM_BOUND_START_JSON"] = str(output / "START.json")
    try:
        with (output / "stdout_stderr.log").open("x") as stream:
            result = execute_group(command, env, stream, max(0.001, cap - (time.monotonic() - begin)), grace)
    except Exception as exc:
        result = {"complete": False, "timed_out": False,
                  "interruption": type(exc).__name__, "child_exit_code": None}
    evidence = {}
    for name in ("environment.json", "command.json", "completion.json", "authorization.json", "stdout_stderr.log"):
        path = inner / name
        evidence[name] = {"sha256": sha(path), "bytes": path.stat().st_size} if path.is_file() else None
    result.update({"schema": "gnnm-whole-process-bound-terminal-v1",
                   "START_sha256": sha(output / "START.json"),
                   "stdout_stderr_sha256": sha(output / "stdout_stderr.log"),
                   "terminal_UTC": datetime.now(timezone.utc).isoformat(),
                   "whole_supervised_seconds": time.monotonic() - begin,
                   "root_request_unchanged": sha(request) == request_sha,
                   "inner_evidence": evidence})
    result["within_whole_cap"] = result["whole_supervised_seconds"] <= cap
    if not result["within_whole_cap"]:
        result["complete"] = False
    if not result["root_request_unchanged"] or any(v is None for v in evidence.values()):
        result["complete"] = False
    write(output / "TERMINAL.json", result)
    print(json.dumps({"bounded_output": str(output), "complete": result["complete"],
                      "timed_out": result["timed_out"],
                      "whole_supervised_seconds": result["whole_supervised_seconds"]}), flush=True)
    raise SystemExit(0 if result["complete"] else 124 if result["timed_out"] else 1)


if __name__ == "__main__":
    main()
