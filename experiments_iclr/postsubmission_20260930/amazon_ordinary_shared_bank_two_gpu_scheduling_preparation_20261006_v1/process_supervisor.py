"""Disabled stdlib qualifier supervisor and reusable owned-child process helpers.

No Torch import, scientific run, source release or retry. Exact77 only.
"""
import time
STARTED = time.monotonic()
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import traceback
from types import SimpleNamespace

SOURCE_RELEASED = False
TARGET = "/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930"
ORDINARY = ("amazon_ordinary_shared_bank_own_pool_reference_preparation_20261006_v4/ordinary_reference.py", "5044a16f3f710aaf234057b115ab928d589af3a06a595940afbfa8876636c97b")
QUALIFIER = ("amazon_ordinary_shared_bank_first_order_qualification_preparation_20261006_v1/qualify.py", "d175dc6bf44886d874959cecd9833dc9f192f30b764bc7e9b938be83f2ad8af2")


def require(value, message):
    if not value: raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""): digest.update(block)
    return digest.hexdigest()


def load_ordinary(root):
    path = root / ORDINARY[0]
    require(sha(path) == ORDINARY[1], "Pinned disabled ordinary source differs")
    name = "_ordinary_owned_process_helpers"
    require(name not in sys.modules, "Fresh supervisor required")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    require(module.SOURCE_RELEASED is False, "Original V4 source gate must remain disabled")
    return module


def new_path(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "Relative assigned output required")
    path = root / relative
    require(not path.is_symlink() and path.resolve().is_relative_to(root) and not path.exists()
        and path.parent.is_dir() and path.parent.resolve().is_relative_to(root), "Fresh in-root assigned output required")
    return path.resolve()


def immutable_descriptor(root, path):
    path = Path(path)
    require(not path.is_symlink() and path.resolve().is_relative_to(root) and path.is_file()
        and path.stat().st_mode & 0o222 == 0, "Immutable in-root child result required")
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha(path)}


def validate_caps(caps, watchdog, grace, reap_timeout):
    keys = ("max_elapsed_seconds", "max_process_rss_bytes", "max_cuda_allocated_bytes", "max_cuda_reserved_bytes")
    require(all(type(caps[k]) in (int, float) and math.isfinite(caps[k]) and caps[k] > 0 for k in keys)
        and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in (watchdog, grace, reap_timeout))
        and watchdog > caps["max_elapsed_seconds"], "Root-frozen whole-child caps/watchdog/termination closure required")


def start_identity(pid):
    text = Path("/proc") / str(pid) / "stat"
    tail = text.read_text().rsplit(")", 1)[1].strip().split()
    return {"pid": pid, "starttime_ticks": int(tail[19]), "parent_pid": int(tail[1])}


def launch(root, owner, token, job_id, python, worker, argv, environment, children):
    """Exactly one fork/exec; stdlib caller has no numerical state to inherit."""
    require(not any(n == "torch" or n.startswith("torch.") for n in sys.modules), "Supervisor must stay pre-Torch")
    path = owner.verify(token)
    stdout_fd = os.open(path / (job_id + ".stdout.log"), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        stderr_fd = os.open(path / (job_id + ".stderr.log"), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except BaseException:
        os.close(stdout_fd); raise
    started = time.monotonic()
    try:
        pid = os.fork()
    except BaseException:
        os.close(stdout_fd); os.close(stderr_fd); raise
    if pid == 0:
        try:
            os.setsid(); os.chdir(root)
            os.dup2(stdout_fd, 1); os.dup2(stderr_fd, 2)
            os.close(stdout_fd); os.close(stderr_fd)
            os.execve(python, [python, "-B", str(worker), *argv], environment)
        except BaseException:
            traceback.print_exc(); os._exit(127)
    os.close(stdout_fd); os.close(stderr_fd)
    # Register the fork result before any later read/check can fail. Until wait4,
    # the direct child cannot be reaped or have its PID recycled by this owner.
    child = {"job_id": job_id, "pid": pid, "starttime_ticks": None, "parent_pid": os.getpid(),
        "started_monotonic": started, "python": python, "worker": str(worker), "argv": list(argv),
        "CUDA_VISIBLE_DEVICES": environment["CUDA_VISIBLE_DEVICES"], "wait4_closed": False,
        "external_timeout": False, "termination_signals": []}
    children.append(child)
    identity = start_identity(pid)
    require(identity["parent_pid"] == os.getpid(), "Launched direct-child identity differs")
    child.update(identity)
    return child


def poll_child(child):
    if child["wait4_closed"]: return True
    pid, raw_status, usage = os.wait4(child["pid"], os.WNOHANG)
    if pid == 0: return False
    require(pid == child["pid"], "Unexpected wait4 child")
    child.update(wait4_closed=True, child_exit_code=os.waitstatus_to_exitcode(raw_status), wait_status_raw=raw_status,
        wall_elapsed_seconds=time.monotonic() - child["started_monotonic"],
        wait4_peak_rss_bytes=int(usage.ru_maxrss) * 1024, wait4_user_seconds=usage.ru_utime, wait4_system_seconds=usage.ru_stime)
    return True


def signal_owned(child, signum):
    if child["wait4_closed"] or poll_child(child): return
    current = start_identity(child["pid"])
    if child["starttime_ticks"] is None:
        require(current["parent_pid"] == os.getpid(), "Incomplete launch no longer owns direct child")
        child["starttime_ticks"] = current["starttime_ticks"]
    require(current == {k: child[k] for k in ("pid", "starttime_ticks", "parent_pid")}
        and current["parent_pid"] == os.getpid(), "Refuse to signal a changed/unowned PID")
    group = os.getpgid(child["pid"])
    try:
        if group == child["pid"]: os.killpg(group, signum)
        else: os.kill(child["pid"], signum)
    except ProcessLookupError:
        require(poll_child(child), "Owned child vanished without wait4 closure")
    child["termination_signals"].append({"signal": int(signum), "elapsed_seconds": time.monotonic() - child["started_monotonic"]})


def watch(child, watchdog, grace, reap_timeout):
    """Nonblocking wait4 state machine, called repeatedly by one/two-child owner."""
    if poll_child(child): return True
    elapsed = time.monotonic() - child["started_monotonic"]
    if elapsed >= watchdog:
        child["external_timeout"] = True
        if not child["termination_signals"]: signal_owned(child, signal.SIGTERM)
        elif elapsed >= watchdog + grace and not any(row["signal"] == signal.SIGKILL for row in child["termination_signals"]):
            signal_owned(child, signal.SIGKILL)
        require(elapsed <= watchdog + grace + reap_timeout, "Owned child failed bounded wait4 closure after termination")
    return child["wait4_closed"]


def cleanup(children, grace, reap_timeout):
    """Terminate only this invocation's still-owned children and retain failures."""
    active = [c for c in children if not c["wait4_closed"]]
    errors = []
    for child in active:
        try: signal_owned(child, signal.SIGTERM)
        except BaseException as error: errors.append(str(error))
    started = time.monotonic()
    while active and time.monotonic() - started <= grace + reap_timeout:
        for child in list(active):
            try:
                if poll_child(child): active.remove(child)
                elif time.monotonic() - started >= grace and not any(r["signal"] == signal.SIGKILL for r in child["termination_signals"]):
                    signal_owned(child, signal.SIGKILL)
            except BaseException as error: errors.append(str(error)); active.remove(child)
        if active: time.sleep(0.25)
    errors.extend("No final wait4 closure for owned PID " + str(c["pid"]) for c in active)
    return errors


def resource_closure(child, result, caps):
    require(child["wait4_closed"] and child["child_exit_code"] == 0 and child["external_timeout"] is False,
        "Successful once-only child exit and final wait4 required")
    require(child["wall_elapsed_seconds"] <= caps["max_elapsed_seconds"]
        and child["wait4_peak_rss_bytes"] <= caps["max_process_rss_bytes"], "Whole-child wall/RSS cap exceeded")
    row = result["whole_process_resources"]
    require(all(type(row[m]) in (int, float) and math.isfinite(row[m]) and row[m] >= 0 and row[m] <= caps[c] for m, c in
        (("elapsed_seconds", "max_elapsed_seconds"), ("process_peak_rss_bytes", "max_process_rss_bytes"),
         ("cuda_peak_allocated_bytes", "max_cuda_allocated_bytes"), ("cuda_peak_reserved_bytes", "max_cuda_reserved_bytes"))),
        "Final child imports/setup/work/restoration resource closure failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-authorized", action="store_true"); parser.add_argument("--source-root")
    parser.add_argument("--scope"); parser.add_argument("--output")
    args = parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({"status": "DISABLED_OWNED_QUALIFIER_SUPERVISOR", "SOURCE_RELEASED": False, "numeric_imports": False})); return
    require(SOURCE_RELEASED is False, "Engineering supervisor keeps disabled source marker")
    require(args.source_root == TARGET and socket.gethostname() == "peptide", "Exact literal77 supervisor target required")
    require(args.scope and args.output, "Immutable exact engineering scope and new assigned output required")
    root = Path(args.source_root).resolve(); ordinary = load_ordinary(root)
    root, output, scope_path = ordinary.deliberate_paths(SimpleNamespace(source_root=args.source_root, output=args.output, admission=args.scope))
    require(scope_path.stat().st_mode & 0o222 == 0, "Immutable root engineering scope required")
    scope = json.loads(scope_path.read_text())
    require(scope["engineering_execution_authorized"] is True and scope["fit_authorized"] is False
        and scope["A_scoring"] is False and scope["VALID_TEST_access"] is False
        and scope["qualifier_source_review_approved"] is True and scope["ordinary_source_review_approved"] is True
        and scope["qualifier_supervisor_source_review_approved"] is True
        and scope["qualifier_supervisor_worker_sha256"] == sha(__file__)
        and scope["qualifier_worker_sha256"] == QUALIFIER[1] and scope["ordinary_worker_sha256"] == ORDINARY[1],
        "Explicit reviewed root engineering supervision only")
    require(output == new_path(root, scope["supervisor_output_relative"]), "Assigned supervisor output differs")
    qualifier_output = new_path(root, scope["qualification_output_relative"])
    require(output != qualifier_output, "Supervisor and child outputs must be distinct")
    qualifier = root / QUALIFIER[0]; require(sha(qualifier) == QUALIFIER[1], "Sealed qualifier source differs")
    python = scope["runtime"]["python_resolved"]
    require(Path(python).is_absolute() and str(Path(python).resolve()) == python and Path(python).is_file(), "Exact normal interpreter required")
    caps, watchdog = scope["resource_limits"], scope["external_watchdog_seconds"]
    grace, reap_timeout = scope["termination_grace_seconds"], scope["termination_reap_timeout_seconds"]
    validate_caps(caps, watchdog, grace, reap_timeout)
    uuid = scope["CUDA_VISIBLE_DEVICES"]
    require(isinstance(uuid, str) and uuid.startswith("GPU-") and "," not in uuid, "One root-assigned physical GPU required")
    owner = ordinary.CreatedOutput(output, root); token = owner.creator_token
    receipt = {"status": "RUNNING_OWNED_QUALIFIER", "supervisor_worker_sha256": sha(__file__), "scope_sha256": sha(scope_path),
        "qualifier_worker_sha256": QUALIFIER[1], "numeric_imports_in_supervisor": False, "fits_authorized": False,
        "A_scoring": False, "VALID_TEST_access": False, "launch_attempts": 0, "children": []}
    def save():
        receipt["supervisor_elapsed_seconds"] = time.monotonic() - STARTED
        ordinary.atomic(owner.verify(token) / "RESULT.json", receipt)
    old_handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    def interrupted(signum, frame): raise InterruptedError("Owned supervisor interrupted: " + str(signum))
    child, code = None, 1
    try:
        for signum in old_handlers: signal.signal(signum, interrupted)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid)
        receipt["launch_attempts"] = 1; save()
        child = launch(root, owner, token, "qualifier", python, qualifier,
            ["--execute-authorized", "--source-root", TARGET, "--scope", str(scope_path), "--output", str(qualifier_output)], environment, receipt["children"])
        save()
        while not watch(child, watchdog, grace, reap_timeout): time.sleep(0.25)
        save()
        result_path = qualifier_output / "RESULT.json"
        row = immutable_descriptor(root, result_path); result = json.loads(ordinary.bound(root, row).read_text())
        resource_closure(child, result, caps)
        require(result["status"] == "PASS_NORMAL_ORDINARY_FIRST_ORDER_STREAMED_RESOURCE_ONLY"
            and result["worker_sha256"] == QUALIFIER[1] and result["root_scope_sha256"] == sha(scope_path)
            and result["ordinary_first_order_supported"] is True and result["both_objectives_checked"] is True
            and result["same_checkpoint_context_and_RNG_checked"] is True
            and result["coupled_pool_gradient_and_Adam_update_parity_checked"] is True
            and result["stochastic_member_RNG_replay_checked"] is True
            and result["model_fits"] == 0 and result["persistent_updates"] == 0 and not result["restoration_errors"],
            "Exact successful bounded first-order result required")
        require(sha(scope_path) == receipt["scope_sha256"] and sha(qualifier) == QUALIFIER[1], "Scope/qualifier source changed")
        receipt.update(status="PASS_OWNED_QUALIFIER_WHOLE_CHILD_CLOSED", qualifier_result=row,
            child_exit_code=child["child_exit_code"], wait4_closed=True, root_caps=caps, ordinary_first_order_supported=True)
        code = 0
    except BaseException as error:
        receipt.update(status="FAIL_OWNED_QUALIFIER", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        # Ignore repeated user termination signals only during bounded owned cleanup.
        for signum in old_handlers: signal.signal(signum, signal.SIG_IGN)
        errors = cleanup(receipt["children"], grace, reap_timeout)
        receipt["cleanup_errors"] = errors
        if errors: receipt.update(status="FAIL_OWNED_QUALIFIER", ordinary_first_order_supported=False); code = 1
        for signum, handler in old_handlers.items(): signal.signal(signum, handler)
        if code: receipt["ordinary_first_order_supported"] = False
        try: save(); owner.freeze(token)
        except BaseException as error:
            code = 1; receipt.update(status="FAIL_OWNED_QUALIFIER", ordinary_first_order_supported=False, output_finalization_error=str(error))
            try: save(); owner.freeze(token)
            except BaseException as recovery_error: receipt["output_failure_receipt_error"] = str(recovery_error)
    print(json.dumps({"status": receipt["status"], "child_exit_code": receipt.get("child_exit_code"), "error": receipt.get("error")}))
    raise SystemExit(code)


if __name__ == "__main__": main()
