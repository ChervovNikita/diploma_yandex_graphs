"""Dedicated OGB author subprocess worker; TEST-bearing source bytes stay quarantined."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import time

from ogb_author_cli_spec import (AUTHOR_FAMILY, ENVIRONMENT, load_ogb_author_spec,
                                 validate_ogb_author_spec)
from author_cli_spec import register_author_spec, verify_receipt
from registry import Registry
from run_author_cli import atomic, file_receipt, require_master_approval, verify_allocation
from run_study import owner_identity, require_approval
from spec import canonical, confined, fingerprint

NUMBER = rb"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
EPOCH = re.compile(rb"^=====Epoch ([0-9]+)$")
SEEDED_START = re.compile(rb"^DECLARED_SEED_PROCESS_START ([0-9]+)$")
SEEDED_RETURN = b"DECLARED_SEED_PROCESS_RETURN 0"
# The TEST substring is opaque. No regex capture, JSON/ast decoding, float
# conversion or dictionary construction touches any TEST-bearing token.
TRAIN_VALID = re.compile(rb"^\{'Train': \{'rocauc': " + NUMBER
                         + rb"\}, 'Validation': \{'rocauc': " + NUMBER
                         + rb"\}, 'Test': \{[^\r\n]*\}\}$")
BEST_VALID = re.compile(rb"^Best validation score: " + NUMBER + rb"$")


class OGBProgress:
    def __init__(self, seed, max_train_batches):
        self.seed = seed
        self.max_train_batches = max_train_batches
        self.started = False
        self.returned = False
        self.finished = False
        self.epoch = 0
        self.completed_epochs = 0
        self.last_train_valid = None
        self.best_valid = None
        self.best_epoch = None
        self.summary_valid_seen = False
        self.anomalies = []

    def anomaly(self, reason):
        self.anomalies.append(reason)
        return None

    def feed(self, line):
        line = line.rstrip(b"\r\n")
        # Ignore native Test score and unrelated lines without inspecting
        # their payload. Only the strictly defined TRAIN/VALID prefix is read.
        start = SEEDED_START.fullmatch(line)
        if start:
            if self.started or int(start[1]) != self.seed:
                return self.anomaly("seed_start_identity_or_order")
            self.started = True
            return {"event": "declared_seed_started", "seed": self.seed}
        epoch = EPOCH.fullmatch(line)
        if epoch:
            number = int(epoch[1])
            if (not self.started or self.finished or self.returned or self.epoch != self.completed_epochs
                    or number != self.completed_epochs + 1 or number > 100):
                return self.anomaly("noncontiguous_or_excess_epoch")
            self.epoch = number
            return {"event": "native_epoch_started", "epoch": number}
        metrics = TRAIN_VALID.fullmatch(line)
        if metrics:
            if not self.started or self.finished or self.returned or self.epoch != self.completed_epochs + 1:
                return self.anomaly("TRAIN_VALID_without_unique_epoch")
            train, valid = float(metrics[1]), float(metrics[2])
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in [train, valid]):
                return self.anomaly("invalid_TRAIN_VALID_diagnostic")
            self.completed_epochs += 1
            if self.best_valid is None or valid > self.best_valid:
                self.best_valid, self.best_epoch = valid, self.epoch
            self.last_train_valid = {"event": "native_TRAIN_VALID_epoch", "epoch": self.epoch,
                                     "TRAIN_rocauc": train, "VALID_rocauc": valid}
            return self.last_train_valid
        if line.startswith(b"{'Train':"):
            return self.anomaly("invalid_TRAIN_VALID_log_grammar")
        if line == b"Finished training!":
            if self.finished or self.returned or self.completed_epochs != 100 or self.epoch != 100:
                return self.anomaly("native_finish_before_full_budget")
            self.finished = True
            return {"event": "native_100_epochs_finished"}
        best = BEST_VALID.fullmatch(line)
        if best:
            value = float(best[1])
            if (not self.finished or self.summary_valid_seen or self.returned
                    or not math.isfinite(value) or value != self.best_valid):
                return self.anomaly("validation_curve_summary_mismatch")
            self.summary_valid_seen = True
            return {"event": "native_validation_selection", "VALID_rocauc": value,
                    "selected_epoch": self.best_epoch, "ties": "first_np_argmax",
                    "model_checkpoint": False}
        if line == SEEDED_RETURN:
            if self.returned or not self.finished or not self.summary_valid_seen:
                return self.anomaly("seed_return_before_native_completion")
            self.returned = True
            return {"event": "declared_seed_process_returned"}
        return None

    def costs(self, *, child_started, exit_code):
        complete = self.started and self.returned and self.finished and self.completed_epochs == 100 and not self.anomalies
        return {"author_cli_processes_started": int(child_started), "declared_independent_seed_trajectories": 1,
                "completed_native_epochs_observed": self.completed_epochs,
                "registered_maximum_epochs": 100,
                "registered_maximum_optimizer_updates": 100 * self.max_train_batches,
                "optimizer_update_upper_bound_for_observed_completed_epochs": self.completed_epochs * self.max_train_batches,
                "actual_optimizer_updates": 0 if not child_started else None,
                "actual_optimizer_updates_exact": not child_started,
                "native_singleton_batch_skip_accounting_unresolved": bool(child_started),
                "native_train_validation_test_loader_passes_observed": self.completed_epochs * 3,
                "complete_native_process_observation": bool(complete),
                "protocol_observation_anomalies": sorted(set(self.anomalies)),
                "cuda_peak_bytes": None, "cuda_peak_measured": False,
                "exit_code": exit_code}


def preflight_ogb_group(root, value, run):
    validate_ogb_author_spec(value)
    root = Path(root).resolve(); cli = run["cli"]
    repository = Path(cli["execution_repository"]).resolve()
    if not root.is_relative_to(repository):
        raise ValueError("OGB execution repository must contain the project")
    python = Path(cli["python_executable"])
    if not python.is_file() or not python.resolve().is_relative_to(repository):
        raise ValueError("OGB Python must remain inside the declared repository runtime")
    for row in cli["source_files"] + cli["raw_files"]:
        verify_receipt(root, row)
    for row in cli["runtime_provider_files"]:
        verify_receipt(repository, row)
    work = confined(root, cli["cwd"])
    if work.is_symlink() or not work.is_dir():
        raise ValueError("Each OGB process requires its own real private workdir")
    processed = confined(root, cli["fresh_processed_dir"])
    if processed.is_symlink() or not processed.is_dir() or list(processed.iterdir()):
        raise ValueError("Native OGB processing starts from a fresh empty per-process directory")
    for relative in cli["quarantine"].values():
        if confined(root, relative).exists():
            raise ValueError("Retain prior OGB logs and terminal failures; no rerun/overwrite")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None); env.pop("PYTHONHOME", None)
    env.update(ENVIRONMENT)
    check_env = dict(env, CUDA_VISIBLE_DEVICES="")
    script = "import importlib.metadata as m,json,sys;expected=json.loads(sys.argv[1]);assert {n:m.version(n) for n in expected}==expected"
    check = subprocess.run([str(python), "-B", "-c", script, canonical(cli["runtime_versions"])],
                           cwd=work, env=check_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    if check.returncode:
        raise ValueError("Declared OGB runtime package versions changed")
    return [str(python), *cli["argv"]], work, env


def process_identity(pid):
    stat = Path("/proc") / str(pid) / "stat"
    if stat.exists():
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
            return {"pid": pid, "pgid": int(fields[2]), "start_ticks": int(fields[19])}
        except (OSError, ValueError, IndexError):
            return None
    try:
        return {"pid": pid, "pgid": os.getpgid(pid)}
    except ProcessLookupError:
        return None


def owned_cuda_pid_absent(pid):
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
    if result.returncode:
        return False
    return str(pid) not in [line.strip() for line in result.stdout.splitlines()]


def execute_claimed_ogb_group(root, value, run, registry, destination, owner, admission,
                              *, cuda_absence_check=owned_cuda_pid_absent):
    """Only execute an already admitted/claimed process; engineering fixtures use opaque fake CLIs."""
    command, work, env = admission
    registered = registry.run(value["study_id"], run["run_id"])
    if registered["state"] != "running" or registered["owner"] != owner or registered["contract"] != run:
        raise ValueError("OGB subprocess needs this exact registered process claim")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    tick = time.monotonic(); deadline = tick + run["config"]["wall_seconds"]
    deadline_utc = datetime.fromtimestamp(time.time() + run["config"]["wall_seconds"], timezone.utc).isoformat()
    observer = OGBProgress(run["seed"], run["config"]["max_train_batches_per_epoch"])
    process = child_owner = None; reason = None
    stop = {"signal": None}; old_handlers = {}
    for signum in [signal.SIGTERM, signal.SIGINT]:
        old_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, lambda received, frame: stop.update(signal=received))
    stdout_path = confined(root, run["cli"]["quarantine"]["stdout"])
    stderr_path = confined(root, run["cli"]["quarantine"]["stderr"])
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    def stop_child():
        if process is None or process.poll() is not None:
            return
        current = process_identity(process.pid)
        if current is None or current != child_owner:
            raise ValueError("Owned OGB child identity changed; never signal another process")
        os.killpg(child_owner["pgid"], signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if process_identity(process.pid) != child_owner:
                raise ValueError("Owned OGB child identity changed before escalation")
            os.killpg(child_owner["pgid"], signal.SIGKILL)
            process.wait(timeout=5)
    result = None
    try:
        with stdout_path.open("xb") as raw_stdout, stderr_path.open("xb") as raw_stderr, (destination / "TRAIN_VALID_PROGRESS.jsonl").open("x") as progress:
            process = subprocess.Popen(command, cwd=work, env=env, stdout=subprocess.PIPE, stderr=raw_stderr,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            child_owner = process_identity(process.pid)
            if child_owner is None or child_owner["pgid"] != process.pid:
                raise ValueError("Cannot bind fresh OGB child PID/PGID/birth custody")
            start = {"owner": owner, "child_owner": child_owner, "command": command,
                     "cwd": str(work), "declared_environment": run["cli"]["environment"],
                     "deadline_UTC": deadline_utc, "study_hash": fingerprint(value), "run_hash": fingerprint(run)}
            atomic(destination / "START.json", start)
            atomic(destination / "OWNER_START.json", start)
            registry.event(value["study_id"], run["run_id"], "ogb_author_child_started", start)
            selector = selectors.DefaultSelector(); selector.register(process.stdout, selectors.EVENT_READ)
            pending, eof = b"", False
            try:
                while not eof or process.poll() is None:
                    if stop["signal"] is not None or time.monotonic() >= deadline:
                        reason = "interrupted_signal" if stop["signal"] is not None else "declared_wall_deadline"
                        stop_child()
                    for key, _ in selector.select(timeout=.2):
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fileobj); eof = True; continue
                        raw_stdout.write(chunk); raw_stdout.flush(); pending += chunk
                        while b"\n" in pending:
                            line, pending = pending.split(b"\n", 1)
                            event = observer.feed(line)
                            if event is not None:
                                progress.write(canonical(event) + "\n")
                                progress.flush()
                                safe = {"last_TRAIN_VALID": observer.last_train_valid,
                                        "cost": observer.costs(child_started=True, exit_code=None)}
                                atomic(destination / "PROGRESS.json", safe)
                                registry.event(value["study_id"], run["run_id"], "ogb_TRAIN_VALID_progress", safe)
                        if len(pending) > 1024 * 1024:
                            pending = b""; observer.anomaly("oversize_unparsed_source_line")
                if pending:
                    event = observer.feed(pending)
                    if event is not None:
                        progress.write(canonical(event) + "\n")
                process.wait(timeout=5)
            finally:
                selector.close(); process.stdout.close()
        costs = observer.costs(child_started=True, exit_code=process.returncode)
        identity_absent = process_identity(process.pid) is None
        cuda_absent = bool(cuda_absence_check(process.pid))
        success = (process.returncode == 0 and costs["complete_native_process_observation"]
                   and reason is None and identity_absent and cuda_absent)
        result = {"status": "complete" if success else "failed", "owner": owner, "child_owner": child_owner,
                  "command": command, "deadline_UTC": deadline_utc, "exit_code": process.returncode,
                  "terminal_status": "COMPLETE_EXIT_0" if success else "FAILED_RETAINED",
                  "child_pid_absent": identity_absent, "owned_cuda_pid_absent": cuda_absent,
                  "wall_seconds": time.monotonic() - tick,
                  "failure_type": None if success else reason or "native_process_protocol_or_terminal_custody_failure",
                  "cost": costs, "quarantine": {"stdout": file_receipt(root, stdout_path), "stderr": file_receipt(root, stderr_path)},
                  "author_TEST_outcomes_interpreted": False, "whole_author_release_required": True,
                  "resume_supported": False, "published_score_equality_claim": False}
    except BaseException as error:
        try:
            stop_child()
        except BaseException:
            pass
        result = {"status": "failed", "terminal_status": "FAILED_RETAINED", "failure_type": type(error).__name__,
                  "owner": owner, "child_owner": child_owner, "command": command,
                  "deadline_UTC": deadline_utc, "wall_seconds": time.monotonic() - tick,
                  "child_pid_absent": process is None or process_identity(process.pid) is None,
                  "owned_cuda_pid_absent": False if process is not None else True,
                  "cost": observer.costs(child_started=process is not None, exit_code=process.returncode if process else None),
                  "author_TEST_outcomes_interpreted": False, "whole_author_release_required": True, "resume_supported": False}
        # Raw exceptions/tracebacks may contain TEST values; never publish them.
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    atomic(destination / "RESULT.json", result)
    atomic(destination / "OWNER_END.json", result)
    registry.finish(value["study_id"], run["run_id"], result, success=result["status"] == "complete")
    return result


def execute(root, study_path, run_id, registry_path, output, approval_path, device="cuda:0", *, preflight_only=False):
    verify_allocation()  # Before any project access.
    root = Path(root).resolve()
    for path in [study_path, registry_path, output, approval_path]:
        if path and not Path(path).resolve().is_relative_to(root):
            raise ValueError("OGB spec/registry/output/approval must remain inside the project")
    value, digest = load_ogb_author_spec(study_path, root)
    rows = [r for r in value["runs"] if r["run_id"] == run_id]
    if len(rows) != 1 or device != "cuda:0":
        raise ValueError("One bound OGB process on allocation visible cuda0 is required")
    run = rows[0]; admission = preflight_ogb_group(root, value, run)
    if preflight_only:
        return {"preflight": "passed", "author_family": AUTHOR_FAMILY, "run_id": run_id,
                "spec_hash": digest, "scientific_training_runs": 0, "GO_created": False}
    if not approval_path:
        raise ValueError("Explicit user GO and whole-master source-bound admission are required")
    resource = run["config"]["resource_evidence"]
    if not isinstance(resource, dict) or resource.get("status") != "ROOT_QUALIFIED" or not resource.get("receipt"):
        raise ValueError("OGB source resource/protocol/architecture qualification is still pending")
    verify_receipt(root, resource["receipt"])
    approval = require_approval(approval_path, digest)
    gate = require_master_approval(root, value, digest, run, approval)
    registry = Registry(registry_path)
    try:
        register_author_spec(registry, value)
        owner = owner_identity(); owner.update(kind="author_cli_group", author_family=AUTHOR_FAMILY,
            study_hash=digest, run_hash=fingerprint(run), gate=gate)
        registry.claim(value["study_id"], run_id, owner)
        return execute_claimed_ogb_group(root, value, run, registry, Path(output) / value["study_id"] / run_id, owner, admission)
    finally:
        registry.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["project-root", "study", "run-id", "registry", "output"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--approval"); parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not args.preflight_only and not args.approval:
        parser.error("explicit GO is required; source preparation is disabled")
    result = execute(args.project_root, args.study, args.run_id, args.registry, args.output,
                     args.approval, args.device, preflight_only=args.preflight_only)
    print(canonical(result))
    raise SystemExit(0 if result.get("status", "complete") == "complete" else 1)


if __name__ == "__main__":
    main()
