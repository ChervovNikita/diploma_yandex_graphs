"""Execute one admitted fitted screen on the authorized allocation.

No run is admitted by this source file. Root's separate immutable admission
binds the scientific freeze, complete reviewed sources and remaining budget.
"""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
REPO = "/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs"
REMOTE = REPO + "/experiments_iclr/postsubmission_20260930"
DEST = "anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru"
PACKET = "new_learning_gap_independent_v1/ar_fitted_screen_source_v1"
SSH = ["ssh", "-p", "2222", "-i", "/Users/alex/.ssh/mlspace__private_key_anogena.txt",
       "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "UpdateHostKeys=no",
       "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=15", DEST]


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    started = time.perf_counter()
    admission_path = HERE / "EXECUTION_ADMISSION_v1.json"
    admission = json.loads(admission_path.read_text())
    assert admission["status"] == "admitted_by_root"
    for row in admission["bindings"]:
        assert sha256((PHASE / row["path"]).read_bytes()).hexdigest() == row["sha256"], row["path"]
    prior_path = PHASE / "protocols/ar_resource_compatibility_v3_root_v1/CUMULATIVE_BOUND_AFTER_RESOURCE_run01.json"
    prior_bytes = prior_path.read_bytes()
    prior = json.loads(prior_bytes)
    envelope, diagnostic_reserve = 3000, 60
    assert prior["combined_phase_seconds"] + envelope <= prior["phase_cap_seconds"]
    assert prior["diagnostic_seconds"] + diagnostic_reserve <= prior["diagnostic_cap_seconds"]
    assert not (PHASE / "protocols/FETCH_AUTHORIZED_78.json").exists()
    assert not (PHASE / PACKET / "runs").exists()
    write(HERE / "FITTED_START_run01.json", {
        "UTC": datetime.now(timezone.utc).isoformat(),
        "admission_sha256": sha256(admission_path.read_bytes()).hexdigest(),
        "whole_operation_envelope_seconds": envelope,
        "diagnostic_reservation_seconds": diagnostic_reserve,
        "CPU_only": True,
        "stage_bounds_seconds": {"upload": 35, "supervision": 2900, "fetch": 35,
                                 "orchestration_reserve": 30},
        "ssh_destination": DEST, "retry": False})
    terminal = {"status": "started", "stages": []}
    supervision_attempted, fetch_attempted = False, False
    def fetch_once():
        nonlocal fetch_attempted
        fetch_attempted = True
        remaining = envelope - (time.perf_counter() - started)
        if remaining <= 0:
            raise TimeoutError("Whole-coordinator envelope exhausted before retention")
        fetch = [sys.executable, "-B", str(PHASE / "protocols/fetch_authorized_evidence_v2.py"),
            "--receipt", str(PHASE / "protocols/FETCH_AUTHORIZED_78.json"),
            "protocols/ar_fitted_screen_root_v1/fitted_supervision_run01",
            PACKET + "/runs/admitted_run_v1"]
        with (HERE / "FITTED_FETCH_stdout_run01.txt").open("x") as out:
            fetched = subprocess.run(fetch, cwd=PHASE, stdout=out, stderr=subprocess.STDOUT,
                                     timeout=min(35, remaining))
        terminal["stages"].append({"stage": "fetch", "returncode": fetched.returncode})
        return fetched
    try:
        upload = [sys.executable, "-B", str(PHASE / "protocols/upload_authorized_sources_v1.py"),
            "--receipt", str(HERE / "FITTED_UPLOAD_run01.json"),
            "--archive", str(HERE / "FITTED_UPLOAD_run01.tar.gz"), *admission["upload_inventory"]]
        with (HERE / "FITTED_UPLOAD_stdout_run01.txt").open("x") as out:
            completed = subprocess.run(upload, cwd=PHASE, stdout=out, stderr=subprocess.STDOUT, timeout=35)
        terminal["stages"].append({"stage": "upload", "returncode": completed.returncode})
        assert completed.returncode == 0, "Source transfer failed; child not started"
        supervisor = REMOTE + "/protocols/ar_fitted_screen_root_v1/fitted_supervision_run01"
        argv = [REPO + "/.venv/bin/python", REMOTE + "/protocols/run_authorized_v2.py",
            "--output", supervisor, "--", REPO + "/.venv/bin/python",
            REMOTE + "/" + PACKET + "/launch_guard.py",
            "--root-freeze", REMOTE + "/protocols/ar_fitted_screen_root_v1/SCIENTIFIC_FREEZE_v1.json",
            "--root-freeze-sha256", admission["scientific_freeze_sha256"],
            "--root-admission", REMOTE + "/protocols/ar_fitted_screen_root_v1/EXECUTION_ADMISSION_v1.json",
            "--run-name", "admitted_run_v1"]
        command = ("cd " + shlex.quote(REPO)
            + " && source experiments_iclr/postsubmission_20260930/protocols/repo_env.sh"
            + " && export GNNM_SSH_DESTINATION=" + shlex.quote(DEST)
            + " CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"
            + " && " + shlex.join(argv))
        with (HERE / "FITTED_SUPERVISION_stdout_run01.txt").open("x") as out:
            supervision_attempted = True
            completed = subprocess.run(SSH + [command], stdout=out, stderr=subprocess.STDOUT, timeout=2900)
        terminal["stages"].append({"stage": "supervision", "returncode": completed.returncode})
        fetched = fetch_once()
        terminal["status"] = "completed" if completed.returncode == 0 and fetched.returncode == 0 else "failed_or_incomplete"
    except Exception as error:
        terminal.update(status="failed_or_incomplete", error_type=type(error).__name__, error=str(error),
                        no_automatic_restart=True)
    finally:
        if supervision_attempted and not fetch_attempted:
            try:
                fetch_once()
            except Exception as error:
                terminal["retention_error"] = {"type": type(error).__name__, "message": str(error)}
        observed_charges = []
        meter_records = []
        final_snapshot = False
        for name, key in [("diagnostics.json", "charged_union_seconds"),
                          ("progress.json", "diagnostic_charged_union_seconds")]:
            path = PHASE / PACKET / "runs/admitted_run_v1" / name
            if path.is_file():
                try:
                    record = json.loads(path.read_text())
                    value = record.get(key)
                    if name == "diagnostics.json":
                        final_snapshot = record.get("final_snapshot") is True and record.get("active_depth") == 0
                    if type(value) in (int, float) and 0 <= value < float("inf"):
                        observed_charges.append(value)
                        meter_records.append({"path": str(path.relative_to(PHASE)), "sha256": sha256(path.read_bytes()).hexdigest(),
                                              "observed_union_seconds": value})
                except (ValueError, OSError):
                    pass
        observed_diagnostic = max(observed_charges) if observed_charges else None
        diagnostic_charged = max(diagnostic_reserve, observed_diagnostic or 0)
        terminal_meter_path = PHASE / PACKET / "runs/admitted_run_v1/launcher.json"
        meter_terminal = False
        if terminal_meter_path.is_file():
            try:
                launch = json.loads(terminal_meter_path.read_text())
                meter_terminal = launch.get("status") in ("child_exited", "hard_timeout_incomplete", "launcher_failed_or_interrupted")
            except (ValueError, OSError):
                pass
        diagnostic_accounting_known = not supervision_attempted or (meter_terminal and final_snapshot and bool(observed_charges))
        phase_cap_exceeded = prior["combined_phase_seconds"] + max(envelope, time.perf_counter() - started) > prior["phase_cap_seconds"]
        diagnostic_cap_exceeded = prior["diagnostic_seconds"] + diagnostic_charged > prior["diagnostic_cap_seconds"]
        if terminal["status"] == "completed" and (not diagnostic_accounting_known or phase_cap_exceeded or diagnostic_cap_exceeded
                                                   or (observed_diagnostic is not None and observed_diagnostic > 60)):
            terminal["status"] = "accounting_invalid_no_go"
        terminal.update(diagnostic_observed_union_seconds=observed_diagnostic,
                        diagnostic_charged_seconds=diagnostic_charged,
                        diagnostic_accounting_known=diagnostic_accounting_known,
                        diagnostic_final_snapshot=final_snapshot,
                        diagnostic_records=meter_records)
        elapsed = time.perf_counter() - started
        terminal.update(seconds=elapsed, UTC=datetime.now(timezone.utc).isoformat())
        write(HERE / "FITTED_COORDINATOR_run01.json", terminal)
        accounting = dict(prior, schema="gnnm-cumulative-bound-after-fitted-AR-screen-v1-run01",
            UTC=terminal["UTC"], prior_accounting={"path": str(prior_path.relative_to(PHASE)),
            "sha256": sha256(prior_bytes).hexdigest()},
            combined_phase_seconds=prior["combined_phase_seconds"] + max(envelope, elapsed),
            diagnostic_seconds=prior["diagnostic_seconds"] + diagnostic_charged,
            combined_phase_allocated_disk_conservative_bytes=prior["combined_phase_allocated_disk_conservative_bytes"] + 2 * 1024**3,
            fitted_operation_conservative_seconds=max(envelope, elapsed),
            fitted_complete_coordinator_observed_seconds=elapsed,
            fitted_diagnostic_reservation_seconds=diagnostic_reserve,
            fitted_diagnostic_observed_union_seconds=observed_diagnostic,
            fitted_diagnostic_charged_seconds=diagnostic_charged,
            diagnostic_accounting_known=diagnostic_accounting_known,
            unknown_meter_does_not_certify_cap=True,
            phase_cap_exceeded=prior["combined_phase_seconds"] + max(envelope, elapsed) > prior["phase_cap_seconds"],
            diagnostic_cap_exceeded=prior["diagnostic_seconds"] + diagnostic_charged > prior["diagnostic_cap_seconds"],
            actual_GPU_busy_time_claimed=False, original_caps_unchanged=True)
        write(HERE / "CUMULATIVE_BOUND_AFTER_FITTED_run01.json", accounting)
        print(json.dumps(terminal), flush=True)
    return 0 if terminal["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
