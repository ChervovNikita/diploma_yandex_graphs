"""Detach the exact qualified six-cell GPU queue; do not alter its recipe."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PHASE = Path(__file__).resolve().parents[1]
REPO = PHASE.parents[1]
DESTINATION = "anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru"
PACKET = PHASE / "sharing_policy/gine_transfer_gpu_queue_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if len(sys.argv) != 1:
        raise ValueError("This launcher accepts no overrides")
    if Path.cwd().resolve() != REPO or os.environ.get("GNNM_SSH_DESTINATION") != DESTINATION:
        raise ValueError("Require the authorized repository and corrected SSH route")
    if os.environ.get("GNNM_PHASE_ROOT") != str(PHASE) or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Require the repository environment and physical GPU0")
    manifest_path = PACKET / "DELIVERY_MANIFEST_GPU_v1.json"
    if sha(manifest_path) != "7643ddbb973206ef6924f3aa52f0ea2ffd4d4235c5cef7b2a3a628976e625c54":
        raise ValueError("The audited source packet changed")
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest["files_sha256"].items():
        if sha(PACKET / relative) != expected:
            raise ValueError("Audited source changed: " + relative)
    qualification = PACKET / "qualifications/authorized1gpu_run01.json"
    qualified = json.loads(qualification.read_text())
    if qualified.get("status") != "qualified":
        raise ValueError("Successful input qualification is required")
    # The queue independently repeats every semantic qualification and the
    # exact UUID/idle checks before starting any experiment child.
    queue_supervisor = PHASE / "protocols/logged_runs/gine_GPU_queue_authorized1gpu_run01"
    queue_output = PACKET / "queues/all_six_authorized1gpu_run01"
    if queue_supervisor.exists() or queue_output.exists():
        raise ValueError("Refuse a duplicate or resumed queue")
    launch = PHASE / "protocols/detached/gine_GPU_queue_authorized1gpu_run01"
    launch.mkdir(parents=True, exist_ok=False)
    interpreter = REPO / ".venv/bin/python"
    command = [str(interpreter), str(PHASE / "protocols/run_authorized_v2.py"),
               "--output", str(queue_supervisor), "--require-idle-gpu", "--",
               str(interpreter), str(PACKET / "gpu_queue.py"),
               "--python", str(interpreter), "--qualification", str(qualification),
               "--output", str(queue_output)]
    with (launch / "launcher.log").open("x") as stream:
        child = subprocess.Popen(command, cwd=REPO, env=os.environ.copy(),
                                 stdin=subprocess.DEVNULL, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True,
                                 close_fds=True)
    record = {"schema": "gnnm-detached-exact-GPU-queue-launch-v1",
              "launch_UTC": datetime.now(timezone.utc).isoformat(),
              "ssh_destination": DESTINATION, "pid": child.pid,
              "command": command, "launcher_source_sha256": sha(Path(__file__)),
              "manifest_sha256": sha(manifest_path),
              "qualification_sha256": sha(qualification),
              "queue_supervisor": str(queue_supervisor), "queue_output": str(queue_output),
              "start_new_session": True, "automatic_retry": False,
              "completion_not_implied_by_launch": True}
    with (launch / "launch.json").open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "detached_queue_process_started", "pid": child.pid,
                      "launch_record": str(launch / "launch.json")}), flush=True)


if __name__ == "__main__":
    main()
