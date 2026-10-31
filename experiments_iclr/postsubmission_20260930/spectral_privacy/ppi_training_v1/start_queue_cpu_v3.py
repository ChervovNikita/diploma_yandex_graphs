"""Start the frozen complete CPU queue in a persistent logged subprocess."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
REPO = PHASE.parents[1]
if os.environ.get("GNNM_PHASE_ROOT") != str(PHASE) or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise ValueError("Require explicit repository environment and CPU-only visibility")
launch = HERE / "launch_cpu_v3_run01"
if launch.exists():
    raise ValueError("Queue already launched or attempted; inspect the preserved launch")
protocol = HERE / "RUNTIME_PROTOCOL_cpu_v3.json"
queue = HERE / "queue_cpu_v3.py"
v = json.loads(protocol.read_text())
if v["source_sha256"][str(queue.relative_to(PHASE))] != hashlib.sha256(queue.read_bytes()).hexdigest():
    raise ValueError("Frozen queue source differs")
launch.mkdir()
argv = [str(REPO / ".venv/bin/python"), str(PHASE / "protocols/run_logged.py"),
        "--output", str(PHASE / "protocols/logged_runs/spectral_cpu_v3_queue_run01"), "--",
        str(REPO / ".venv/bin/python"), str(queue), "--protocol", str(protocol),
        "--output", str(HERE / "queue_cpu_v3_run01")]
with (launch / "stdout_stderr.log").open("x") as log:
    child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             cwd=PHASE, env=os.environ.copy(), start_new_session=True)
record = {"pid": child.pid, "start_utc": datetime.now(timezone.utc).isoformat(), "argv": argv,
          "protocol_file_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
          "GPU_used": False, "status": "launched; verify queue and fit progress separately"}
with (launch / "launch.json").open("x") as stream:
    json.dump(record, stream, indent=2)
    stream.write("\n")
print(json.dumps(record), flush=True)
