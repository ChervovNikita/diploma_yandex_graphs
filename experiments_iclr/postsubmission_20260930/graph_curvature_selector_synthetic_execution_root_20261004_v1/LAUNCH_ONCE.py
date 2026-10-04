"""Exclusive guard makes an uncertain SSH outcome ineligible for relaunch."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
observed = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                          capture_output=True, text=True, check=True).stdout.splitlines()
assert [line.strip() for line in observed if line.strip()] == ["GPU-44039938-fd82-41d2-fefd-de71514e2fac"]
config = json.loads((HERE / "EXECUTION_INPUT.json").read_text())
with (HERE / "START_GUARD.json").open("x") as stream:
    json.dump(dict(authorized_once=True, launch_unix=time.time(), launcher_pid=os.getpid()), stream)
    stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
env = os.environ.copy(); env.update(config["environment"])
env["TMPDIR"] = str(HERE / "tmp")
with (HERE / "WRAPPER_STDOUT.txt").open("x") as log:
    argv = [sys.executable, "-B", str(HERE / "EXECUTE_ONCE.py")]
    process = subprocess.Popen(argv, cwd=str(HERE), env=env, start_new_session=True,
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    receipt = dict(pid=process.pid, detached=True, argv=argv,
                   source_manifest_sha256=config["source_manifest_sha256"],
                   harness_manifest_sha256=config["harness_manifest_sha256"])
    with (HERE / "PID_RECEIPT.json").open("x") as stream:
        json.dump(receipt, stream, indent=2); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
print(json.dumps(receipt))
