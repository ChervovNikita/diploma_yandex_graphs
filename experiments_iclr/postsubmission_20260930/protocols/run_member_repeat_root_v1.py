"""One separately admitted complete-graph numeric repeat audit. No retry."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shlex
import subprocess
import time

PHASE = Path(__file__).resolve().parents[1]
REPO = "/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs"
REMOTE = REPO + "/experiments_iclr/postsubmission_20260930"
DEST = "anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru"
SSH = ["ssh", "-p", "2222", "-i", "/Users/alex/.ssh/mlspace__private_key_anogena.txt",
       "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "UpdateHostKeys=no",
       "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=15", DEST]
PACKAGE = "gine_closest_control_v1/member_repeat_audit_v1"
REQUEST = "protocols/member_repeat_root_v1/REQUEST_ROOT_run01.json"
OUT = PHASE / "protocols/member_repeat_root_v1"
start_path, result_path = OUT / "COORDINATOR_START.json", OUT / "COORDINATOR_RESULT.json"
if start_path.exists() or result_path.exists():
    raise ValueError("No retry")
raw = (PHASE / REQUEST).read_bytes()
request = json.loads(raw)
if request["root_authorized"] is not True:
    raise ValueError("Root admission required")
template = json.loads((PHASE / PACKAGE / "LAUNCH_ARGUMENT_TEMPLATE.json").read_text())
replacements = {"<REMOTE_REPOSITORY_ROOT>": REPO, "<REMOTE_PHASE_ROOT>": REMOTE,
                "<APPROVED_REQUEST_ABSOLUTE_PATH>": REMOTE + "/" + REQUEST}
argv = []
for item in template["argv_template"]:
    for key, value in replacements.items():
        item = item.replace(key, value)
    if "<" in item:
        raise ValueError("Unresolved template")
    argv.append(item)
record = dict(UTC=datetime.now(timezone.utc).isoformat(), request=REQUEST,
              request_sha256=hashlib.sha256(raw).hexdigest(), argv=argv, destination=DEST,
              coordinator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), retry=False)
with start_path.open("x") as stream:
    json.dump(record, stream, indent=2)
    stream.write("\n")
command = ("cd " + shlex.quote(REPO)
           + " && source experiments_iclr/postsubmission_20260930/protocols/repo_env.sh"
           + " && export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 GNNM_SSH_DESTINATION="
           + shlex.quote(DEST) + " && " + shlex.join(argv))
start = time.perf_counter()
try:
    run = subprocess.run(SSH + [command], capture_output=True, text=True, timeout=180)
    result = dict(record, seconds=time.perf_counter()-start, returncode=run.returncode,
                  stdout=run.stdout, stderr=run.stderr, request_unchanged=(PHASE/REQUEST).read_bytes()==raw)
except subprocess.TimeoutExpired as error:
    result = dict(record, seconds=time.perf_counter()-start, returncode=None, timed_out=True,
                  remote_terminal_must_be_inspected=True,
                  stdout=str(error.stdout), stderr=str(error.stderr))
with result_path.open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(json.dumps(dict(returncode=result["returncode"], seconds=result["seconds"], result=str(result_path.relative_to(PHASE)))))
raise SystemExit(result["returncode"] if result["returncode"] is not None else 1)
