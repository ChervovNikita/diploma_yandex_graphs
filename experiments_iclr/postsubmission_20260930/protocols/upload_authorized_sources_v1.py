"""Transfer an explicit immutable source inventory to the authorized phase.

Existing identical files are preserved. Different files are refused. This is
source transport, not experiment execution or scientific admission.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile

PHASE = Path(__file__).resolve().parents[1]
REPO = "/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs"
REMOTE = REPO + "/experiments_iclr/postsubmission_20260930"
SSH = ["ssh", "-p", "2222", "-i", "/Users/alex/.ssh/mlspace__private_key_anogena.txt",
       "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "UpdateHostKeys=no",
       "-o", "StrictHostKeyChecking=yes",
       "anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru"]
EXTRACTOR = '''import sys,tarfile,pathlib,hashlib,json
root=pathlib.Path(sys.argv[1]).resolve()
records=[]
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz') as archive:
 for member in archive:
  p=pathlib.PurePosixPath(member.name)
  if p.is_absolute() or '..' in p.parts or not member.isfile(): raise ValueError('Unsafe source member')
  target=root.joinpath(*p.parts)
  if not target.resolve().is_relative_to(root): raise ValueError('Escaping source member')
  data=archive.extractfile(member).read()
  if target.exists():
   if target.read_bytes()!=data: raise ValueError('Existing source differs: '+str(p))
   state='identical_preserved'
  else:
   target.parent.mkdir(parents=True,exist_ok=True)
   with target.open('xb') as stream: stream.write(data)
   state='created'
  records.append({'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'state':state})
print(json.dumps({'status':'uploaded','files':records}))
'''


def new_output(path):
    path = path.resolve()
    if not path.is_relative_to(PHASE / "protocols") or path.exists():
        raise ValueError("Require a new transport record inside phase protocols")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    receipt, archive_path = new_output(args.receipt), new_output(args.archive)
    if len(set(args.files)) != len(args.files):
        raise ValueError("Duplicate source inventory member")
    records = []
    for value in args.files:
        relative = PurePosixPath(value)
        path = PHASE.joinpath(*relative.parts)
        if (relative.is_absolute() or ".." in relative.parts or str(relative) != value
                or not path.resolve().is_relative_to(PHASE)
                or not path.is_file() or path.is_symlink()):
            raise ValueError("Require normalized phase-relative regular source files")
        if path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError("Source transport excludes large binaries")
        data = path.read_bytes()
        records.append({"path": value, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "x:gz") as archive:
        for row in records:
            path = PHASE / row["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise RuntimeError("Source changed during transport preparation")
            archive.add(path, arcname=row["path"], recursive=False)
    command = ("cd " + shlex.quote(REPO)
               + " && source experiments_iclr/postsubmission_20260930/protocols/repo_env.sh"
               + " && .venv/bin/python -c " + shlex.quote(EXTRACTOR) + " " + shlex.quote(REMOTE))
    start = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(SSH + [command], input=archive_path.read_bytes(), capture_output=True, check=False)
    stdout = result.stdout.decode(errors="replace")
    remote_result = json.loads(stdout) if result.returncode == 0 else None
    if remote_result is not None:
        remote_records = [{key: row[key] for key in ("path", "bytes", "sha256")}
                          for row in remote_result["files"]]
        if remote_records != records:
            raise RuntimeError("Remote transport records differ from source inventory")
    value = {"schema": "gnnm-authorized-immutable-source-transport-v1",
             "start_UTC": start, "completion_UTC": datetime.now(timezone.utc).isoformat(),
             "ssh_destination": SSH[-1], "remote_phase": REMOTE,
             "files": records, "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
             "transport_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "exit_code": result.returncode, "remote_result": remote_result,
             "stdout_on_failure": stdout if result.returncode else None,
             "stderr": result.stderr.decode(errors="replace")}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"exit_code": result.returncode, "files": len(records), "receipt": str(receipt)}))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
