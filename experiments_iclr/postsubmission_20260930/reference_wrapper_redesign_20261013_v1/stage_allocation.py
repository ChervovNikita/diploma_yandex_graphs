"""Stage a disabled source/design packet on the verified authorized allocation.

No training, model construction, Git commit or push occurs in this operation.
The archive contains an explicit source/design inventory, never research data.
"""
from __future__ import annotations
import hashlib
import io
import json
import shlex
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
DESIGN = HERE.name
LOGIN = "anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru"
REMOTE_PHASE = "/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930"


def main():
    files = sorted(p for p in HERE.rglob("*") if p.is_file()
                   and not ({"transport", "__pycache__"} & set(p.relative_to(HERE).parts))
                   and p.suffix != ".pyc")
    files += [PHASE / "AGENTS.md",
              PHASE / "final_sync_master_v29_20261013_v1/run_gpu.sh",
              PHASE / "final_sync_master_v29_20261013_v1/retired_entrypoint_history_20261013/run_gpu_v29_original.sh.txt"]
    manifest = [{"path": str(p.relative_to(PHASE)), "bytes": p.stat().st_size,
                 "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as tar:
        for p in files:
            info = tar.gettarinfo(str(p), arcname=str(p.relative_to(PHASE)))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with p.open("rb") as stream:
                tar.addfile(info, stream)
    packet = output.getvalue()
    digest = hashlib.sha256(packet).hexdigest()
    remote = r'''
import hashlib,io,json,os,socket,subprocess,sys,tarfile
from pathlib import Path
assert socket.gethostname() == "anogena-2-0"
assert subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],text=True).splitlines() == ["GPU-44039938-fd82-41d2-fefd-de71514e2fac"]
repo=Path("/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs")
os.chdir(repo)
assert subprocess.check_output(["git","branch","--show-current"],text=True).strip()=="codex/postsubmission-research-20260930"
assert subprocess.check_output(["git","remote","get-url","origin"],text=True).strip()=="git@github.com:ChervovNikita/diploma_yandex_graphs.git"
changed=subprocess.check_output(["git","--no-optional-locks","status","--short","--untracked-files=no"],text=True,stderr=subprocess.PIPE).splitlines()
expected={"experiments_iclr/postsubmission_20260930/AGENTS.md","experiments_iclr/postsubmission_20260930/final_sync_master_v29_20261013_v1/run_gpu.sh"}
assert all(line[3:] in expected for line in changed), "Unrelated tracked changes appeared before staging"
phase=Path(PHASE_LITERAL)
assert phase.is_dir() and phase.resolve().is_relative_to(repo)
print("READY",flush=True)
data=sys.stdin.buffer.read(PACKET_BYTES)
assert len(data)==PACKET_BYTES and hashlib.sha256(data).hexdigest()==PACKET_HASH
with tarfile.open(fileobj=io.BytesIO(data),mode="r:gz") as tar:
 members=tar.getmembers()
 for m in members:
  path=phase/m.name
  assert m.isfile() and path.resolve().is_relative_to(phase) and not path.is_symlink()
  assert m.name.startswith(DESIGN_LITERAL+"/") or m.name in ["AGENTS.md","final_sync_master_v29_20261013_v1/run_gpu.sh","final_sync_master_v29_20261013_v1/retired_entrypoint_history_20261013/run_gpu_v29_original.sh.txt"]
 for m in members:
  path=phase/m.name
  path.parent.mkdir(parents=True,exist_ok=True)
  payload=tar.extractfile(m).read()
  if m.name.endswith("run_gpu_v29_original.sh.txt") and path.exists():
   assert path.read_bytes()==payload, "Historical launcher differs"
  else:
   temporary=path.with_name(path.name+".reference-redesign.tmp")
   assert not temporary.exists()
   with temporary.open("xb") as stream: stream.write(payload)
   temporary.chmod(0o755 if m.name.endswith("/run_gpu.sh") else 0o644)
   temporary.replace(path)
new=phase/DESIGN_LITERAL
roster=json.loads((new/"ROSTER.json").read_text())
for recipe in roster["recipes"]:
 pin=recipe["source_bindings"]["native_program"]
 source=phase/pin["path"]
 assert source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest()==pin["sha256"], "Existing source dependency differs: "+pin["path"]
receipt={"schema":"disabled-reference-wrapper-stage-v1","hostname":socket.gethostname(),"files":len(members),"packet_bytes":len(data),"packet_sha256":PACKET_HASH,"scientific_jobs_started":0,"GO":False,"runtime_ready":False,"plan_hash":json.loads((new/"SUMMARY.json").read_text())["plan_hash"]}
(new/"ALLOCATION_STAGE.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt),flush=True)
'''.replace("PHASE_LITERAL", repr(REMOTE_PHASE)).replace("PACKET_BYTES", str(len(packet))).replace("PACKET_HASH", repr(digest)).replace("DESIGN_LITERAL", repr(DESIGN))
    command = "stty raw -echo; python3 -B -c " + shlex.quote(remote)
    argv = ["ssh", "-tt", "-e", "none", "-p", "2222", "-i", "/Users/alex/.ssh/mlspace__private_key_anogena.txt",
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=yes",
            "-o", "UpdateHostKeys=no", LOGIN, command]
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    first = proc.stdout.readline()
    if first.strip() != b"READY":
        proc.terminate()
        _, err = proc.communicate(timeout=15)
        raise RuntimeError("Allocation packet was not admitted: " + first.decode(errors="replace") + err.decode(errors="replace"))
    out, err = proc.communicate(input=packet, timeout=120)
    if proc.returncode:
        raise RuntimeError(out.decode(errors="replace") + err.decode(errors="replace"))
    receipt = json.loads(out.decode().strip())
    receipt["inventory"] = manifest
    (HERE / "ALLOCATION_STAGE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "inventory"}))


if __name__ == "__main__":
    main()
