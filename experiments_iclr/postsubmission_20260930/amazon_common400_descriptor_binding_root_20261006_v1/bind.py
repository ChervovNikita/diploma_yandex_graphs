"""Read exact owned common acquisition byte descriptors; no torch/scoring."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shlex, subprocess
HERE=Path(__file__).resolve().parent
REPO="/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs"
REMOTE=r'''
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,socket,subprocess
repo=Path("/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs")
assert Path.cwd().resolve()==repo and socket.gethostname()=="anogena-2-0"
assert subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],text=True,timeout=15).split()==["GPU-44039938-fd82-41d2-fefd-de71514e2fac"]
phase=repo/"experiments_iclr/postsubmission_20260930"
origin=phase/"amazon_learnability_responsibility_strict_scientific_execution_root_20261006_v2/output/scientific_run"
reply={"UTC":datetime.now(timezone.utc).isoformat(),"host":socket.gethostname(),"repository":str(repo),"files":[],"checkpoint_deserialized":False,"scores_or_labels_read":False}
for name in ("initial.pt","RUN.json"):
 p=origin/name
 assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(phase) and p.stat().st_mode&0o222==0
 before=p.stat();h=hashlib.sha256()
 with p.open("rb") as stream:
  for block in iter(lambda:stream.read(1048576),b""):h.update(block)
 after=p.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 row={"path":str(p.relative_to(phase)),"bytes":after.st_size,"sha256":h.hexdigest(),"immutable_mode":oct(after.st_mode&0o777)}
 if name=="RUN.json":
  assert after.st_size<2_000_000;row["utf8"]=p.read_text();json.loads(row["utf8"])
 reply["files"].append(row)
print(json.dumps(reply))
'''
command=["ssh","-T","-p","2222","-i","/Users/alex/.ssh/mlspace__private_key_anogena.txt","-o","BatchMode=yes","-o","IdentitiesOnly=yes","-o","StrictHostKeyChecking=yes","-o","UpdateHostKeys=no","-o","ConnectTimeout=20","anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru","cd "+shlex.quote(REPO)+" && exec /usr/bin/python3 -I -S -B -c "+shlex.quote(REMOTE)]
assert not (HERE/"DESCRIPTOR_RECEIPT.json").exists()
r=subprocess.run(command,capture_output=True,text=True,timeout=55)
receipt={"UTC":datetime.now(timezone.utc).isoformat(),"exit_code":r.returncode,"stderr":r.stderr,"remote_source_sha256":hashlib.sha256(REMOTE.encode()).hexdigest(),"checkpoint_payload_saved_locally":False}
if r.returncode==0:
 receipt["result"]=json.loads(r.stdout)
 for row in receipt["result"]["files"]:
  if "utf8" in row:
   b=row["utf8"].encode();assert len(b)==row["bytes"] and hashlib.sha256(b).hexdigest()==row["sha256"]
   (HERE/"ORIGIN_RUN.json").write_bytes(b);(HERE/"ORIGIN_RUN.json").chmod(0o444)
else:receipt["stdout"]=r.stdout
(HERE/"DESCRIPTOR_RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n")
(HERE/"DESCRIPTOR_RECEIPT.json").chmod(0o444)
print(json.dumps(receipt))
