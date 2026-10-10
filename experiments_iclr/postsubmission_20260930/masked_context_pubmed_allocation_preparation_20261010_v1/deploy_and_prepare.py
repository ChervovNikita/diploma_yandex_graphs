"""Narrow source stage and one finite public-data export on the allowed route."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import zlib

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
REMOTE_PHASE = REPO+'/experiments_iclr/postsubmission_20260930'
SSH = ['/usr/bin/ssh', '-tt', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt', '-p', '2222', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=yes', 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(code, receipt):
    assert not receipt.exists()
    result = subprocess.run(SSH+[shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', code])], capture_output=True, text=True, timeout=650)
    record = dict(UTC=datetime.now(timezone.utc).isoformat(), exit_code=result.returncode, command_sha256=hashlib.sha256(code.encode()).hexdigest(), stdout=result.stdout, stderr=result.stderr)
    receipt.write_text(json.dumps(record, indent=2, sort_keys=True)+'\n')
    print(result.stdout, flush=True)
    if result.returncode:
        print(result.stderr, flush=True)
        raise SystemExit(result.returncode)


def main():
    source = PHASE/'masked_context_be_source_prototype_20261010_v2'
    assert sha(source/'MANIFEST.json') == '18459e4b8c72da623b179d2bd9177e1a9618e8b2ad72de2e2649cec3dcafc531'
    rows = []
    for root in (source, HERE):
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix in ('.py', '.md', '.json', '.sha256', '.diff') and 'RECEIPT' not in path.name:
                rows.append(dict(path=str(path.relative_to(PHASE)), sha256=sha(path), content=base64.b64encode(path.read_bytes()).decode()))
    code = '''import base64,hashlib,json,pathlib,socket,subprocess,os,zlib
assert socket.gethostname()=="anogena-2-0"
assert subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],text=True).splitlines()==["GPU-44039938-fd82-41d2-fefd-de71514e2fac"]
r=pathlib.Path(REPO)
p=r/"experiments_iclr/postsubmission_20260930"
rows=json.loads(zlib.decompress(base64.b64decode(PAYLOAD)))
for row in rows:
 path=p/row["path"]
 assert path.resolve().is_relative_to(p)
 data=base64.b64decode(row["content"])
 assert hashlib.sha256(data).hexdigest()==row["sha256"]
 if path.exists(): assert path.read_bytes()==data
 else:
  path.parent.mkdir(parents=True,exist_ok=True)
  path.write_bytes(data)
v=json.loads((p/"masked_context_be_source_prototype_20261010_v2/SOURCE_BINDINGS.json").read_text())
for row in v["dependencies"].values():
 path=p/row["path"]
 assert path.resolve().is_relative_to(p) and hashlib.sha256(path.read_bytes()).hexdigest()==row["sha256"]
prep=p/"masked_context_pubmed_allocation_preparation_20261010_v1"
runtime=p/"native_ncn_runtime_20261005_v1/.venv/bin/python"
plan=dict(enabled=True,automatic_retry=False,seconds=600,RSS_bytes=16*1024**3,GPU_bytes=0,log="EXPORT.log",terminal="EXPORT_TERMINAL.json",scope="Complete public data and native split projection only, no model or scoring",argv=[str(runtime),"-B",str(prep/"export_train.py")])
(prep/"EXPORT_OWNER_PLAN.json").write_text(json.dumps(plan,indent=2)+"\\n")
result=subprocess.run(["/usr/bin/python3","-I","-S","-B",str(prep/"finite_owner.py"),"--plan",str(prep/"EXPORT_OWNER_PLAN.json")],cwd=r,timeout=620,check=False)
print("export_owner_exit",result.returncode,flush=True)
if result.returncode==0: print((prep/"DATA_CUSTODY.json").read_text(),flush=True)
raise SystemExit(result.returncode)
'''.replace('REPO', repr(REPO), 1).replace('PAYLOAD', repr(base64.b64encode(zlib.compress(json.dumps(rows).encode(),9)).decode()), 1)
    run(code, HERE/'DEPLOY_EXPORT_RECEIPT.json')


if __name__ == '__main__':
    main()
