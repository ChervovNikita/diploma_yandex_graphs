"""Stage the exact missing native dependencies, then one first data export."""
import base64
import json
from pathlib import Path
import zlib
from deploy_and_prepare import PHASE, HERE, REPO, run, sha


def main():
    bindings = json.loads((PHASE/'masked_context_be_source_prototype_20261010_v2/SOURCE_BINDINGS.json').read_text())
    rows = []
    for row in bindings['dependencies'].values():
        source = PHASE/row['path']
        assert source.resolve().is_relative_to(PHASE) and sha(source) == row['sha256']
        rows.append(dict(path=row['path'], sha256=row['sha256'], content=base64.b64encode(source.read_bytes()).decode()))
    payload = base64.b64encode(zlib.compress(json.dumps(rows).encode(),9)).decode()
    code = '''import base64,hashlib,json,pathlib,socket,subprocess,zlib
assert socket.gethostname()=="anogena-2-0"
assert subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],text=True).splitlines()==["GPU-44039938-fd82-41d2-fefd-de71514e2fac"]
r=pathlib.Path(REPO)
p=r/"experiments_iclr/postsubmission_20260930"
for row in json.loads(zlib.decompress(base64.b64decode(PAYLOAD))):
 path=p/row["path"]
 assert path.resolve().is_relative_to(p)
 content=base64.b64decode(row["content"])
 assert hashlib.sha256(content).hexdigest()==row["sha256"]
 if path.exists(): assert path.read_bytes()==content
 else:
  path.parent.mkdir(parents=True,exist_ok=True)
  path.write_bytes(content)
prep=p/"masked_context_pubmed_allocation_preparation_20261010_v1"
assert not (prep/"data").exists() and not (prep/"EXPORT_OWNER_PLAN.json").exists()
runtime=p/"native_ncn_runtime_20261005_v1/.venv/bin/python"
plan=dict(enabled=True,automatic_retry=False,seconds=600,RSS_bytes=16*1024**3,GPU_bytes=0,log="EXPORT.log",terminal="EXPORT_TERMINAL.json",scope="Complete public data and native split projection only, no model or scoring",argv=[str(runtime),"-B",str(prep/"export_train.py")])
(prep/"EXPORT_OWNER_PLAN.json").write_text(json.dumps(plan,indent=2)+"\\n")
result=subprocess.run(["/usr/bin/python3","-I","-S","-B",str(prep/"finite_owner.py"),"--plan",str(prep/"EXPORT_OWNER_PLAN.json")],cwd=r,timeout=620,check=False)
print("export_owner_exit",result.returncode,flush=True)
if (prep/"DATA_CUSTODY.json").exists(): print((prep/"DATA_CUSTODY.json").read_text(),flush=True)
if result.returncode: print((prep/"EXPORT.log").read_text()[-8000:],flush=True)
raise SystemExit(result.returncode)
'''.replace('REPO',repr(REPO),1).replace('PAYLOAD',repr(payload),1)
    run(code,HERE/'DEPENDENCY_STAGE_EXPORT_RECEIPT.json')


if __name__ == '__main__':
    main()
