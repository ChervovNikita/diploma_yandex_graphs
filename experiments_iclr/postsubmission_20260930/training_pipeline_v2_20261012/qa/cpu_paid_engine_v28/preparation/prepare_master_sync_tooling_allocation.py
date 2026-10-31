"""Freeze new sync tooling, stdlib QA, exact environment, actual master union."""
import base64
import hashlib
import json
from pathlib import Path
import zlib
from prepare_molecular_packets_allocation import GUARD, TRANSFER, invoke

PACKAGE = Path(__file__).resolve().parents[1]
DESTINATION = "master_sync_tooling_20261012_v4"
LABEL = "master_sync_tooling_allocation_20261012_v4"
MASTER = "massive_practical_master_20261012_v2/PLAN.json"

REMOTE = GUARD + r'''
transport=phase/(sys.argv[1]+'_transport');encoded=(transport/'payload.b64').read_text()
assert hashlib.sha256(encoded.encode()).hexdigest()==sys.argv[2]
payload=json.loads(zlib.decompress(base64.b64decode(encoded)));stage=phase/payload['destination'];assert not stage.exists();stage.mkdir()
package=stage/'runtime';package.mkdir()
def atomic(path,value):
 temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temporary.replace(path)
def receipt(path):
 digest=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1048576),b''):digest.update(chunk)
 return {'path':str(path.relative_to(phase)),'bytes':path.stat().st_size,'sha256':digest.hexdigest()}
for row in payload['files']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 p=package/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);p.chmod(0o444)
atomic(stage/'SOURCE_INVENTORY.json',[receipt(p) for p in sorted(package.rglob('*')) if p.is_file()])
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
 GNNM_SHARED_RUNTIME=str(phase/'master_controller_source_20261012_v1/runtime'))
def run(command,label,timeout):
 tick=time.monotonic();child=subprocess.Popen(command,cwd=package,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 fields=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()
 atomic(stage/(label+'_START.json'),{'pid':child.pid,'start_ticks':int(fields[19]),'command':command,'science_fits':0})
 try:stdout,stderr=child.communicate(timeout=timeout)
 except subprocess.TimeoutExpired:
  child.kill();stdout,stderr=child.communicate();stderr+='\nBounded metadata-only preparation timeout retained.\n'
 result={'label':label,'exit_code':child.returncode,'wall_seconds':time.monotonic()-tick,'stdout':stdout,'stderr':stderr,
  'scientific_fits':0,'tensor_arrays_deserialized':0,'GO_created':False}
 atomic(stage/(label+'_RESULT.json'),result);print(json.dumps(result),flush=True);return result
qa=run([str(repo/'.venv/bin/python'),'-B','-m','unittest','discover','-s','tests','-p','test_sync_bundle.py','-v'],'SYNC_STDLIB_QA',300)
if qa['exit_code']!=0:sys.exit(1)
result=run([str(repo/'.venv/bin/python'),'-B',str(package/'prepare_sync_bundle.py'),'environment','--output',str(stage/'ENVIRONMENT.json')],
 'EXACT_ENVIRONMENT_METADATA',120)
if result['exit_code']!=0:sys.exit(1)
result=run([str(repo/'.venv/bin/python'),'-B',str(package/'prepare_sync_bundle.py'),'prepare','--project-root',str(phase),
 '--plan',payload['master'],'--environment',str((stage/'ENVIRONMENT.json').relative_to(phase)),
 '--output','master_sync_manifest_20261012_v4'],'ACTUAL_MASTER_UNION_MANIFEST',900)
if result['exit_code']!=0:sys.exit(1)
paths=[stage/'ENVIRONMENT.json',stage/'SOURCE_INVENTORY.json',stage/'SYNC_STDLIB_QA_RESULT.json',
 phase/'master_sync_manifest_20261012_v4/MANIFEST.json',phase/'master_sync_manifest_20261012_v4/SUMMARY.json']
print(json.dumps({'mirror_files':[{'receipt':receipt(p),'data':base64.b64encode(p.read_bytes()).decode()} for p in paths]}))
'''


def main():
    out = PACKAGE / "preparation" / LABEL
    if out.exists(): raise FileExistsError("Preserve prior attempts")
    out.mkdir(); files = []
    for relative in ("prepare_sync_bundle.py", "tests/test_sync_bundle.py", "docs/MASTER_SYNC_BUNDLE.md"):
        raw = (PACKAGE / relative).read_bytes()
        files.append({"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "data": base64.b64encode(raw).decode()})
    encoded = base64.b64encode(zlib.compress(json.dumps({"destination": DESTINATION, "files": files, "master": MASTER}).encode())).decode()
    (out / "OWNED_SOURCE_INVENTORY.json").write_text(json.dumps([{k:v for k,v in row.items() if k != "data"} for row in files],indent=2)+"\n")
    result = invoke(TRANSFER, [DESTINATION, "0", encoded], out, "TRANSFER", 60)
    if result["exit_code"] != 0: print(json.dumps(result)); raise SystemExit(1)
    result = invoke(REMOTE, [DESTINATION, hashlib.sha256(encoded.encode()).hexdigest()], out, "TRANSPORT", 1300)
    lines = result["stdout"].splitlines()
    for line in lines:
        if line.startswith('{'):
            try: payload = json.loads(line)
            except json.JSONDecodeError: continue
            if "mirror_files" in payload:
                for row in payload["mirror_files"]:
                    raw = base64.b64decode(row["data"]); assert hashlib.sha256(raw).hexdigest() == row["receipt"]["sha256"]
                    p = out / "allocation_artifacts" / row["receipt"]["path"]; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(raw)
            else: print(json.dumps(payload),flush=True)
    if result["exit_code"] != 0: print(result["stderr"]); raise SystemExit(1)


if __name__ == "__main__": main()
