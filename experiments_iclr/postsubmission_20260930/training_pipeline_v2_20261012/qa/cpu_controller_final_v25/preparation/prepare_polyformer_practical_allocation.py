"""Freeze explicit practical policy using unchanged verified v2 numerics."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib
from prepare_molecular_packets_allocation import GUARD, TRANSFER, invoke

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from compile_polyformer_practical_matrix import POLICY
DESTINATION = "massive_polyformer_practical_packet_20261012_v1"
LABEL = "polyformer_practical_allocation_20261012_v1"
OWNED = ["compile_polyformer_practical_matrix.py", "tests/test_polyformer_practical.py", "docs/POLYFORMER_PRACTICAL_V1.md"]

REMOTE = GUARD + r'''
transport=phase/(sys.argv[1]+'_transport');encoded=(transport/'payload.b64').read_text()
assert hashlib.sha256(encoded.encode()).hexdigest()==sys.argv[2]
payload=json.loads(zlib.decompress(base64.b64decode(encoded)));stage=phase/payload['destination'];assert not stage.exists();stage.mkdir()
old=phase/'massive_polyformer_packet_20261012_v2';old_package=old/'sources/training_pipeline_v2_20261012'
package=stage/'sources/training_pipeline_v2_20261012';shutil.copytree(old_package,package)
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
qa=json.loads((old/'POLYFORMER_BOUNDARY_CPU_QA_RESULT.json').read_text());assert qa['exit_code']==0 and 'Ran 15 tests' in qa['stderr']
unchanged={}
for rel in ['engine.py','run_study.py','run_endpoint.py','spec.py','registry.py','launcher.py',
 'adapters/polyformer.py','adapters/polyformer_boundary.py','adapters/polyformer_endpoints.py','adapters/classifier.py','adapters/node_losses.py','adapters/node_endpoints.py']:
 assert (package/rel).read_bytes()==(old_package/rel).read_bytes();unchanged[rel]=receipt(package/rel)
atomic(stage/'NUMERICAL_QA_INHERITANCE.json',{'v2_QA_result':receipt(old/'POLYFORMER_BOUNDARY_CPU_QA_RESULT.json'),
 'unchanged_numerical_and_common_controller_sources':unchanged,'prior15_CPU_tests_passed':True,'new_scientific_fits':0})
atomic(stage/'POLICY.json',payload['policy']);(stage/'POLICY.json').chmod(0o444)
atomic(stage/'SOURCE_INVENTORY.json',[receipt(p) for p in sorted(package.rglob('*')) if p.is_file() and '__pycache__' not in p.parts])
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',GNNM_REFERENCE_PROJECT=str(phase))
tmp=stage/'engineering_tmp';tmp.mkdir();env['TMPDIR']=str(tmp)
def run(command,label,timeout):
 tick=time.monotonic();child=subprocess.Popen(command,cwd=package,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 fields=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()
 atomic(stage/(label+'_START.json'),{'pid':child.pid,'start_ticks':int(fields[19]),'command':command,'scientific_training_runs':0})
 try:stdout,stderr=child.communicate(timeout=timeout)
 except subprocess.TimeoutExpired:
  child.kill();stdout,stderr=child.communicate();stderr+='\nBounded engineering timeout retained.\n'
 result={'phase':label,'exit_code':child.returncode,'wall_seconds':time.monotonic()-tick,'stdout':stdout,'stderr':stderr,'scientific_training_runs':0,'benchmark_scores':0}
 atomic(stage/(label+'_RESULT.json'),result);print(json.dumps(result),flush=True);return result
result=run([str(repo/'.venv/bin/python'),'-B','-m','unittest','discover','-s','tests','-p','test_polyformer_practical.py','-v'],'PRACTICAL_POLICY_CPU_QA',300)
if result['exit_code']!=0:sys.exit(1)
result=run([str(repo/'.venv/bin/python'),'-B','-u',str(package/'compile_polyformer_practical_matrix.py'),
 '--project-root',str(phase),'--v2-packet',str(old.relative_to(phase)),'--runtime-prefix',str(package.relative_to(phase)),
 '--policy',str((stage/'POLICY.json').relative_to(phase)),'--output',str((stage/'matrix').relative_to(phase))],
 'DISABLED_POLYFORMER_PRACTICAL_COMPILER',600)
atomic(stage/'PREPARATION_SUMMARY.json',{'status':'prepared' if result['exit_code']==0 else 'blocked_compiler',
 'scientific_fits':0,'benchmark_scores':0,'GO_created':False,'unrestricted_v2_modified':False,'common_controller_modified':False,
 'resource_status':'PROVISIONAL_REQUIRES_ROOT_QUALIFICATION','heldout_target_tensors_decoded':False})
sys.exit(result['exit_code'])
'''


def main():
    out = PACKAGE / "preparation" / LABEL
    if out.exists(): raise FileExistsError("Preserve prior attempts")
    out.mkdir(); files = []
    for relative in OWNED:
        raw = (PACKAGE / relative).read_bytes()
        files.append({"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "data": base64.b64encode(raw).decode()})
    encoded = base64.b64encode(zlib.compress(json.dumps({"destination": DESTINATION, "files": files, "policy": POLICY}).encode())).decode()
    (out / "OWNED_SOURCE_INVENTORY.json").write_text(json.dumps([{k:v for k,v in row.items() if k != "data"} for row in files],indent=2)+"\n")
    for offset in range(0, len(encoded), 50000):
        result = invoke(TRANSFER, [DESTINATION, str(offset), encoded[offset:offset+50000]], out, "TRANSFER_"+str(offset), 50)
        if result["exit_code"] != 0: print(json.dumps(result)); raise SystemExit(1)
    result = invoke(REMOTE, [DESTINATION, hashlib.sha256(encoded.encode()).hexdigest()], out, "TRANSPORT", 1000)
    print(result["stdout"], flush=True)
    if result["exit_code"] != 0: print(result["stderr"]); raise SystemExit(1)


if __name__ == "__main__": main()
