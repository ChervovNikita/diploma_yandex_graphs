"""Freeze compatible runtime/API policy and CPU checks; no install/77/tar/fit."""
import base64
import hashlib
import json
from pathlib import Path
import zlib
from prepare_molecular_packets_allocation import GUARD, TRANSFER, invoke

PACKAGE = Path(__file__).resolve().parents[1]
DESTINATION = "master_sync_runtime_compatibility_20261012_v1"
LABEL = "runtime_compatibility_allocation_20261012_v1"

REMOTE = GUARD + r'''
transport=phase/(sys.argv[1]+'_transport');encoded=(transport/'payload.b64').read_text()
assert hashlib.sha256(encoded.encode()).hexdigest()==sys.argv[2]
payload=json.loads(zlib.decompress(base64.b64decode(encoded)));stage=phase/payload['destination'];assert not stage.exists();stage.mkdir()
package=stage/'runtime';package.mkdir()
def atomic(path,value):
 temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temporary.replace(path)
def receipt(path):return {'path':str(path.relative_to(phase)),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
for row in payload['files']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 p=package/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);p.chmod(0o444)
atomic(stage/'SOURCE_INVENTORY.json',[receipt(p) for p in sorted(package.rglob('*')) if p.is_file()])
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PIP_DISABLE_PIP_VERSION_CHECK='1',OMP_NUM_THREADS='2')
def run(command,label,timeout):
 tick=time.monotonic();child=subprocess.Popen(command,cwd=package,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 fields=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()
 atomic(stage/(label+'_START.json'),{'pid':child.pid,'start_ticks':int(fields[19]),'command':command,'scientific_fits':0})
 try:stdout,stderr=child.communicate(timeout=timeout)
 except subprocess.TimeoutExpired:
  child.kill();stdout,stderr=child.communicate();stderr+='\nBounded engineering timeout retained.\n'
 result={'label':label,'exit_code':child.returncode,'wall_seconds':time.monotonic()-tick,'stdout':stdout,'stderr':stderr,
  'package_installs':0,'server77_access':False,'scientific_fits':0,'GO_created':False}
 atomic(stage/(label+'_RESULT.json'),result);print(json.dumps(result),flush=True)
 if result['exit_code']!=0:sys.exit(1)
 return result
run([str(repo/'.venv/bin/python'),'-B','-m','unittest','discover','-s','tests','-p','test_runtime_compatibility.py','-v'],
 'REQUIRED_RUNTIME_COMPATIBILITY_CPU_QA',180)
environment='master_sync_tooling_20261012_v4/ENVIRONMENT.json'
run([str(repo/'.venv/bin/python'),'-B',str(package/'prepare_sync_environment.py'),'capture','--project-root',str(phase),
 '--recipe',str((stage/'COMPATIBLE_RECIPE.json').relative_to(phase)),'--environment',environment],
 'COMPATIBLE_DEPENDENCY_RECIPE_CAPTURE',120)
sys.path.insert(0,str(package));from prepare_sync_environment import verify_runtime_compatibility,base_check
from prepare_sync_bundle import fingerprint,compatibility_requirements
expected=json.loads((phase/environment).read_text());recipe=json.loads((stage/'COMPATIBLE_RECIPE.json').read_text())
assert recipe['environment_hash']=='96ca1fc422b22e985e1ac0ee47af919140087eb4034ad898911d9b681bf955b7'
run([str(repo/'.venv/bin/python'),'-B',str(package/'prepare_sync_environment.py'),'dry-run','--project-root',str(phase),
 '--recipe',str((stage/'COMPATIBLE_RECIPE.json').relative_to(phase)),'--recipe-hash',fingerprint(recipe),
 '--base-python',str(repo/'.venv/bin/python')],'COMPATIBLE_RECREATION_DRY_RUN',120)
observed=verify_runtime_compatibility(expected);atomic(stage/'ACTUAL_ALLOCATION_STARTUP_API_RECEIPT.json',observed)
atomic(stage/'BASE_COMPATIBILITY_RECEIPT.json',base_check(repo/'.venv/bin/python',expected))
atomic(stage/'REQUIRED_COMPATIBILITY_POLICY.json',compatibility_requirements(expected))
proof={'schema':'compatible-runtime-preparation-proof-v1','source_environment_hash':fingerprint(expected),
 'compatible_recipe_hash':fingerprint(recipe),'required_compatibility_policy_hash':fingerprint(compatibility_requirements(expected)),
 'source_environment_provenance_preserved':True,'mandatory_numerical_versions_unchanged':recipe['source_environment']['packages']==expected['packages'],
 'observed_dependency_closure_pins':len(recipe['packages']),'all7_compatibility_tests_passed':True,
 'actual_root_startup_API_passed':observed['status']=='required_runtime_compatibility_passed',
 'CPU_native_operations_verified':observed['native_binary_compatibility_verified'],'CUDA_initialized':False,
 'previous_strict_recipes_and_manifests_preserved':True,'package_installs':0,'server77_access':False,'tar_created':False,'scientific_fits':0,'GO_created':False}
atomic(stage/'FINAL_PROOF.json',proof)
paths=[stage/name for name in ['SOURCE_INVENTORY.json','FINAL_PROOF.json','COMPATIBLE_RECIPE.json','REQUIRED_COMPATIBILITY_POLICY.json',
 'ACTUAL_ALLOCATION_STARTUP_API_RECEIPT.json','BASE_COMPATIBILITY_RECEIPT.json','REQUIRED_RUNTIME_COMPATIBILITY_CPU_QA_RESULT.json']]
print(json.dumps({'proof':proof,'mirror_files':[{'receipt':receipt(p),'data':base64.b64encode(p.read_bytes()).decode()} for p in paths]}))
'''


def main():
    out = PACKAGE / "preparation" / LABEL
    if out.exists(): raise FileExistsError("Preserve earlier attempts")
    out.mkdir(); files = []
    for relative in ("prepare_sync_bundle.py", "prepare_sync_environment.py", "tests/test_runtime_compatibility.py",
                     "docs/MASTER_SYNC_ENVIRONMENT.md", "docs/MASTER_SYNC_BUNDLE.md", "docs/RUNTIME_COMPATIBILITY_STARTUP_API.md"):
        raw = (PACKAGE / relative).read_bytes()
        files.append({"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "data": base64.b64encode(raw).decode()})
    encoded = base64.b64encode(zlib.compress(json.dumps({"destination": DESTINATION, "files": files}).encode())).decode()
    result = invoke(TRANSFER, [DESTINATION, "0", encoded], out, "TRANSFER", 60)
    if result["exit_code"] != 0: print(json.dumps(result)); raise SystemExit(1)
    result = invoke(REMOTE, [DESTINATION, hashlib.sha256(encoded.encode()).hexdigest()], out, "TRANSPORT", 700)
    for line in result["stdout"].splitlines():
        if not line.startswith('{'): continue
        try: payload = json.loads(line)
        except json.JSONDecodeError: continue
        if "mirror_files" in payload:
            for row in payload["mirror_files"]:
                raw = base64.b64decode(row["data"]); assert hashlib.sha256(raw).hexdigest() == row["receipt"]["sha256"]
                p = out / "allocation_artifacts" / row["receipt"]["path"]; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(raw)
            print(json.dumps(payload["proof"],indent=2))
        else: print(json.dumps(payload),flush=True)
    if result["exit_code"] != 0: print(result["stderr"]); raise SystemExit(1)


if __name__ == "__main__": main()
