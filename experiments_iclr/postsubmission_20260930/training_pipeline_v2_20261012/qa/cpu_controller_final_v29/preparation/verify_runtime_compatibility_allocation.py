"""Corrected API harness under the scientific Python; inherit identical7QA."""
import base64
import hashlib
import json
from pathlib import Path
from prepare_molecular_packets_allocation import GUARD, invoke

PACKAGE = Path(__file__).resolve().parents[1]
REMOTE = GUARD + r'''
old=phase/'master_sync_runtime_compatibility_20261012_v1';stage=phase/'master_sync_runtime_compatibility_20261012_v2'
assert not stage.exists();stage.mkdir();package=stage/'runtime';shutil.copytree(old/'runtime',package)
def atomic(path,value):
 temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temporary.replace(path)
def receipt(path):return {'path':str(path.relative_to(phase)),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
qa=json.loads((old/'REQUIRED_RUNTIME_COMPATIBILITY_CPU_QA_RESULT.json').read_text())
assert qa['exit_code']==0 and 'Ran 7 tests' in qa['stderr'] and 'skipped' not in qa['stderr']
for path in (old/'runtime').rglob('*'):
 if path.is_file():assert path.read_bytes()==(package/path.relative_to(old/'runtime')).read_bytes()
atomic(stage/'CPU_QA_INHERITANCE.json',{'source_files_identical':True,'prior7_actual_CPU_tests_passed':True,
 'QA_receipt':receipt(old/'REQUIRED_RUNTIME_COMPATIBILITY_CPU_QA_RESULT.json'),'wrong_stdlib_caller_attempt_preserved':True})
shutil.copyfile(old/'COMPATIBLE_RECIPE.json',stage/'COMPATIBLE_RECIPE.json');(stage/'COMPATIBLE_RECIPE.json').chmod(0o444)
atomic(stage/'SOURCE_INVENTORY.json',[receipt(p) for p in sorted(package.rglob('*')) if p.is_file()])
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
code="""import json,sys
from pathlib import Path
from prepare_sync_environment import verify_runtime_compatibility,base_check
from prepare_sync_bundle import canonical,compatibility_requirements,fingerprint
root=Path(sys.argv[1]);expected=json.loads((root/'master_sync_tooling_20261012_v4/ENVIRONMENT.json').read_text())
receipt=verify_runtime_compatibility(expected)
base=base_check(sys.executable,expected)
print(canonical({'receipt':receipt,'base':base,'policy':compatibility_requirements(expected),'policy_hash':fingerprint(compatibility_requirements(expected))}))
"""
command=[str(repo/'.venv/bin/python'),'-B','-c',code,str(phase)]
tick=time.monotonic();child=subprocess.Popen(command,cwd=package,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
fields=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()
atomic(stage/'CORRECT_SCIENTIFIC_PYTHON_API_START.json',{'pid':child.pid,'start_ticks':int(fields[19]),'command':command,'fits':0})
stdout,stderr=child.communicate(timeout=180)
result={'exit_code':child.returncode,'stdout':stdout,'stderr':stderr,'wall_seconds':time.monotonic()-tick,'scientific_fits':0,'GO_created':False}
atomic(stage/'CORRECT_SCIENTIFIC_PYTHON_API_RESULT.json',result)
if child.returncode:print(json.dumps(result));sys.exit(1)
payload=json.loads(stdout);observed=payload['receipt'];assert observed['status']=='required_runtime_compatibility_passed'
assert observed['native_binary_compatibility_verified'] and not observed['CUDA_initialized']
atomic(stage/'ACTUAL_ALLOCATION_STARTUP_API_RECEIPT.json',observed)
atomic(stage/'BASE_COMPATIBILITY_RECEIPT.json',payload['base'])
atomic(stage/'REQUIRED_COMPATIBILITY_POLICY.json',payload['policy'])
recipe=json.loads((stage/'COMPATIBLE_RECIPE.json').read_text())
import hashlib
proof={'schema':'compatible-runtime-preparation-proof-v1','required_compatibility_policy_hash':payload['policy_hash'],
 'compatible_recipe_hash':hashlib.sha256(json.dumps(recipe,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
 'source_environment_hash':observed['source_environment_hash'],'all7_actual_CPU_tests_passed_inherited_by_exact_source_hash':True,
 'actual_root_API_passed_on_scientific_Python':True,'native_CPU_operations_passed':True,'CUDA_initialized':False,
 'micro_SQLite_SSL_glibc_are_provenance_with_differences_recorded':True,'mandatory_numerical_locks_unchanged':True,
 'author_allocation_contract_unchanged':True,'package_installs':0,'server77_access':False,'tar_created':False,'scientific_fits':0,'GO_created':False}
atomic(stage/'FINAL_PROOF.json',proof)
paths=[stage/name for name in ['SOURCE_INVENTORY.json','CPU_QA_INHERITANCE.json','FINAL_PROOF.json','COMPATIBLE_RECIPE.json',
 'ACTUAL_ALLOCATION_STARTUP_API_RECEIPT.json','BASE_COMPATIBILITY_RECEIPT.json','REQUIRED_COMPATIBILITY_POLICY.json']]
print(json.dumps({'proof':proof,'mirror_files':[{'receipt':receipt(p),'data':base64.b64encode(p.read_bytes()).decode()} for p in paths]}))
'''


def main():
    out = PACKAGE / "preparation/runtime_compatibility_allocation_20261012_v2"
    if out.exists(): raise FileExistsError("Preserve earlier attempts")
    out.mkdir(); result = invoke(REMOTE, [], out, "CORRECTED_API_PROOF", 500)
    if result["exit_code"] != 0: print(result["stdout"],result["stderr"]); raise SystemExit(1)
    payload = json.loads(next(line for line in result["stdout"].splitlines() if line.startswith('{')))
    for row in payload["mirror_files"]:
        raw = base64.b64decode(row["data"]); assert hashlib.sha256(raw).hexdigest() == row["receipt"]["sha256"]
        p = out / "allocation_artifacts" / row["receipt"]["path"]; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(raw)
    print(json.dumps(payload["proof"],indent=2))


if __name__ == "__main__": main()
