"""Fresh guarded v5 manifest and CPU source-access proof; no archive or science."""
import base64
import hashlib
import json
from pathlib import Path
import zlib
from prepare_molecular_packets_allocation import GUARD, TRANSFER, invoke

PACKAGE = Path(__file__).resolve().parents[1]
DESTINATION = "master_sync_manifest_20261012_v6"
LABEL = "master_sync_manifest_allocation_20261012_v6"

REMOTE = GUARD + r'''
transport=phase/(sys.argv[1]+'_transport');encoded=(transport/'payload.b64').read_text()
assert hashlib.sha256(encoded.encode()).hexdigest()==sys.argv[2]
payload=json.loads(zlib.decompress(base64.b64decode(encoded)))
stage=phase/payload['destination'];assert not stage.exists();stage.mkdir()
package=stage/'runtime';package.mkdir()
def atomic(path,value):
 assert not path.exists()
 with path.open('x') as stream:stream.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
 path.chmod(0o444)
def receipt(path):
 digest=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1048576),b''):digest.update(chunk)
 return {'path':str(path.relative_to(phase)),'bytes':path.stat().st_size,'sha256':digest.hexdigest()}
for row in payload['files']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 target=package/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw);target.chmod(0o444)
atomic(stage/'SOURCE_INVENTORY.json',[receipt(package/row['path']) for row in payload['files']])
controller=phase/'master_controller_source_20261012_v3/runtime'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
 GNNM_SHARED_RUNTIME=str(controller),PYTHONPATH=str(controller)+os.pathsep+str(package/'tests'))
def run(command,label,timeout,cwd=package):
 tick=time.monotonic();child=subprocess.Popen(command,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 fields=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()
 atomic(stage/(label+'_START.json'),{'pid':child.pid,'start_ticks':int(fields[19]),'command':command,'scientific_fits':0})
 try:stdout,stderr=child.communicate(timeout=timeout)
 except subprocess.TimeoutExpired:
  child.kill();stdout,stderr=child.communicate();stderr+='\nBounded metadata-only preparation timeout retained.\n'
 result={'label':label,'exit_code':child.returncode,'wall_seconds':time.monotonic()-tick,'stdout':stdout,'stderr':stderr,
  'scientific_fits':0,'GO_created':False}
 atomic(stage/(label+'_RESULT.json'),result)
 print(json.dumps({k:v for k,v in result.items() if k!='stdout'}),flush=True)
 return result
python=str(repo/'.venv/bin/python')
prior_fixture=phase/'master_sync_manifest_20261012_v5_retry1/ABSENT_PROVIDER_FIXTURE_RESULT.json'
prior=json.loads(prior_fixture.read_text());assert prior['exit_code']==0
for name in ('prepare_sync_bundle.py','prepare_sync_environment.py'):
 assert (controller/name).read_bytes()==(phase/'master_controller_source_20261012_v2/runtime'/name).read_bytes()
atomic(stage/'ABSENT_PROVIDER_FIXTURE_RESULT.json',{'exit_code':0,'inherited':True,
 'prior_fixture':receipt(prior_fixture),'exact_sync_helper_bytes_equal':True,
 'repeated_CPU_test':False,'scientific_fits':0})
build=run([python,'-B',str(package/'build_master_sync_v5.py'),'--project-root',str(phase),'--output',payload['destination']],
 'DISABLED_V5_MANIFEST',900)
if build['exit_code']!=0:sys.exit(1)
audit=run([python,'-B',str(package/'audit_master_v5_source_loading.py'),'--project-root',str(phase),
 '--manifest',payload['destination']+'/MANIFEST.json'],'ACTUAL_LAUNCHER_SOURCE_ACCESS',300)
if audit['exit_code']!=0:sys.exit(1)
audit_value=json.loads(audit['stdout']);atomic(stage/'ACTUAL_LAUNCHER_SOURCE_ACCESS_AUDIT.json',audit_value)
summary=json.loads((stage/'SUMMARY.json').read_text())
proof={'schema':'disabled-v5-portable-sync-preparation-proof-v1','manifest_hash':summary['manifest_hash'],
 'plan_hash':summary['plan_hash'],'environment_hash':summary['environment_hash'],'recipe_hash':summary['recipe_hash'],
 'statistics':summary['statistics'],'model_groups':summary['model_groups'],'controller_files':summary['controller_files'],
 'actual_launcher_load_plan_passed':True,'loaded_studies':audit_value['loaded_studies'],
 'all_records_exactly_once_and_same_host_dependencies':True,'all_required_source_bytes_validated_by_actual_launcher':True,
 'actual_phase_file_reads_within_manifest_union':True,'optional_provider_cache_denied_and_unused':True,
 'absent_provider_fixture_inherited_from_exact_sync_helper_bytes':True,'scientific_or_tensor_loader_imports_denied_during_load_plan':True,
 'targets_or_tensor_arrays_deserialized':False,'compatible_runtime_preflight_passed':True,
 'mandatory_numerical_ABI_locks_preserved':True,'author_absolute_contracts_allocation_only':True,
 'protected_PT_source_inventory_rechecked_exact_original_bytes':True,'failed_v5_preserved':True,
 'resources_pending':True,'inventory_migrations_pending':True,'final_ALL_plan':False,
 'tar_created':False,'package_installs':0,'server77_access':False,'scientific_fits':0,'GO_created':False,
 'receipt_files':{name:receipt(stage/name) for name in ['MANIFEST.json','SUMMARY.json','COMPACT_ROSTER.json',
  'MANIFEST_BINDING_PROOF.json','ACTUAL_BOUND_RUNTIME_COMPATIBILITY_RECEIPT.json','ABSENT_PROVIDER_FIXTURE_RESULT.json',
  'ACTUAL_LAUNCHER_SOURCE_ACCESS_AUDIT.json','PROTECTED_SOURCE_PRECHECK.json']}}
atomic(stage/'PORTABLE_PREPARATION_PROOF.json',proof)
note="""# Disabled master v5 portable sync preparation (v5)

This manifest binds the current 27-study disabled v5 snapshot. It is not the final ALL plan. Resource qualification and remaining method migrations are pending. No tar was created.

The compatible environment map and final recipe are exact plan-bound inputs. Numerical and compiled ABI pins, CPython 3.11/Linux x86_64, required stdlib capabilities and native CPU operations are mandatory. CPython micro, glibc, SQLite and OpenSSL version strings are provenance. An existing qualifying environment need not be recreated.

The actual frozen launcher loaded all 27 studies and 61,588 records with all source bytes validated. During that call, omitted provider paths and scientific/tensor-deserializer imports were denied. Only plan/spec JSON was decoded. A separate absent-provider fixture passed against the same frozen controller. Required opaque data is hashed for transfer integrity; CAL/evaluation targets were not decoded.

Every host still needs all source/data bindings and the existing Git checkout contract. Absolute author CLI execution remains allocation-only. Server77 access, target GPU/driver qualification, installs, fitting and GO did not occur.

Earlier manifests and failed attempts remain preserved. Final archive creation waits for the parent final-archive instruction after resources and migrations are resolved.
"""
target=stage/'HANDOFF.md';assert not target.exists();target.write_text(note);target.chmod(0o444)
paths=[stage/name for name in ['MANIFEST.json','SUMMARY.json','COMPACT_ROSTER.json','MANIFEST_BINDING_PROOF.json',
 'ACTUAL_BOUND_RUNTIME_COMPATIBILITY_RECEIPT.json','ABSENT_PROVIDER_FIXTURE_RESULT.json',
 'ACTUAL_LAUNCHER_SOURCE_ACCESS_AUDIT.json','PORTABLE_PREPARATION_PROOF.json','HANDOFF.md','SOURCE_INVENTORY.json','PROTECTED_SOURCE_PRECHECK.json']]
print(json.dumps({'proof':proof,'mirror_files':[{'receipt':receipt(p),'data':base64.b64encode(p.read_bytes()).decode()} for p in paths]}),flush=True)
'''


def main():
    out = PACKAGE / "preparation" / LABEL
    if out.exists():
        raise FileExistsError("Preserve prior attempt receipts")
    out.mkdir()
    sources = {
        "build_master_sync_v5.py": PACKAGE / "preparation/build_master_sync_v6.py",
        "audit_master_v5_source_loading.py": PACKAGE / "preparation/audit_master_v5_source_loading.py",
        "refresh_master_sync_v6_allocation.py": Path(__file__).resolve(),
        "tests/test_sync_bundle.py": PACKAGE / "tests/test_sync_bundle.py",
        "tests/test_runtime_compatibility.py": PACKAGE / "tests/test_runtime_compatibility.py",
    }
    files = []
    for relative, source in sources.items():
        raw = source.read_bytes()
        files.append({"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "data": base64.b64encode(raw).decode()})
    encoded = base64.b64encode(zlib.compress(json.dumps({"destination": DESTINATION, "files": files}).encode())).decode()
    (out / "OWNED_SOURCE_INVENTORY.json").write_text(json.dumps([{k: v for k, v in row.items() if k != "data"} for row in files], indent=2) + "\n")
    result = invoke(TRANSFER, [DESTINATION, "0", encoded], out, "TRANSFER", 60)
    if result["exit_code"] != 0:
        print(result["stderr"])
        raise SystemExit(1)
    result = invoke(REMOTE, [DESTINATION, hashlib.sha256(encoded.encode()).hexdigest()], out, "TRANSPORT", 1400)
    for line in result["stdout"].splitlines():
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "mirror_files" in value:
            for row in value["mirror_files"]:
                raw = base64.b64decode(row["data"])
                assert len(raw) == row["receipt"]["bytes"] and hashlib.sha256(raw).hexdigest() == row["receipt"]["sha256"]
                target = out / "allocation_artifacts" / row["receipt"]["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
            print(json.dumps(value["proof"], indent=2))
        else:
            print(json.dumps(value), flush=True)
    if result["exit_code"] != 0:
        print(result["stdout"][-12000:], result["stderr"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
