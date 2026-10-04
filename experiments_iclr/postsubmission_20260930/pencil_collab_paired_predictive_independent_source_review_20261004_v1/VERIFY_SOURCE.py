"""Independent local stdlib source/hash verification. Never execute candidate code."""
import ast
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
CANDIDATE = ROOT/'pencil_collab_paired_predictive_preparation_20261004_v1'
PARENT = ROOT/'pencil_collab_resource_qualifier_preparation_20261004_v3'
EXPECTED_MANIFEST = '1a22966029e9a1b4b4affad7e5cc24f6b3d795245a8d3fc8d0ac12af4b2983c2'
EXPECTED_SEAL = 'ca3099d5116e3aa8a06dfe87678dc0dc370a9a2ed27c709ac12bc7c0c95818f8'
ALLOWED_TEXT_SUFFIXES = {'.py','.json','.md','.yaml','.diff','.txt',''}
rows = {}

def digest(b):
    return hashlib.sha256(b).hexdigest()

def source(path, scope, expected=None):
    assert path.resolve().is_relative_to(ROOT) and not path.is_symlink()
    assert path.is_file() and path.suffix in ALLOWED_TEXT_SUFFIXES, str(path)
    value = path.read_bytes()
    value.decode('utf-8')
    row = {'path':str(path.relative_to(ROOT)), 'bytes':len(value), 'sha256':digest(value)}
    if expected is not None:
        assert len(value)==expected['bytes'] and row['sha256']==expected['sha256'],str(path)
    stored = rows.setdefault(row['path'],dict(row,scopes=[]))
    assert all(stored[k]==row[k] for k in ('path','bytes','sha256'))
    if scope not in stored['scopes']: stored['scopes'].append(scope)
    return value

def json_source(path, scope):
    return json.loads(source(path,scope))

def write(name, value):
    path=OUT/name
    assert not path.exists(),name
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')

mb=source(CANDIDATE/'MANIFEST.json','candidate_manifest')
sb=source(CANDIDATE/'SEAL.json','candidate_seal')
assert digest(mb)==EXPECTED_MANIFEST and digest(sb)==EXPECTED_SEAL
manifest=json.loads(mb);seal=json.loads(sb)
assert seal['manifest_sha256']==EXPECTED_MANIFEST
actual={str(p.relative_to(CANDIDATE)) for p in CANDIDATE.rglob('*') if p.is_file() and p.name not in ('MANIFEST.json','SEAL.json')}
assert actual=={r['path'] for r in manifest['files']}
parsed={}
for row in manifest['files']:
    p=CANDIDATE/row['path'];b=source(p,'candidate_payload',row)
    if p.suffix=='.py': parsed[row['path']]=ast.parse(b.decode(),filename=str(p))
assert len(parsed)==25
assert all(stat.S_IMODE(p.stat().st_mode)==0o444 for p in CANDIDATE.rglob('*') if p.is_file())
assert stat.S_IMODE(CANDIDATE.stat().st_mode)==0o555
assert all(stat.S_IMODE(p.stat().st_mode)==0o555 for p in CANDIDATE.rglob('*') if p.is_dir())
inputs=json_source(CANDIDATE/'INPUT_BINDINGS.json','input_binding_control')['inputs']
for row in inputs: source(ROOT/row['path'],'candidate_bound_external_text',row)
# The historical TEST summary is digest-bound only. Its scientific values are not parsed or used.
native=json_source(CANDIDATE/'NATIVE_SOURCE_BINDINGS.json','native_binding_control')['files']
assert len(native)==20
for row in native:
    b=source(CANDIDATE/row['path'],'native_candidate',row)
    origin=source(ROOT/row['origin'],'native_origin_text')
    assert b==origin
    assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==row['git_blob_sha1']
pm=json_source(PARENT/'MANIFEST.json','parent_manifest')
source(PARENT/'SEAL.json','parent_seal')
for row in pm['files']: source(PARENT/row['path'],'parent_source_closure',row)
retained=['data_adapter.py','NATIVE_SOURCE_BINDINGS.json','ACTUAL_DEPENDENCY_BINDING.json','OFFICIAL_CONFIG_REFERENCE.yaml','metadata/config.json','metadata/DATA_AUTHORITY.json','metadata/RUNTIME_AUTHORITY.json','native/.project-root']+[r['path'] for r in native]
for name in retained: assert source(CANDIDATE/name,'retained_prerequisite')==source(PARENT/name,'retained_parent_prerequisite')
provenance=json_source(CANDIDATE/'PROVENANCE.json','provenance_control')
regenerated=''
for change in provenance['changed_existing_harness_and_controls']:
    a=source(ROOT/change['before']['path'],'diff_before',change['before']).decode().splitlines(keepends=True)
    b=source(ROOT/change['after']['path'],'diff_after',change['after']).decode().splitlines(keepends=True)
    regenerated+=''.join(difflib.unified_diff(a,b,fromfile=change['before']['path'],tofile=change['after']['path']))
assert regenerated==source(CANDIDATE/'V3_TO_SCIENTIFIC.diff','reviewed_exact_diff').decode()
old_ast=ast.parse(source(PARENT/'supervise.py','inherited_supervisor_text').decode())
old_functions={n.name:n for n in old_ast.body if isinstance(n,ast.FunctionDef)}
new_functions={n.name:n for n in parsed['supervise.py'].body if isinstance(n,ast.FunctionDef)}
helpers=['fsync_directory','durable_write','process_identity','members_of_session','held_identity','kill_owned','read_json','inventory']
for name in helpers: assert ast.dump(old_functions[name])==ast.dump(new_functions[name])
plan=json_source(CANDIDATE/'PLAN.json','reviewed_plan')
command=json_source(CANDIDATE/'PROPOSED_COMMAND.json','reviewed_command')
release=json_source(CANDIDATE/'ROOT_RELEASE_TEMPLATE.json','disabled_release')
assert plan['seeds']==[0,1,2] and plan['workload']['native_epoch_indices']==list(range(20))
assert plan['workload']['native_epochs']==20 and plan['workload']['world_size']==1
assert command['execution_order']==[0,1,2] and command['maximum_concurrent_fits']==1
assert [r['seed'] for r in command['commands']]==[0,1,2]
assert release['status']=='DISABLED_TEMPLATE_NOT_AUTHORIZATION' and release['plan_sha256']==digest(source(CANDIDATE/'PLAN.json','release_plan_binding'))
evidence=json_source(CANDIDATE/'evidence/V3_RESOURCE_SUMMARY.json','adopted_resource_text')
adoption=json_source(CANDIDATE/'evidence/V3_ROOT_RESOURCE_ADOPTION.json','root_adoption_text')
assert evidence['adoption_eligible'] is True and evidence['physical_exit_code']==0 and evidence['physical_session_closed'] is True
assert evidence['supervisor_identity'] is None and evidence['resource']['predictive_metrics_computed'] is False
assert adoption['status']=='ROOT_ADOPTED_COMPLETE_RESOURCE_ONLY' and adoption['state_donation'] is False and adoption['predictive_result'] is False
binding=json_source(CANDIDATE/'ACTUAL_DEPENDENCY_BINDING.json','existing_installation_binding')
install=source(ROOT/binding['monitor']['path'],'installation_monitor_text',binding['monitor'])
install=json.loads(install)
assert install['DISTRIBUTION_ADMISSION.json']['value']['versions']==plan['distribution_versions']
for key in ('DISTRIBUTION_ADMISSION','INSTALL_RESULT'):
    remote=install[key+'.json'];bound=binding[key]
    assert remote['bytes']==bound['bytes'] and remote['sha256']==bound['sha256']
assert install['INSTALL_RESULT.json']['value']['inventory_sha256']==binding['INSTALLED_FILE_INVENTORY']['sha256']
assert install['INSTALL_RESULT.json']['value']['installed_files']==binding['INSTALLED_FILE_INVENTORY']['files']
assert not {'torch','numpy','pandas','transformers'}.intersection(sys.modules)
source_manifest={'schema':'independent-pencil-technical-source-review-input-manifest-v1','candidate_manifest_sha256':EXPECTED_MANIFEST,'candidate_seal_sha256':EXPECTED_SEAL,'execution_authorized':False,'scope':'UTF-8 source/configuration/diff/control/resource metadata only; historical TEST summary digest only; no scientific binary, graph, score-array or checkpoint payload access','files':sorted(rows.values(),key=lambda r:r['path'])}
write('SOURCE_MANIFEST.json',source_manifest)
result={'schema':'independent-local-stdlib-source-verification-v1','status':'PASS_SOURCE_HASH_AST_AND_BINDINGS_ONLY','UTC':datetime.now(timezone.utc).isoformat(),'candidate_manifest_sha256':EXPECTED_MANIFEST,'candidate_seal_sha256':EXPECTED_SEAL,'candidate_payload_files_verified':len(manifest['files']),'candidate_files_mode':'0444','candidate_directories_mode':'0555','python_AST_parsed_files':len(parsed),'exact_external_input_bindings_verified':len(inputs),'native_origin_and_git_blob_matches':len(native),'parent_payload_files_verified':len(pm['files']),'regenerated_diff_exact':True,'inherited_ownership_functions_AST_equal':helpers,'retained_prerequisites_byte_identical':True,'installation_monitor_metadata_binding_matches':True,'candidate_or_native_code_executed':False,'numerical_modules_imported':False,'scientific_execution_performed':False,'server_actions':False,'graph_score_array_checkpoint_payload_reads':False,'TEST_data_reads':False,'historical_TEST_summary_values_parsed_or_used':False,'execution_authorized':False,'limitations':'Source verification only. Remote interpreter, installed inventory, numerical behavior and scientific completion are not re-executed here.'}
write('VERIFY_SOURCE_RESULTS.json',result)
print(json.dumps(result,indent=2,sort_keys=True))
