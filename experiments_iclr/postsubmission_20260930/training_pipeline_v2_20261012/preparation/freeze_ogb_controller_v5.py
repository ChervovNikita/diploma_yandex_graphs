"""Freeze the OGB author extension without modifying any earlier controller."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib
from prepare_molecular_packets_allocation import GUARD, TRANSFER, invoke

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from compile_master_plan import CONTROLLER_FILES
LABEL='controller_ogb_family_root_20261012_v5'

REMOTE=GUARD+r'''
encoded=(phase/(sys.argv[1]+'_transport')/'payload.b64').read_text()
assert hashlib.sha256(encoded.encode()).hexdigest()==sys.argv[2]
payload=json.loads(zlib.decompress(base64.b64decode(encoded)))
stage=phase/'master_controller_source_20261012_v5';assert not stage.exists();stage.mkdir()
runtime=stage/'runtime';runtime.mkdir();qa=stage/'engineering';qa.mkdir()
for row in payload['files']:
 relative=Path(row['path']);assert not relative.is_absolute() and '..' not in relative.parts
 raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 path=(qa if str(relative).startswith('tests/') else runtime)/relative
 path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
 path.chmod(0o555 if path.suffix=='.sh' else 0o444)
prior=json.loads((phase/'master_controller_source_20261012_v4/MANIFEST.json').read_text())
unchanged=[];changed=[]
for row in prior['files']:
 name=Path(row['path']).name;old=(phase/row['path']).read_bytes();assert hashlib.sha256(old).hexdigest()==row['sha256']
 if old==(runtime/name).read_bytes():unchanged.append(name)
 else:changed.append(name)
assert set(changed)=={'author_cli_spec.py','run_author_cli.py','compile_master_plan.py'}
assert len(unchanged)==14
import ast
old=ast.parse((phase/'master_controller_source_20261012_v4/runtime/compile_master_plan.py').read_text())
new=ast.parse((runtime/'compile_master_plan.py').read_text())
def omit_controller_files(tree):
 return ast.dump(ast.Module(body=[n for n in tree.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONTROLLER_FILES' for t in n.targets))],type_ignores=[]),include_attributes=False)
assert omit_controller_files(old)==omit_controller_files(new)
tmp=qa/'tmp';tmp.mkdir()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(tmp),
 PYTHONPATH=os.pathsep.join([str(runtime),str(qa/'tests')]),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
command=[str(repo/'.venv/bin/python'),'-B','-m','unittest','test_ogb_master_extension','-v']
start=time.monotonic();result=subprocess.run(command,cwd=runtime,env=env,capture_output=True,text=True,timeout=60)
evidence={'command':command,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
 'wall_seconds':time.monotonic()-start,'scientific_fits':0,'CUDA_executions':0,'GO_created':False}
(qa/'RESULT.json').write_text(json.dumps(evidence,indent=2)+'\n')
assert result.returncode==0 and 'Ran 2 tests' in result.stderr and '\nOK\n' in result.stderr
import importlib
sys.path.insert(0,str(runtime));master=importlib.import_module('compile_master_plan')
def receipt(path):
 raw=path.read_bytes();return {'path':str(path.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
author=phase/'ogb_author_family_implementation_delivery_20261012_v1/ALLOCATION_ENGINEERING_CHECK.json'
author_proof=json.loads(author.read_text());assert author_proof['scientific_fits']==0 if 'scientific_fits' in author_proof else author_proof['actual_author_fits']==0
manifest={'schema':'frozen-master-controller-v1','runtime_package':str(runtime.relative_to(phase)),
 'files':master.controller_bindings(phase,str(runtime.relative_to(phase))),
 'engineering_QA':receipt(qa/'RESULT.json'),'passed_engineering_tests':2,
 'unchanged_controller_files':unchanged,'changed_existing_files':changed,
 'common_master_semantics_AST_unchanged_except_controller_file_catalogue':True,
 'prior_controller_QA':prior['engineering_QA'],'OGB_author_CPU_QA':receipt(author),
 'prior90_plus8_CPU_checks_inherited_except_explicit_new_author_dispatch':True,
 'native_learning_and_standardized_model_engine_changed':False,
 'scientific_fits':0,'benchmark_scores':0,'start_commands_are_disabled_until_GO':True,
 'phase_rendezvous':prior['phase_rendezvous'],'host_bindings':prior['host_bindings'],
 'runtime_environment':prior['runtime_environment'],'runtime_preflight_required_before_workers':True,
 'new_scope':'dedicated_OGB_author_family_with_native_CLI_and_explicit_separate_seed_policy'}
assert len(manifest['files'])==20
for name,value in [('MANIFEST.json',manifest),('HOST_BINDINGS.json',manifest['host_bindings']),('RENDEZVOUS_POLICY.json',manifest['phase_rendezvous'])]:
 (stage/name).write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(manifest))
'''


def main():
    out=PACKAGE/'preparation'/LABEL;out.mkdir(exist_ok=False)
    files=[]
    for name in list(CONTROLLER_FILES)+['tests/test_ogb_master_extension.py','tests/test_ogb_author_cli.py']:
        raw=(PACKAGE/name).read_bytes()
        files.append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
    encoded=base64.b64encode(zlib.compress(json.dumps({'files':files}).encode(),9)).decode()
    for offset in range(0,len(encoded),30000):
        result=invoke(TRANSFER,[LABEL,str(offset),encoded[offset:offset+30000]],out,'TRANSFER_'+str(offset),60)
        assert result['exit_code']==0
    result=invoke(REMOTE,[LABEL,hashlib.sha256(encoded.encode()).hexdigest()],out,'QUALIFY_AND_FREEZE',120)
    assert result['exit_code']==0, result['stdout']
    value=json.loads(result['stdout']);(out/'MANIFEST.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'frozen_runtime':value['runtime_package'],'controller_files':len(value['files']),
                      'new_focused_CPU_tests':2,'unchanged_existing_files':len(value['unchanged_controller_files']),
                      'scientific_fits':0,'GO_created':False}))


if __name__=='__main__':main()
