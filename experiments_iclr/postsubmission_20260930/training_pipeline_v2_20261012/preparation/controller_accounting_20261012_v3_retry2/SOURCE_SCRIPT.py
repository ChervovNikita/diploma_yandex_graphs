"""Freeze and test the new controller with synthetic CPU fixtures on allocation."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib

from prepare_molecular_packets_allocation import GUARD, TRANSFER, invoke

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from compile_master_plan import CONTROLLER_FILES

LABEL = 'controller_accounting_20261012_v3_retry2'
HERE = Path(__file__).resolve().parent / LABEL
TESTS = ('controller_final_integration', 'launcher', 'detached_startup', 'memory_admission',
         'phase_closure', 'phase_rendezvous', 'master_plan', 'master_polyformer_ownership',
         'dependency_grouping', 'author_master_integration', 'execution_receipts',
         'paid_continuation', 'sync_bundle', 'landmark_accounting', 'gradient_owner_accounting',
         'original_paper_scope', 'runtime_compatibility', 'engine', 'author_cli', 'orchestration')

REMOTE = GUARD + r'''
encoded=(phase/(sys.argv[1]+'_transport')/'payload.b64').read_text()
assert hashlib.sha256(encoded.encode()).hexdigest()==sys.argv[2]
payload=json.loads(zlib.decompress(base64.b64decode(encoded)))
qa=phase/'training_pipeline_v2_20261012/qa/cpu_controller_accounting_v3_retry2'
assert not qa.exists();qa.mkdir(parents=True)
rows=[]
for row in payload['files']:
 raw=base64.b64decode(row['data']);relative=Path(row['path'])
 assert not relative.is_absolute() and '..' not in relative.parts
 assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 target=qa/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
 rows.append({k:v for k,v in row.items() if k!='data'})
(qa/'SOURCE_INVENTORY.json').write_text(json.dumps(rows,indent=2)+'\n')
tmp=qa/'engineering_tmp';tmp.mkdir()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TMPDIR=str(tmp),PYTHONDONTWRITEBYTECODE='1',
 OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(qa/'tests'))
command=[str(repo/'.venv/bin/python'),'-B','-m','unittest',
 'test_controller_final_integration','test_engine','-v']
start=time.monotonic()
completed=subprocess.run(command,cwd=qa,env=env,capture_output=True,text=True,timeout=120)
result={'command':command,'exit_code':completed.returncode,'stdout':completed.stdout,
 'stderr':completed.stderr,'wall_seconds':time.monotonic()-start,
 'scope':'synthetic_CPU_engineering_only_no_dataset_access_or_scientific_fit'}
(qa/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
assert completed.returncode==0,completed.stderr
assert 'skipped=' not in completed.stderr,completed.stderr
import importlib,re
sys.path.insert(0,str(qa));master=importlib.import_module('compile_master_plan')
stage=phase/'master_controller_source_20261012_v3';assert not stage.exists();stage.mkdir()
runtime=stage/'runtime';runtime.mkdir()
for name in master.CONTROLLER_FILES:
 raw=(qa/name).read_bytes();target=runtime/name;target.write_bytes(raw)
 target.chmod(0o555 if name.endswith('.sh') else 0o444)
def receipt(path):
 raw=path.read_bytes()
 return {'path':str(path.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
prior=json.loads((phase/'master_controller_source_20261012_v2/MANIFEST.json').read_text())
manifest={'schema':'frozen-master-controller-v1','runtime_package':str(runtime.relative_to(phase)),
 'files':master.controller_bindings(phase,str(runtime.relative_to(phase))),
 'engineering_QA':receipt(qa/'RESULT.json'),
 'passed_engineering_tests':int(re.search(r'Ran (\d+) tests',completed.stderr).group(1)),
 'scientific_fits':0,'benchmark_scores':0,'start_commands_are_disabled_until_GO':True,
 'phase_rendezvous':prior['phase_rendezvous'],'host_bindings':prior['host_bindings'],
 'runtime_environment':prior['runtime_environment'],'runtime_preflight_required_before_workers':True,
 'new_scope':'GRU I16, landmark paid sources, CMCL and CORE optimizer ownership, original-five hard exclusions',
 'source_snapshot':receipt(qa/'SOURCE_INVENTORY.json')}
for name,value in [('MANIFEST.json',manifest),('HOST_BINDINGS.json',manifest['host_bindings']),
 ('RENDEZVOUS_POLICY.json',manifest['phase_rendezvous'])]:
 (stage/name).write_text(json.dumps(value,indent=2)+'\n')
receipts=[]
for name in ('Expanded_finite_master_PT_and_compatibility_20261012_v1',
             'Expanded_finite_master_metadata_format_repair_20261012_v1'):
 path=phase/'publication'/name/'PUSH_RECEIPT_SSH22_V1.json'
 raw=path.read_bytes();receipts.append({**receipt(path),'data':base64.b64encode(raw).decode()})
print(json.dumps({'manifest':manifest,'push_receipts':receipts}))
'''


def main():
    assert not HERE.exists()
    HERE.mkdir()
    (HERE/'SOURCE_SCRIPT.py').write_bytes(Path(__file__).read_bytes())
    names=list(CONTROLLER_FILES)+['compile_polyformer_author_study.py']+[
        'tests/test_'+name+'.py' for name in TESTS]
    files=[]
    for name in names:
        raw=(PACKAGE/name).read_bytes()
        files.append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                      'data':base64.b64encode(raw).decode()})
    encoded=base64.b64encode(zlib.compress(json.dumps({'files':files}).encode(),9)).decode()
    for offset in range(0,len(encoded),30000):
        value=invoke(TRANSFER,[LABEL,str(offset),encoded[offset:offset+30000]],HERE,'TRANSFER_'+str(offset),45)
        assert value['exit_code']==0, 'Controller transfer failure retained'
    value=invoke(REMOTE,[LABEL,hashlib.sha256(encoded.encode()).hexdigest()],HERE,'QUALIFY_AND_FREEZE',180)
    assert value['exit_code']==0, 'Controller QA/freeze failure retained'
    report=json.loads(value['stdout'])
    for row in report.pop('push_receipts'):
        raw=base64.b64decode(row.pop('data'))
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        destination=PACKAGE.parent/row['path']
        assert destination.resolve().is_relative_to(PACKAGE.parent.resolve())
        if destination.exists():
            assert destination.read_bytes()==raw
        else:
            destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw)
    (HERE/'MANIFEST.json').write_text(json.dumps(report['manifest'],indent=2)+'\n')
    print(json.dumps({'frozen_runtime':report['manifest']['runtime_package'],
                      'passed_checks':report['manifest']['passed_engineering_tests'],
                      'scientific_training_started':False,'GO_created':False}))


if __name__=='__main__':
    main()
