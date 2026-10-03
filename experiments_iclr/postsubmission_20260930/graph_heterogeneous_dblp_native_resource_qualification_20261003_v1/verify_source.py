"""Stdlib source/admission checks only; no Torch, dataset labels or remote call."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PHASE=HERE.parent


def require(ok,message):
    if not ok:raise ValueError(message)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value


def main():
    binding=json.loads((HERE/'BINDINGS.json').read_text())
    def local(record):
        path=PHASE/Path(record['path']).relative_to(binding['canonical_phase']);data=path.read_bytes()
        require(hashlib.sha256(data).hexdigest()==record['sha256'] and len(data)==record['bytes'],'Original source/metadata changed: '+str(path))
        return path
    for record in binding['source_records']+[binding['freeze'],binding['paired_HGT_freeze']]:local(record)
    native=PHASE/'graph_heterogeneous_dblp_native_challengers_preparation_20261003_v2'
    require(json.loads((native/'SEAL.json').read_text())['manifest_sha256']==binding['native_manifest_sha256'],'Original native seal differs')
    for path in HERE.glob('*.py'):ast.parse(path.read_text())
    helper=load('native_resource_stdlib_helper',HERE/'run_cpu_remote.py')
    compile(helper.REMOTE,'native_resource_remote_wrapper','exec')
    qualifier=load('native_resource_stdlib_admission',HERE/'qualify_native.py')
    frozen=json.loads(local(binding['freeze']).read_text())
    release=json.loads((HERE/'QUALIFICATION_RELEASE_TEMPLATE.json').read_text())
    release.update(qualification_authorized=True,run_name='source_admission_only',qualification_manifest_sha256='synthetic_manifest')
    environment={key:'1' for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')}
    environment['CUDA_VISIBLE_DEVICES']=''
    rejected=0
    with patch.dict(os.environ,environment),patch.object(qualifier,'verify',side_effect=local):
        qualifier.admission(binding,frozen,release,'synthetic_manifest','source_admission_only')
        invalid=dict(qualification_authorized=False,execution_authorized=True,CPU_fixture_passed=False,device='cuda:0',
            mode='training',run_name='different',qualification_manifest_sha256='different',
            prepared_native_manifest_sha256='different',native_study_freeze_sha256='different',
            paired_HGT_freeze_sha256='different',native_CPU_original_receipt_sha256='different',
            address_space_limit_bytes=8*2**30,RSS_limit_bytes=8*2**30,wall_budget_seconds=1200,threads=2)
        for key,value in invalid.items():
            try:qualifier.admission(binding,frozen,dict(release,**{key:value}),'synthetic_manifest','source_admission_only')
            except ValueError:rejected+=1
            else:raise AssertionError('Invalid root qualification metadata accepted: '+key)
    tree=ast.parse((HERE/'qualify_native.py').read_text())
    driver_calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and isinstance(n.func.value,ast.Name) and n.func.value.id=='driver']
    require(driver_calls.count('train_epoch')==1 and not set(driver_calls)&{'fit','predict_validation','validation_metrics','main'},
        'Only exact native TRAIN epoch and unscored forward may execute')
    require('preexec_fn=bounded' in helper.REMOTE and 'resource.setrlimit(resource.RLIMIT_AS' in helper.REMOTE
        and 'resource.setrlimit(resource.RLIMIT_CPU' in helper.REMOTE,'Preimport qualification bounds absent')
    plan=json.loads((HERE/'EXECUTION_PLAN.json').read_text())
    require(plan['complete_cases']==15 and plan['maximum_total_study_epochs']==4000
        and plan['seeds']==frozen['seeds'] and plan['arms_in_each_seed']==frozen['arms'],'Complete frozen native plan differs')
    own_payloads=None
    if (HERE/'MANIFEST.json').exists():
        manifest_bytes=(HERE/'MANIFEST.json').read_bytes()
        require(json.loads((HERE/'SEAL.json').read_text())['manifest_sha256']==hashlib.sha256(manifest_bytes).hexdigest(),'Own qualification seal differs')
        payload=json.loads(manifest_bytes)['payload']
        for row in payload:
            data=(HERE/row['path']).read_bytes();require(hashlib.sha256(data).hexdigest()==row['sha256'] and len(data)==row['bytes'],'Own source payload differs')
        own_payloads=len(payload)
    require('torch' not in sys.modules,'Source checks imported Torch')
    result=dict(schema='native_DBLP_resource_preparation_stdlib_source_check_v1',status='PASS',
        original_source_records_verified=len(binding['source_records']),root_freezes_verified=2,
        own_syntax_sources=len(list(HERE.glob('*.py'))),remote_wrapper_syntax_verified=True,
        accepted_synthetic_metadata_only_qualification_admission=True,rejected_incompatible_admission_fields=rejected,
        own_sealed_payloads_verified=own_payloads,original_native_manifest_sha256=binding['native_manifest_sha256'],
        native_study_freeze_sha256=binding['freeze']['sha256'],actual_native_CPU_original_receipt_sha256=binding['CPU_original_receipt']['sha256'],
        existing_numerical_fixtures_repeated=False,Torch_imported=False,real_dataset_labels_read=False,
        training_driver_main_called=False,remote_or_GPU_execution=False)
    print(json.dumps(result,sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
