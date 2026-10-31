"""Source-only checks for the single pinned zlib-file route. No imports/SSH."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
V2=BASE/'industrial_cpu_native_import_qualification_v2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_checks():
    python={}
    for path in sorted(HERE.glob('*.py')):
        tree=ast.parse(path.read_text(),filename=str(path))
        compile(tree,str(path),'exec')
        python[path.name]={'sha256':sha(path),'AST_and_compile':'passed'}
    assert (HERE/'proven_cpu_policy.py').read_bytes()==(V2/'proven_cpu_policy.py').read_bytes()
    for name in ['native_import_worker.py','root_outer.py','root_run_native_imports.py']:
        normalized=(HERE/name).read_text().replace('qualification-v3','qualification-v2').replace('launch-v3','launch-v2')
        assert normalized==(V2/name).read_text()
    receipt=json.loads((HERE/'ZLIB_ACQUISITION_RECEIPT_BOUND.json').read_text())
    original=BASE/'industrial_runtime_zlib_dependency_v1/ACQUISITION_RECEIPT_LOCAL.json'
    assert (HERE/'ZLIB_ACQUISITION_RECEIPT_BOUND.json').read_bytes()==original.read_bytes()
    assert receipt['regular_file_copy'] is True and receipt['all_writes_inside_repo'] is True
    assert receipt['scientific_imports'] is False and receipt['dataset_or_model_access'] is False
    assert receipt['GPU_compute'] is False and receipt['dependency_execution_qualified'] is False
    expected_file={'path':receipt['target_path'],'bytes':receipt['target_bytes'],'sha256':receipt['target_sha256']}
    assert expected_file['bytes']==108936
    assert expected_file['sha256']=='64c206f0146cc58bbddc4f22054436f4ff278f5a554aa3ce6921ddf7e9133370'
    old=json.loads((V2/'IMPORT_CONTRACT.json').read_text())
    new=json.loads((HERE/'IMPORT_CONTRACT.json').read_text())
    assert new['run_container']=='industrial_cpu_native_import_qualification_v3/root_runs'
    assert new['host_readonly_library_files']==old['host_readonly_library_files']+[expected_file]
    parent=str(Path(expected_file['path']).parent)
    assert Path(parent).parent==Path(old['phase'])
    assert new['ld_library_path']==[parent]+old['ld_library_path']
    normalized=dict(new,run_container=old['run_container'],
        host_readonly_library_files=old['host_readonly_library_files'],ld_library_path=old['ld_library_path'])
    assert normalized==old
    assert new['image_readonly_directories']==old['image_readonly_directories']
    assert all('industrial_runtime_zlib_dependency_v1' not in p for p in new['image_readonly_directories'])
    preserved=json.loads((HERE/'PREDECESSORS_PRESERVED_HASHES.json').read_text())
    for item in preserved['files']:
        p=BASE/item['path']
        assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
    manifest=json.loads((BASE/'industrial_runtime_image_root_v1/runtime_image_v1/RUNTIME_IMAGE_MANIFEST.json').read_text())
    assert not any(Path(item['path']).name=='libz.so.1' for item in manifest['payload'])
    root_receipt=json.loads((V2/'root_runs/cpu_native_imports_root02_exception_chain/NATIVE_IMPORT_CAPABILITY.json').read_text())
    errors=root_receipt['worker_result']['exception_diagnostics']['nodes']
    loader=[e for e in errors if e['attributes'].get('name')=='_multiarray_umath']
    assert len(loader)==1 and loader[0]['message']=='libz.so.1: cannot open shared object file: No such file or directory'
    return dict(schema='CPU-native-import-v3-source-checks-v1',status='PASSED_SOURCE_ONLY',python=python,
        policy_byte_identical=True,worker_outer_launcher_identical_except_receipt_schema=True,
        acquisition_receipt_bound_unchanged=True,additional_readonly_files=[expected_file],
        additional_readonly_directories=[],LD_LIBRARY_PATH_single_prepend=parent,
        all_other_contract_fields_identical=True,public_label_archive_boundaries_unchanged=True,
        predecessors_preserved_files=len(preserved['files']),v2_original_loader_error_verified_in_retained_receipt=True,
        libz_so_1_absent_from_existing_image_manifest=True,
        scientific_imports=False,scientific_or_remote_execution=False,
        data_model_or_GPU_access=False,additional_dependency_acquisition=False,
        dependency_execution_qualified=False)


if __name__=='__main__':
    result=run_checks()
    (HERE/'SOURCE_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'additional_readonly_files':1,
        'additional_readonly_directories':0,'policies_and_boundaries_unchanged':True}))
