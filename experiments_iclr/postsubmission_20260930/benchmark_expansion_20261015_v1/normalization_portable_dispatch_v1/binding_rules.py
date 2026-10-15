"""Exact composed entrypoint and same-allocation binding; no host relocation."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import bootstrap

E=Path(__file__).resolve().parent.parent
NORMALIZATION_SHA='cedb1bb2125f6d5ea78ea136d26ce17ac4ebbd33bd806b88545d270989c9eb5d'
PRODUCER_SHA='518784a9e83f4ba92c4f7bff55705999a787c7a67ff69c0cc9c09b5530550d38'
BANKS_SHA='d90a74bf98be32408cad306f35ab5ebd1ec80153981a0b7f92cba07e7f447fea'
ROSTER_SHA='e6a36dee1adb30e0fdb323ccbf802195230d7f7e0e7f00e52b9402c66a602e80'
ROOT_LITERAL='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
PHASE_LITERAL='experiments_iclr/postsubmission_20260930'


def require(value,message):
    if not value:raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def pin(path):
    path=Path(path).resolve();b=path.read_bytes()
    return {'path':str(path),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def json_pin(value,root):
    path=Path(value['path']).resolve()
    require(path.is_relative_to(Path(root).resolve())and path.suffix=='.json','Confined JSON metadata required')
    require(pin(path)==value,'Exact metadata receipt changed')
    return json.loads(path.read_text())


def expected_entrypoint(task,root):
    if task['kind']=='fit':
        job=json_pin(task['source_descriptor'],root)
        require(job['id']==task['id'],'Task and canonical JOB differ')
        if job.get('normalization_composition')is not None:
            return E/'composable_normalization_runtime_v1/runtime/run_fit.py'
    family=task.get('family')if task['kind']=='fit'else'I4'
    return E/'portable_dispatcher_v3'/('runtime_link/run_fit.py'if family=='link'else'runtime_I4/aggregate_selected.py'if family=='I4'else'runtime_node_molecular/run_fit.py')


def normalized_execution_binding(job,root,phase_relative):
    require(str(root)==ROOT_LITERAL and phase_relative==PHASE_LITERAL,
            'Composed execution is currently allocation-only; explicit parent relocation is not implemented')
    declaration=job.get('normalization_composition',{})
    require(declaration.get('source_manifest_sha256')==NORMALIZATION_SHA and declaration.get('banks_sha256')==BANKS_SHA,
            'Exact frozen normalization source and bank law required')
    return {'schema':'composed-normalization-same-allocation-execution-v1','id':job['id'],
        'canonical_runtime_job_hash':digest(job),'bound_runtime_job_hash':digest(job),
        'science_contract_hash':job['expanded_contract_hash'],'field_changes':[],
        'parent_descriptor':copy.deepcopy(declaration['parent_descriptor']),
        'parent_runtime_job_hash':declaration['parent_job_sha256'],
        'normalization_is_part_of_canonical_scientific_contract':True,
        'scientific_fields_changed_by_execution_binding':False,'launch_admitted':False,
        'destination_source_or_data_bytes_verified':False}


def _module(path,name):
    if name in sys.modules:
        require(Path(sys.modules[name].__file__).resolve()==Path(path).resolve(),'Conflicting metadata module identity')
        return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value


def verify_composed_job(job,root,source_files):
    """Composition authority stays additional to the unchanged native gate checks."""
    require(job.get('normalization_composition')is not None and job.get('TEST_access')is False,'Declared no-TEST composition required')
    for folder,expected in [('composable_normalization_v2',NORMALIZATION_SHA),('composable_normalization_runtime_v1',PRODUCER_SHA)]:
        manifest=E/folder/'MANIFEST.json';require(source_files.get(str(manifest))==expected and pin(manifest)['sha256']==expected,
            'Frozen composed producer/normalization source absent from closure')
        for row in json.loads(manifest.read_text())['files']:
            path=(manifest.parent/row['path']).resolve();require(path.is_relative_to(manifest.parent),'Manifest member escapes source')
            require(source_files.get(str(path))==row['sha256'],'Composed source member absent from reviewed closure')
    bank=E/'private_bn_state_v3/banks.py';roster_path=E/'expanded_study_design_v1/PROPOSED_ROSTER.json'
    require(source_files.get(str(bank))==BANKS_SHA and pin(bank)['sha256']==BANKS_SHA,'Bank law source differs')
    require(source_files.get(str(roster_path))==ROSTER_SHA and pin(roster_path)['sha256']==ROSTER_SHA,'Frozen applicability roster differs')
    parent=json_pin(job['normalization_composition']['parent_descriptor'],root)
    contracts=_module(E/'composable_normalization_v2/contracts.py','_normalization_dispatch_composition_contracts')
    roster=json.loads(roster_path.read_text())
    require(job==contracts.migrated_job(parent,job['normalization_composition'],roster),'Canonical composed JOB changed a parent scientific field')
    typed=_module(E/'composable_normalization_runtime_v1/descriptor_binding.py','_normalization_dispatch_typed_paths')
    typed.audit_runtime_paths(job,str(Path(root).resolve()),PHASE_LITERAL)
    return True
