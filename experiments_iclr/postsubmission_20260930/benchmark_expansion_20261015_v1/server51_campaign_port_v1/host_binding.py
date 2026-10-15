"""Explicit allocation/server51 host binding; no model imports or PID whitelist."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import socket
import subprocess
import sys
HERE=Path(__file__).resolve().parent
WORKERS=('allocation/gpu0','server51/gpu0','server51/gpu1')
ALLOCATION_ROOT='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
SERVER51_ROOT='/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'
PHASE_RELATIVE='experiments_iclr/postsubmission_20260930'
ALLOCATION_UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SERVER51_UUIDS=['GPU-9c6c20fa-a620-479d-f72f-163ac74deef2','GPU-30283973-140d-9e7b-e780-0fcc487dec54']
PACKAGE_RELATIVE='benchmark_expansion_20261015_v1/server51_campaign_port_v1'
def require(ok,message):
    if not ok:raise ValueError(message)
def normalized_absolute(value):
    require(isinstance(value,str) and value.startswith('/'),'Absolute POSIX path required')
    path=PurePosixPath(value);require(str(path)==value and '..' not in path.parts,'Normalized path required');return path
def confined(path,root):
    path=Path(path).resolve();require(path.is_relative_to(Path(root).resolve()),'Path escapes project root');return path
def pin(path):
    path=Path(path).resolve();raw=path.read_bytes();return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def verify(value,root,parse_json=True):
    path=confined(value['path'],root);actual=pin(path)
    require(all(actual[k]==value[k] for k in ('bytes','sha256') if k in value),'Receipt bytes changed: '+str(path))
    return json.loads(path.read_text()) if parse_json else actual
def validate_config(c):
    require(c['schema']=='allocation-server51-campaign-host-v1' and c['worker'] in WORKERS,'Unknown host schema/worker')
    index=int(c['worker'][-1]);host=c['worker'].split('/')[0]
    root=ALLOCATION_ROOT if host=='allocation' else SERVER51_ROOT
    uuids=[ALLOCATION_UUID] if host=='allocation' else SERVER51_UUIDS
    require(c['project_root']==root and c['phase_relative']==PHASE_RELATIVE,'Verified roots/layout are fixed')
    require(c['expected_hostname']==('anogena-2-0' if host=='allocation' else 'olig'),'Exact verified host required')
    require(c['expected_gpu_uuids_by_index']==uuids and c['physical_gpu_index']==index and c['physical_gpu_uuid']==uuids[index],'Physical device identity differs')
    require(c['logical_device']=='cuda:0' and c['maximum_children_per_physical_GPU']==2,'Keep masked logical cuda:0/two slots')
    require(c['host_enabled'] is True and c['campaign_GO'] is False,'Identity config never grants campaign GO')
    return c
def config_path(worker):
    require(worker in WORKERS,'Unknown physical worker');return HERE/'hosts'/(worker.replace('/','_')+'.json')
def load_config(path=None,expected_sha256=None):
    path=Path(path or os.environ.get('BENCHMARK_HOST_CONFIG',config_path('allocation/gpu0'))).resolve()
    require(path.parent==HERE/'hosts','Only bundled reviewed host config is allowed')
    actual=pin(path);expected=expected_sha256 or os.environ.get('BENCHMARK_HOST_CONFIG_SHA256')
    if expected:require(actual['sha256']==expected,'Host configuration changed')
    c=validate_config(json.loads(path.read_text()));require(path==config_path(c['worker']),'Config filename/worker mismatch');return c,actual
def check_observed_identity(c,hostname,uuids):
    validate_config(c);require(hostname==c['expected_hostname'],'Unexpected scientific hostname')
    require(uuids==c['expected_gpu_uuids_by_index'],'Unexpected complete GPU identity/index order')
def guard(config=None):
    c=load_config()[0] if config is None else validate_config(config)
    require(socket.gethostname()==c['expected_hostname'],'Wrong host; no further project reads')
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()
    check_observed_identity(c,socket.gethostname(),uuids)
    require(HERE==Path(c['project_root'])/c['phase_relative']/PACKAGE_RELATIVE,'Guard source is outside declared deployment')
    mask=os.environ.get('CUDA_VISIBLE_DEVICES')
    require(mask==c['physical_gpu_uuid'] or(c['worker']=='allocation/gpu0' and mask=='0'),'Exact physical UUID mask required')
    return c
def bound_project_root(path):
    c,_=load_config();root=Path(c['project_root']).resolve();require(Path(path).resolve()==root,'Runtime root differs');return root
def _bindings():
    name='_server51_execution_descriptor_binding';path=HERE/'descriptor_binding.py'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]
def validate_host_job(job,root):
    c,_=load_config();require(Path(root).resolve()==Path(c['project_root']).resolve(),'Wrong host descriptor root')
    require(job.get('TEST_access') is False and job.get('device')=='cuda:0','TEST remains closed/logical device fixed')
    for row in _bindings().audit_runtime_paths(job,c['project_root'],c['phase_relative']):confined(row['path'],root)
    if job.get('normalization_composition'):
        decl=job['normalization_composition'];parent=verify(decl['parent_descriptor'],root)
        E=HERE.parent;source=E/'composable_normalization_v2';require(pin(source/'MANIFEST.json')['sha256']==decl['source_manifest_sha256'],'Normalization source differs')
        spec=importlib.util.spec_from_file_location('_server51_normalization_contract',source/'contracts.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        roster=json.loads((E/'expanded_study_design_v1/PROPOSED_ROSTER.json').read_text())
        require(job==module.migrated_job(parent,decl,roster),'Execution composition differs from actual rebound parent')
