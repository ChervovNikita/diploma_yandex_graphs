"""Root integration of two bounded qualifications, without scientific scoring."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess

R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
H=Path(__file__).resolve().parent
S=P/'native_SAGE_sequential_own_CE_Physics_source_20261011_v5'
E=P/'native_SAGE_endpoint_value_transport_source_20261011_v2'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'

def sha(p):
    d=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):d.update(b)
    return d.hexdigest()

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def verify_route():
    assert socket.gethostname()=='anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==[UUID]
    assert Path.cwd()==R and H.is_relative_to(R)

def root_validate(cfg,*,action,executing_source,port_receipt):
    verify_route()
    a=json.loads(Path(cfg['root_admission']['path']).read_text())
    assert sha(cfg['root_admission']['path'])==cfg['root_admission']['sha256']
    assert a['action']==action and a['hostname']==socket.gethostname() and a['gpu_uuid']==UUID
    canonical=hashlib.sha256(json.dumps({k:v for k,v in cfg.items() if k!='root_admission'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert a['science_config_sha256']==canonical
    for k,v in cfg['source_receipts'].items():
        q=Path(v['path']).resolve();assert q.is_relative_to(R) and sha(q)==v['sha256']
    for k,v in cfg['safe_roles'].items():
        q=Path(v['path']).resolve();assert q.is_relative_to(R) and sha(q)==v['sha256']
    for v in a['integration_sources']:
        q=Path(v['path']).resolve();assert q.is_relative_to(R) and sha(q)==v['sha256']
    assert Path(executing_source)==E/'run_family.py'
    assert sha(S/'port.py')==port_receipt['adapter_sha256']
    assert sha(S/'own_ce.py')==port_receipt['own_ce_sha256']
    if cfg['profile']['dataset']=='CoauthorPhysics':
        engine=P/'native_SAGE_ogbn_arxiv_confirmation_root_20261011_v1/engine_qualification_v3/ENGINE_EQUIVALENCE.json'
        assert sha(engine)==a['engine_receipt_sha256']
        assert json.loads(engine.read_text())['status']=='PASS'
    return dict(stage='F',K=None,pilot_provenance_read=False,study_kind='native_SAGE_sequential_own_CE_qualification',action=action,
        source_sha256={v['path']:v['sha256'] for v in cfg['source_receipts'].values()},
        safe_role_sha256={k:v['sha256'] for k,v in cfg['safe_roles'].items()},root_runtime_admission=a,baseline_policy='fresh_required')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['resources'],required=True);args=ap.parse_args()
    verify_route()
    config=H/(args.stage.upper()+'_CONFIG_V1.json');cfg=json.loads(config.read_text())
    release=dict(status='ROOT_ENGINE_EQUIVALENCE_ADMITTED' if args.stage=='engine' else 'ROOT_PHYSICS_RESOURCE_QUALIFICATION_ADMITTED')
    if args.stage=='resources':release['engine_equivalence']='ROOT_VERIFIED_PASS'
    port=load('root_sequential_port',S/'port.py');qualify=load('root_sequential_qualify',S/'qualify.py')
    Family,receipt=port.bind(E,cfg,root_validate,release)
    def factory(output):return Family(cfg,output,sha(config),training_only=True)
    if args.stage=='engine':
        result=qualify.equivalence(factory,H/'engine_qualification_v3',seed=cfg['seeds'][0],release=release,
            limits=cfg['family_limits'],tolerances=cfg['engineering_tolerances'])
    else:
        import numpy as np
        import torch
        ids=torch.as_tensor(np.load(cfg['label_free_SELECT_ids'],allow_pickle=False),dtype=torch.long,device=cfg['device'])
        result=qualify.resource_integration(factory,ids,H/'resources_qualification_v1',seed=cfg['active_role_seed'],release=release,limits=cfg['family_limits'],dataset='CoauthorPhysics')
    print(json.dumps(dict(status=result['status'],dataset=result['dataset'],seconds=result['elapsed_seconds'],
        cuda_peak_allocated_bytes=result['cuda_peak_allocated_bytes'],scientific_fits=0,TEST_scoring=False)))

if __name__=='__main__':main()
