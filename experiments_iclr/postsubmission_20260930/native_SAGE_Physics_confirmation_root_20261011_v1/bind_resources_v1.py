"""Admit one full wide-feature Physics resource check, not scientific fits."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess

R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930';H=Path(__file__).resolve().parent
S=P/'native_SAGE_sequential_own_CE_Physics_source_20261011_v5'
E=P/'native_SAGE_endpoint_value_transport_source_20261011_v2'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'

def item(path):
    p=Path(path).resolve();assert p.is_relative_to(R)
    d=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):d.update(b)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=d.hexdigest())
def write(p,r):assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\n')

def main():
    assert socket.gethostname()=='anogena-2-0' and Path.cwd()==R
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[UUID]
    for row in json.loads((S/'MANIFEST.json').read_text())['files']:
        assert item(S/row['path'])['sha256']==row['sha256']
    export=json.loads((H/'prepared_v1/CUSTODIAL_EXPORT_RECEIPT.json').read_text())
    owner=json.loads((H/'custodial_export_owner_v1/OWNER_END.json').read_text())
    assert export['complete'] and owner['status']=='COMPLETE_EXIT_0' and owner['child_pid_absent']
    engine=P/'native_SAGE_ogbn_arxiv_confirmation_root_20261011_v1/engine_qualification_v3/ENGINE_EQUIVALENCE.json'
    assert json.loads(engine.read_text())['status']=='PASS'
    cfg=json.loads((P/'native_SAGE_endpoint_value_execution_root_20261011_v1/QUALIFICATION_CONFIG_V1.json').read_text())
    cfg.pop('root_admission',None)
    for name,key in (('run_family.py','training_family'),('value_installer.py','value_installer'),('value_transport.py','value_helper'),('contracts.py','contracts')):
        cfg['source_receipts'][key]=item(E/name)
    seeds=[8101,8219,8363,8501,8663,8819,9001,9199,9403,9623]
    role_blocks={str(seed):dict(train_npz=str(H/f'prepared_v1/block_{seed}/train.npz'),select_npz=str(H/f'prepared_v1/block_{seed}/select.npz')) for seed in seeds}
    cfg.update(accumulation='sequential_own_ce',profile=export['profiles']['8101'],seeds=seeds,active_role_seed=8101,role_blocks=role_blocks,
        train_npz=role_blocks['8101']['train_npz'],valid_npz=role_blocks['8101']['select_npz'],
        label_free_SELECT_ids=str(H/'prepared_v1/block_8101/select_ids.npy'),
        root_frozen_protocol=json.loads((H/'PROSPECTIVE_SCIENTIFIC_PROTOCOL_V1.json').read_text()),
        family_limits=dict(wall_seconds=900,cuda_bytes=48*1024**3,rss_bytes=64*1024**3))
    cfg['safe_roles']={k:item(cfg[k+'_npz']) for k in ('train','valid')}
    sources=[item(S/row['path']) for row in json.loads((S/'MANIFEST.json').read_text())['files']]
    sources.extend(item(H/n) for n in ('run_qualification_v1.py','bind_resources_v1.py','qualification_owner_v1.py'))
    sources.extend(item(block[k]) for block in role_blocks.values() for k in ('train_npz','select_npz'))
    sources.append(item(cfg['label_free_SELECT_ids']))
    sources.extend(item(H/n) for n in ('PROSPECTIVE_SCIENTIFIC_PROTOCOL_V1.json','prepared_v1/PROVIDER_IDENTITY.json','prepared_v1/CUSTODIAL_EXPORT_RECEIPT.json'))
    canonical=hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    a=dict(UTC=datetime.now(timezone.utc).isoformat(),action='physics_sequential_resource',hostname=socket.gethostname(),gpu_uuid=UUID,
        science_config_sha256=canonical,integration_sources=sources,engine_receipt_sha256=item(engine)['sha256'],
        scientific_fits=0,SELECT_CAL_TEST_scoring=False,
        runtime_fingerprint_sha256=item(P/'native_SAGE_endpoint_value_execution_root_20261011_v1/RUNTIME_FINGERPRINT_V1.json')['sha256'],
        overlap='Existing unchanged Penn94 on sole authorized GPU; this check is not an isolated speed experiment')
    write(H/'RESOURCES_ADMISSION_V1.json',a);cfg['root_admission']=item(H/'RESOURCES_ADMISSION_V1.json')
    write(H/'RESOURCES_CONFIG_V1.json',cfg)
    print(json.dumps(dict(complete=True,config=item(H/'RESOURCES_CONFIG_V1.json'),profile=cfg['profile'],scientific_fits=0,TEST_scoring=False)))
if __name__=='__main__':main()
