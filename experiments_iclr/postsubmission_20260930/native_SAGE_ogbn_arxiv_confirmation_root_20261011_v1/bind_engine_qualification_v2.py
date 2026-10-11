"""Bind existing safe complete WikiCS inputs; no selector labels or scores read."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess

R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
H=Path(__file__).resolve().parent
S=P/'native_SAGE_sequential_own_CE_arxiv_source_20261011_v2'
E=P/'native_SAGE_endpoint_value_transport_source_20261011_v2'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'

def item(path):
    p=Path(path).resolve();assert p.is_relative_to(R)
    d=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):d.update(b)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=d.hexdigest())

def write(path,value):
    assert not path.exists();path.write_text(json.dumps(value,indent=2)+'\n')

def main():
    assert socket.gethostname()=='anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[UUID]
    cfg=json.loads((P/'native_SAGE_endpoint_value_execution_root_20261011_v1/QUALIFICATION_CONFIG_V1.json').read_text())
    for name in ('run_family.py','value_installer.py','value_transport.py','contracts.py'):
        key=dict(run_family='training_family',value_installer='value_installer',value_transport='value_helper',contracts='contracts')[Path(name).stem]
        cfg['source_receipts'][key]=item(E/name)
    for row in json.loads((S/'MANIFEST.json').read_text())['files']:
        assert item(S/row['path'])['sha256']==row['sha256']
    import numpy as np
    with np.load(cfg['train_npz'],allow_pickle=False) as z:
        edge=z['edge_index'];assert edge.shape[0]==2
        profile=dict(dataset='WikiCS',nodes=11701,input_dim=300,classes=10,train_rows=580,select_rows=5274,
            edges=edge.shape[1],self_edges=int((edge[0]==edge[1]).sum()),
            graph_support_sha256=hashlib.sha256(edge.tobytes(order='C')).hexdigest(),
            provider_identity_sha256=item(cfg['train_npz'])['sha256'])
    cfg.pop('root_admission',None)
    cfg.update(accumulation='sequential_own_ce',profile=profile,seeds=[8101],
        root_frozen_protocol=dict(study_kind='sequential_engine_qualification',scientific_fits=0,TEST_access=False),
        family_limits=dict(wall_seconds=300,cuda_bytes=32*1024**3,rss_bytes=16*1024**3),
        engineering_tolerances=dict(gradient_atol=1e-5,gradient_rtol=1e-4,update_atol=1e-5,update_rtol=1e-4))
    sources=[item(S/row['path']) for row in json.loads((S/'MANIFEST.json').read_text())['files']]
    sources.extend(item(H/n) for n in ('run_qualification_v2.py','bind_engine_qualification_v2.py','qualification_owner_v2.py'))
    canonical=hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    a=dict(UTC=datetime.now(timezone.utc).isoformat(),action='engine_equivalence_WikiCS',hostname=socket.gethostname(),gpu_uuid=UUID,
        science_config_sha256=canonical,integration_sources=sources,scientific_fits=0,SELECT_CAL_TEST_scoring=False,
        runtime_fingerprint_sha256=item(P/'native_SAGE_endpoint_value_execution_root_20261011_v1/RUNTIME_FINGERPRINT_V1.json')['sha256'])
    write(H/'ENGINE_ADMISSION_V2.json',a);cfg['root_admission']=item(H/'ENGINE_ADMISSION_V2.json')
    write(H/'ENGINE_CONFIG_V2.json',cfg)
    print(json.dumps(dict(complete=True,config=item(H/'ENGINE_CONFIG_V2.json'),profile=profile,scientific_fits=0)))

if __name__=='__main__':main()
