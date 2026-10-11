"""Source/runtime/provider admission for one reference TRAIN-only check."""
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess

R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930';H=Path(__file__).resolve().parent
S=P/'native_CoauthorPhysics_author_meanSAGE_qualification_source_20261011_v1'
C=P/'native_CoauthorPhysics_author_meanSAGE_context_source_20261011_v1'
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
    sources=[]
    for folder in (S,C):
        for row in json.loads((folder/'MANIFEST.json').read_text())['files']:
            v=item(folder/row['path']);assert v['sha256']==row['sha256'];sources.append(v)
    spec=importlib.util.spec_from_file_location('native_source_point_only',C/'author_model.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    cfg=dict(status='ROOT_AUTHOR_CONTEXT_RESOURCE_QUALIFICATION_ADMITTED',context_directory=str(C),seed=8101,device='cuda:0',point=m.POINT,
        profile=dict(dataset='CoauthorPhysics',nodes=34493,input_dim=8415,classes=5,edges=530417,self_edges=34493),
        paths=dict(train=str(H/'prepared_v1/block_8101/train.npz'),select_ids=str(H/'prepared_v1/block_8101/select_ids.npy')),
        limits=dict(wall_seconds=600,cuda_bytes=48*1024**3,rss_bytes=64*1024**3))
    sources.extend(item(q) for q in cfg['paths'].values())
    sources.extend(item(H/n) for n in ('run_author_resources_v1.py','bind_author_resources_v1.py','author_resource_owner_v1.py','prepared_v1/PROVIDER_IDENTITY.json','PROSPECTIVE_SCIENTIFIC_PROTOCOL_V1.json'))
    write(H/'AUTHOR_RESOURCES_CONFIG_V1.json',cfg)
    write(H/'AUTHOR_RESOURCES_ADMISSION_V1.json',dict(UTC=datetime.now(timezone.utc).isoformat(),action='fixed_Physics_author_meanSAGE_resource_qualification',cfg_sha256=item(H/'AUTHOR_RESOURCES_CONFIG_V1.json')['sha256'],bound_files=sources,scientific_fits=0,SELECT_CAL_TEST_scoring=False,finite_external_owner_seconds=600))
    print(json.dumps(dict(complete=True,config=item(H/'AUTHOR_RESOURCES_CONFIG_V1.json'),scientific_fits=0)))
if __name__=='__main__':main()
