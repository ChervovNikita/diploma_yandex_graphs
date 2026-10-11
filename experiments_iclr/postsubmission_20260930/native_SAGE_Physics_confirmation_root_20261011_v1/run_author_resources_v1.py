"""One actual full-graph native-reference resource check before scientific fits."""
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess

R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930';H=Path(__file__).resolve().parent
S=P/'native_CoauthorPhysics_author_meanSAGE_qualification_source_20261011_v1'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
def sha(p):
    d=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):d.update(b)
    return d.hexdigest()
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def validate(cfg,*,action,source_receipts):
    assert socket.gethostname()=='anogena-2-0' and Path.cwd()==R
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[UUID]
    a=json.loads((H/'AUTHOR_RESOURCES_ADMISSION_V1.json').read_text())
    assert a['action']==action and a['cfg_sha256']==sha(H/'AUTHOR_RESOURCES_CONFIG_V1.json')
    for v in a['bound_files']:
        q=Path(v['path']).resolve();assert q.is_relative_to(R) and sha(q)==v['sha256']
    for v in source_receipts.values():
        assert sha(v['path'])==v['sha256'] and v in a['bound_files']
    return dict(status='ROOT_AUTHOR_CONTEXT_RESOURCE_QUALIFICATION_ADMITTED',action=action,admission_sha256=sha(H/'AUTHOR_RESOURCES_ADMISSION_V1.json'),hostname=socket.gethostname(),gpu_uuid=UUID,scientific_fits=0,TEST_scoring=False)
def main():
    cfg=json.loads((H/'AUTHOR_RESOURCES_CONFIG_V1.json').read_text())
    m=load('one_native_resource_qualifier',S/'qualify.py')
    result=m.qualify(cfg,H/'author_resources_qualification_v1',validate)
    print(json.dumps(dict(status=result['status'],scientific_fits=0,TEST_scoring=False)))
if __name__=='__main__':main()
