"""Bind actual native source, safe Wiki roles and runtime before qualification."""
import hashlib
import importlib
import json
from pathlib import Path
import socket
import subprocess
from datetime import datetime, timezone

R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P = R / 'experiments_iclr/postsubmission_20260930'
H = P / 'native_SAGE_endpoint_value_execution_root_20261011_v1'
S = P / 'native_SAGE_endpoint_value_transport_source_20261011_v1'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'

def item(path):
    path = Path(path).resolve()
    assert path.is_relative_to(R)
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def write(path, obj):
    assert not path.exists()
    path.write_text(json.dumps(obj, indent=2) + '\n')

def main():
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == [UUID]
    for row in json.loads((S / 'SEAL.json').read_text())['files']:
        assert item(S / row['path'])['sha256'] == row['sha256']
    cfg = json.loads((S / 'CONFIG_TEMPLATE.json').read_text())
    old = json.loads((P / 'common_wrapper_SAGE_root_20261010_v1/CONFIG.json').read_text())
    for key in ('native_repo', 'common_routes', 'factors', 'train_npz', 'valid_npz', 'device'):
        cfg[key] = old[key]
    paths = dict(training_family=S/'run_family.py', value_installer=S/'value_installer.py',
                 value_helper=S/'value_transport.py', contracts=S/'contracts.py',
                 common_routes=cfg['common_routes'], factors=cfg['factors'],
                 models=R/'models.py', run_base=R/'run_base.py', run_common=R/'run_common.py')
    pyg = R / '.venv/lib/python3.11/site-packages/torch_geometric/nn/conv'
    paths.update(sage_conv=pyg/'sage_conv.py', message_passing=pyg/'message_passing.py')
    cfg['source_receipts'] = {k:item(v) for k,v in paths.items()}
    cfg['safe_roles'] = {k:item(cfg[k+'_npz']) for k in ('train','valid')}
    fingerprint = dict(hostname=socket.gethostname(), gpu_uuid=UUID,
        driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),
        python_sources={k:item(importlib.import_module(k).__file__) for k in ('torch','numpy','torch_geometric')})
    write(H/'RUNTIME_FINGERPRINT_V1.json',fingerprint)
    # Preserve the prepared finite bound for qualification; measured timing
    # determines the separately admitted full family, without predictive access.
    canonical = hashlib.sha256(json.dumps({k:v for k,v in cfg.items() if k!='root_admission'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    admission = dict(status='HOST_GPU_SOURCE_DATA_ADMITTED',action='full_TRAIN_qualification',
        hostname=socket.gethostname(),gpu_uuid=UUID,runtime_fingerprint_sha256=item(H/'RUNTIME_FINGERPRINT_V1.json')['sha256'],
        source_sha256={str(Path(v['path']).resolve()):v['sha256'] for v in cfg['source_receipts'].values()},
        safe_role_sha256={k:v['sha256'] for k,v in cfg['safe_roles'].items()},science_config_sha256=canonical,
        UTC=datetime.now(timezone.utc).isoformat(),scientific_fits=0,VALID_scoring=False,TEST_access=False)
    write(H/'QUALIFICATION_ADMISSION_V1.json',admission)
    cfg['root_admission']=item(H/'QUALIFICATION_ADMISSION_V1.json')
    write(H/'QUALIFICATION_CONFIG_V1.json',cfg)
    result=dict(qualification_config=item(H/'QUALIFICATION_CONFIG_V1.json'),runtime_fingerprint=fingerprint,
                source_seal=item(S/'SEAL.json'),safe_roles=cfg['safe_roles'],scientific_fits=0,
                resource_observation=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines())
    write(H/'QUALIFICATION_BINDING_V1.json',result)
    print(json.dumps(result))

if __name__ == '__main__':main()
