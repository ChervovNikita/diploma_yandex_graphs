"""Bind the single FiLM runtime check to existing WikiCS roles and runtime."""
import hashlib
import importlib
import json
from pathlib import Path
import socket
import subprocess
from datetime import datetime, timezone

R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P = R / 'experiments_iclr/postsubmission_20260930'
H = P / 'native_PyG_FiLM_WikiCS_comparator_execution_root_20261011_v1'
S = P / 'native_PyG_FiLM_WikiCS_comparator_source_20261011_v1'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def item(path):
    path = Path(path).resolve()
    assert path.is_relative_to(R) and path.is_file()
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == [UUID]
    for row in json.loads((S / 'MANIFEST.json').read_text())['files']:
        assert item(S / row['path'])['sha256'] == row['sha256']
    cfg = json.loads((S / 'CONFIG_TEMPLATE.json').read_text())
    old = json.loads((P / 'native_SAGE_endpoint_value_execution_root_20261011_v1/QUALIFICATION_CONFIG_V1.json').read_text())
    for key in ('train_npz', 'valid_npz', 'device'):
        cfg[key] = old[key]
    cfg['native_repo'] = str(S / 'deps/native')
    cfg['common_routes'] = str(S / 'deps/common_routes.py')
    cfg['factors'] = str(S / 'deps/factors.py')
    cfg['family_limits'] = dict(wall_seconds=240, cuda_bytes=32 * 2**30, rss_bytes=16 * 2**30)
    pyg = R / '.venv/lib/python3.11/site-packages/torch_geometric/nn/conv'
    paths = dict(training_family=S/'run_family.py', film_adapter=S/'film_adapter.py',
                 contracts=S/'contracts.py', analysis=S/'analysis.py', math_utils=S/'math_utils.py',
                 readiness=S/'readiness.py', common_routes=S/'deps/common_routes.py',
                 factors=S/'deps/factors.py', models=S/'deps/native/models.py',
                 run_common=S/'deps/native/run_common.py', film_conv=pyg/'film_conv.py',
                 message_passing=pyg/'message_passing.py')
    cfg['source_receipts'] = {key: item(value) for key, value in paths.items()}
    cfg['safe_roles'] = {key: item(cfg[key+'_npz']) for key in ('train', 'valid')}
    fingerprint = dict(hostname=socket.gethostname(), gpu_uuid=UUID,
        driver=subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip(),
        python_sources={key: item(importlib.import_module(key).__file__) for key in ('torch', 'numpy', 'torch_geometric')})
    write(H/'RUNTIME_FINGERPRINT_V1.json', fingerprint)
    canonical = hashlib.sha256(json.dumps({key: value for key, value in cfg.items() if key != 'root_admission'}, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    admission = dict(status='HOST_GPU_SOURCE_DATA_ADMITTED', action='runtime_readiness_check',
        study_kind=cfg['study_kind'], hostname=socket.gethostname(), gpu_uuid=UUID,
        runtime_fingerprint_sha256=item(H/'RUNTIME_FINGERPRINT_V1.json')['sha256'],
        source_sha256={str(Path(value['path']).resolve()): value['sha256'] for value in cfg['source_receipts'].values()},
        safe_role_sha256={key: value['sha256'] for key, value in cfg['safe_roles'].items()},
        science_config_sha256=canonical, UTC=datetime.now(timezone.utc).isoformat(),
        one_disposable_update=True, scientific_fit_or_selection=False,
        VALID_scoring=False, TEST_access=False, science_launch_admitted=False)
    write(H/'READINESS_ADMISSION_V1.json', admission)
    cfg['root_admission'] = item(H/'READINESS_ADMISSION_V1.json')
    write(H/'READINESS_CONFIG_V1.json', cfg)
    value = dict(config=item(H/'READINESS_CONFIG_V1.json'), admission=item(H/'READINESS_ADMISSION_V1.json'),
        runtime=item(H/'RUNTIME_FINGERPRINT_V1.json'), scientific_fit_or_selection=False,
        resources=subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free,utilization.gpu', '--format=csv,noheader'], text=True).splitlines())
    write(H/'READINESS_BINDING_V1.json', value)
    print(json.dumps(value))


if __name__ == '__main__':
    main()
