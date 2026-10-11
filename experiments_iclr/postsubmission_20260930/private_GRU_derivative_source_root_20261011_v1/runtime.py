"""One admitted derived-weight gradient fixture; standard finite execution."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def route():
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
        text=True, timeout=15).splitlines() == [UUID]


def write(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(PHASE)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def admit(config):
    assert config['stage'] == 'derivative'
    assert config['fixtures'] == 1 and config['scientific_fits'] == 0
    for field in ('output', 'owner_output', 'core_root'):
        assert Path(config[field]).resolve().is_relative_to(PHASE)
    core = Path(config['core_root'])
    manifest_bytes = (core / 'MANIFEST.json').read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == config['core_manifest_sha256']
    manifest = json.loads(manifest_bytes)
    for relative, receipt in manifest['files'].items():
        path = core / relative
        assert path.resolve().is_relative_to(core)
        data = path.read_bytes()
        assert len(data) == receipt['bytes']
        assert hashlib.sha256(data).hexdigest() == receipt['sha256']
    for relative, receipt in config['execution_source'].items():
        path = Path(__file__).parent / relative
        data = path.read_bytes()
        assert len(data) == receipt['bytes']
        assert hashlib.sha256(data).hexdigest() == receipt['sha256']
    return manifest
