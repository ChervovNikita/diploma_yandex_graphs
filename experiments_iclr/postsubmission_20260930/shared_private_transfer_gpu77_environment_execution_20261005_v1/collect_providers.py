"""Report installed providers after the approved isolated runtime setup; no models/data."""
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import sysconfig
from datetime import datetime, timezone
from pathlib import Path

PREFIX = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/.gnnm_runtime/private_transfer_cp311_cu118_20261005_v1')
OUT = Path(__file__).resolve().parent
assert Path(sys.prefix) == PREFIX
assert sys.version_info[:3] == (3, 11, 14)
assert not os.environ.get('PYTHONPATH')
expected = {'torch': '2.1.2+cu118', 'numpy': '1.26.4', 'scipy': '1.14.1',
            'torch_geometric': '2.7.0', 'torch_sparse': '0.6.18+pt21cu118',
            'torch_scatter': '2.1.2+pt21cu118', 'triton': '2.1.0'}

def record(path):
    path = Path(path)
    raw = path.read_bytes()
    return {'path': str(path), 'resolved_path': str(path.resolve()), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}

providers = {}
for name, version in expected.items():
    module = importlib.import_module(name)
    assert module.__version__ == version, (name, module.__version__, version)
    path = Path(module.__file__)
    assert path.resolve().is_relative_to(PREFIX)
    native = sorted(path.parent.glob('*.so'))
    if name == 'torch':
        native += sorted((path.parent/'lib').glob('*.so*'))
    providers[name] = {'version': module.__version__, 'module': record(path),
                       'native_files': [record(p) for p in native if p.is_file()]}

distributions = []
for dist in sorted(importlib.metadata.distributions(), key=lambda d: d.metadata['Name'].lower()):
    root = Path(dist.locate_file('')).resolve()
    assert root.is_relative_to(PREFIX), str(root)
    distributions.append({'name': dist.metadata['Name'], 'version': dist.version,
                          'root': str(root), 'direct_url': dist.read_text('direct_url.json')})
conda_records = []
for p in sorted((PREFIX/'conda-meta').glob('*.json')):
    doc = json.loads(p.read_text())
    conda_records.append({'record': record(p), **{k: doc.get(k) for k in
                          ('name','version','build','build_number','channel','url','sha256','md5','subdir')}})
driver = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total',
                         '--format=csv,noheader'], capture_output=True, text=True, check=True, timeout=10)
libraries = subprocess.run(['ldd', str(PREFIX/'bin/python')], capture_output=True, text=True, check=True, timeout=10)
report = {'UTC': datetime.now(timezone.utc).isoformat(), 'status': 'PASS',
          'python': {'version': sys.version, 'executable': record(PREFIX/'bin/python'),
                     'prefix': sys.prefix, 'base_prefix': sys.base_prefix, 'platform': platform.platform(),
                     'config_args': sysconfig.get_config_var('CONFIG_ARGS'), 'ldd': libraries.stdout},
          'providers': providers, 'distributions': distributions, 'conda_records': conda_records,
          'torch_cuda': importlib.import_module('torch').version.cuda,
          'driver_inventory': driver.stdout, 'sys_path': sys.path,
          'PYTHONPATH': os.environ.get('PYTHONPATH'), 'models_imported': False, 'data_reads': False,
          'fits_executed': False, 'host_binary_identity_required': False}
assert report['torch_cuda'] == '11.8'
(OUT/'PROVIDERS.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
print(json.dumps({'status': 'PASS', 'python': platform.python_version(), 'providers': expected,
                  'provider_report_sha256': hashlib.sha256((OUT/'PROVIDERS.json').read_bytes()).hexdigest()}))
