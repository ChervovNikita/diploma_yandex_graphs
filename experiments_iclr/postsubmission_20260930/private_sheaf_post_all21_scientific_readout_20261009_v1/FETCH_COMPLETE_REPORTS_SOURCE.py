from pathlib import Path
import base64
import hashlib
import json
import os
import socket
import subprocess

R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P = R / 'experiments_iclr/postsubmission_20260930'
A = P / 'private_sheaf_post_all21_scientific_readout_20261009_v1'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert socket.gethostname() == 'anogena-2-0'
assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == [GPU]
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == '24c315a7-3c08-471f-b550-b9a3e1faf75d'
os.chdir(R)
receipt = json.loads((A / 'EXECUTION_RECEIPT.json').read_text())
assert receipt['complete'] is True and receipt['exit_code'] == 0 and len(receipt['complete_outputs']) == 3
files = []
for row in receipt['complete_outputs']:
    path = Path(row['path'])
    assert path.parent == A / 'complete_analysis' and path.name in {'ANALYSIS.json', 'EVERY_MEMBER_QUALITY.csv', 'REPORT.md'}
    assert path.stat().st_size == row['bytes'] and row['bytes'] < 8 * 1024**2
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == row['sha256']
    files.append(dict(name='complete_analysis/' + path.name, bytes=len(payload), sha256=row['sha256'], data=base64.b64encode(payload).decode()))
metadata = [('EXECUTION_RECEIPT.json', A / 'EXECUTION_RECEIPT.json'), ('RELEASE.json', A / 'RELEASE.json'), ('ROOT_CUSTODY.json', A / 'ROOT_CUSTODY.json')]
for family, folder in [('original18', 'geometry_only_core_scientific18_activation_root_20261009_v1'), ('centered3', 'geometry_only_core_centered3_activation_root_20261009_v1')]:
    metadata += [('closure_metadata/' + family + '_' + name, P / folder / name) for name in ('LAUNCH.json', 'TERMINAL.json', 'WORKER_OWNER.json')]
metadata += [('closure_metadata/ALL21_CLOSURE.json', P / 'geometry_only_core_centered3_execution_root_20261009_v1/ALL21_CLOSURE.json')]
for name, path in metadata:
    assert path.is_relative_to(P) and path.stat().st_size < 128 * 1024
    payload = path.read_bytes()
    files.append(dict(name=name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(), data=base64.b64encode(payload).decode()))
print(json.dumps(dict(files=files, complete=True, raw_arrays_checkpoints_or_logits_downloaded=False), sort_keys=True))
