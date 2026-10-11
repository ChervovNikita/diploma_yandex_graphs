"""Transfer small frozen integration sources and bind the complete cohort."""
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import zlib

H = Path(__file__).resolve().parent
P = H.parent
R = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
REMOTE = r'''
import base64,zlib,json,sys,socket,subprocess,pathlib,hashlib,os
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==[sys.argv[2]]
repo=pathlib.Path(sys.argv[1]);phase=repo/'experiments_iclr/postsubmission_20260930'
H=phase/'native_SAGE_Physics_confirmation_root_20261011_v1'
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
for row in payload['files']:
    path=phase/row['relative']; assert path.resolve().is_relative_to(phase) and not path.is_symlink()
    data=base64.b64decode(row['payload']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert path.read_bytes()==data
    else:path.write_bytes(data)
os.chdir(repo)
bound=subprocess.run([str(repo/'.venv/bin/python'),'-B',str(H/'bind_complete_cohort_v2.py')],cwd=repo,text=True,capture_output=True,timeout=300)
if bound.returncode: print(json.dumps(dict(complete=False,exit_code=bound.returncode,stdout=bound.stdout,stderr=bound.stderr)));sys.exit(bound.returncode)
print(bound.stdout)
'''


def main():
    paths = []
    for folder in ('native_SAGE_Physics_coupled_fit_integration_source_20261011_v1',):
        paths.extend(p for p in sorted((P / folder).iterdir()) if p.is_file() and p.suffix in ('.py', '.json', '.md'))
    paths.extend(H / n for n in ('ROOT_HOOK_V1.py', 'bind_complete_cohort_v2.py', 'complete_cohort_outer_owner_v1.py'))
    files = []
    for path in paths:
        data = path.read_bytes(); assert len(data) < 200000
        if path.suffix == '.py': compile(data, str(path), 'exec')
        files.append(dict(relative=str(path.relative_to(P)), bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), payload=base64.b64encode(data).decode()))
    spec = importlib.util.spec_from_file_location('guarded_transport', P / 'publication/publish_exact_inventory_v19.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    args = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'UpdateHostKeys=no', 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
            shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', REMOTE, R, UUID, 'Physics-complete-binding'])]
    r = m.terminal_transport(args, input=base64.b64encode(zlib.compress(json.dumps(dict(files=files)).encode())).decode(), capture_output=True, text=True, timeout=330)
    (H / 'COMPLETE_COHORT_BIND_TRANSPORT_V3.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(), exit_code=r.returncode, stdout=r.stdout, stderr=r.stderr), indent=2) + '\n')
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-1000:]
    value = json.loads(r.stdout); assert value['complete']
    for name, field in (('CORE_CONFIG_V1.json', 'core_config'), ('AUTHOR_CONFIG_V1.json', 'author_config'), ('FROZEN_BINDINGS_V1.json', 'admission'), ('ROOT_INTEGRATION_V1.json', 'spec')):
        data = (json.dumps(value[field], indent=2, allow_nan=False) + '\n').encode()
        path = H / name; assert not path.exists(); path.write_bytes(data)
    assert hashlib.sha256((H / 'ROOT_INTEGRATION_V1.json').read_bytes()).hexdigest() == value['spec_receipt']['sha256']
    (H / 'COMPLETE_COHORT_BIND_RECEIPT_V1.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(), complete=True, spec_receipt=value['spec_receipt'], scientific_fits=0, CAL_CONFIRM_scoring=False), indent=2) + '\n')
    print(json.dumps(dict(complete=True, spec_receipt=value['spec_receipt'], scientific_fits=0)))


if __name__ == '__main__': main()
