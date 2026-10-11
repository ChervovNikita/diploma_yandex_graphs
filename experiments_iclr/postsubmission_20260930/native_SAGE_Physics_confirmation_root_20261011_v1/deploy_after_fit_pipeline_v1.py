"""Transfer and bind fixed small after-fit sources; no numerical stage launch."""
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
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
repo=pathlib.Path(sys.argv[1]);phase=repo/'experiments_iclr/postsubmission_20260930';H=phase/'native_SAGE_Physics_confirmation_root_20261011_v1'
for row in payload['files']:
    path=phase/row['relative'];assert path.resolve().is_relative_to(phase) and not path.is_symlink()
    data=base64.b64decode(row['payload']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert path.read_bytes()==data
    else:path.write_bytes(data)
os.chdir(repo)
env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
r=subprocess.run([str(repo/'.venv/bin/python'),'-B',str(H/'bind_after_fit_pipeline_v1.py')],cwd=repo,env=env,text=True,capture_output=True,timeout=90)
if r.returncode:print(json.dumps(dict(complete=False,stdout=r.stdout,stderr=r.stderr,exit_code=r.returncode)));sys.exit(r.returncode)
print(r.stdout)
'''


def main():
    folders = ('native_SAGE_Physics_after_fit_pipeline_source_20261011_v1',)
    paths = [v for folder in folders for v in sorted((P / folder).iterdir()) if v.is_file() and v.suffix in ('.py', '.json', '.md')]
    paths.append(H / 'bind_after_fit_pipeline_v1.py')
    files = []
    for v in paths:
        b = v.read_bytes(); assert len(b) < 200000
        if v.suffix == '.py': compile(b, str(v), 'exec')
        files.append(dict(relative=str(v.relative_to(P)), bytes=len(b), sha256=hashlib.sha256(b).hexdigest(), payload=base64.b64encode(b).decode()))
    spec = importlib.util.spec_from_file_location('after_fit_deploy_transport', P / 'publication/publish_exact_inventory_v19.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    args = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'UpdateHostKeys=no', 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
            shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', REMOTE, R, UUID, 'Physics-after-fit-binding'])]
    r = m.terminal_transport(args, input=base64.b64encode(zlib.compress(json.dumps(dict(files=files)).encode())).decode(), capture_output=True, text=True, timeout=120)
    (H / 'AFTER_FIT_PIPELINE_BIND_TRANSPORT_V1.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(), exit_code=r.returncode, stdout=r.stdout, stderr=r.stderr), indent=2) + '\n')
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-1000:]
    value = json.loads(r.stdout); assert value['complete']
    for name, field in (('AFTER_FIT_PIPELINE_CONFIG_V1.json', 'config'), ('AFTER_FIT_PIPELINE_ADMISSION_V1.json', 'admission')):
        q = H / name; assert not q.exists(); q.write_text(json.dumps(value[field], indent=2, allow_nan=False) + '\n')
    for name, field in (('AFTER_FIT_PIPELINE_CONFIG_V1.json', 'config_receipt'), ('AFTER_FIT_PIPELINE_ADMISSION_V1.json', 'admission_receipt')):
        assert hashlib.sha256((H / name).read_bytes()).hexdigest() == value[field]['sha256']
    record = {k: value[k] for k in ('complete', 'config_receipt', 'admission_receipt', 'scientific_updates', 'CAL_TEST_scoring')}
    (H / 'AFTER_FIT_PIPELINE_BIND_RECEIPT_V1.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__': main()
