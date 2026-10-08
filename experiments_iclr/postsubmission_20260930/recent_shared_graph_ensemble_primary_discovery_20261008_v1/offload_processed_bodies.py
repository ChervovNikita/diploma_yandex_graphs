"""Move seven processed literature bodies to verified project custody.

The raw source history remains on the authorized allocation; compact inspected
passages and conclusions stay local. This does not run scientific compute.
"""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import zlib

HERE = Path(__file__).resolve().parent
NAMES = ('query_0.json', 'query_1.json', 'query_2.json',
         '2609_02638v1.html', '2605_11987v1.html',
         '2609_02638v1_PARAGRAPHS.json', '2605_11987v1_PARAGRAPHS.json')
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
DEST = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/' + HERE.name
SSH = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
       '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
       '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10', LOGIN]
REMOTE = r'''
import base64,hashlib,json,pathlib,socket,subprocess,sys,zlib
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
D=pathlib.Path(sys.argv[1]); operation=sys.argv[2]
assert D.resolve()==D and D.parent==pathlib.Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
rows=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
if operation=='stage':D.mkdir(parents=True,exist_ok=True)
observed=[]
for row in rows:
 name=row['name'];assert pathlib.PurePosixPath(name).name==name
 f=D/name;assert not f.is_symlink()
 if operation=='stage':
  data=base64.b64decode(row['data']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
  if f.exists():assert f.read_bytes()==data
  else:
   with f.open('xb') as h:h.write(data)
 data=f.read_bytes();assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
 observed.append(dict(name=name,server_path=str(f),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
print(json.dumps(dict(operation=operation,route=socket.gethostname(),files=observed,scientific_compute=False)))
'''


def run(operation, rows):
    command = shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', REMOTE, DEST, operation])
    result = subprocess.run([*SSH, command], input=base64.b64encode(zlib.compress(json.dumps(rows).encode(), 9)).decode(),
                            capture_output=True, text=True, timeout=60)
    receipt = dict(UTC=datetime.now(timezone.utc).isoformat(), operation=operation,
                   exit_code=result.returncode, stderr=result.stderr,
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if result.returncode == 0:
        receipt['result'] = json.loads(result.stdout)
    else:
        receipt['stdout'] = result.stdout
    with (HERE / ('RAW_' + operation.upper() + '_RECEIPT.json')).open('x') as h:
        json.dump(receipt, h, indent=2); h.write('\n')
    result.check_returncode()
    return receipt['result']


def main():
    rows = []
    for name in NAMES:
        f = HERE / name; data = f.read_bytes()
        rows.append(dict(name=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                         data=base64.b64encode(data).decode()))
    staged = run('stage', rows)
    expected = [{k: r[k] for k in ('name', 'bytes', 'sha256')} for r in rows]
    verified = run('verify', expected)
    assert staged['files'] == verified['files']
    for row in rows:
        f = HERE / row['name']
        assert f.stat().st_size == row['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest() == row['sha256']
    custody = dict(schema='processed-literature-raw-server-custody-v1',
                   UTC=datetime.now(timezone.utc).isoformat(), files=verified['files'],
                   local_source_history_retained_on_server=True,
                   local_body_removal_after_independent_verification=True,
                   new_scientific_fits=0)
    with (HERE / 'RAW_SERVER_CUSTODY.json').open('x') as h:
        json.dump(custody, h, indent=2); h.write('\n')
    for row in rows:
        (HERE / row['name']).unlink()
    print(json.dumps(dict(files=len(rows), bytes=sum(r['bytes'] for r in rows),
                         server_verified=True, raw_local_bodies_removed=True)))


if __name__ == '__main__':
    main()
