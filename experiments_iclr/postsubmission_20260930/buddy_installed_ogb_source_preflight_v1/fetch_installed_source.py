"""Read only four repo-confined installed OGB sources, not any dataset member."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
CODE = '''
from pathlib import Path
import hashlib,json,subprocess
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
expected='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True)
if repo.resolve()!=repo or gpu.stdout.strip().splitlines()!=[expected]: raise RuntimeError('Authorized root/GPU required')
package=repo/'.venv/lib/python3.11/site-packages/ogb'
result=[]
for name in ('linkproppred/dataset_pyg.py','io/read_graph_pyg.py','io/read_graph_raw.py','linkproppred/master.csv'):
 path=package/name
 if not path.resolve().is_relative_to(repo) or path.is_symlink(): raise RuntimeError('Unconfined installed source')
 data=path.read_bytes();result.append(dict(path=name,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),text=data.decode()))
print(json.dumps(dict(GPU_uuid=expected,files=result,dataset_members_read=False,GPU_compute=False)))
'''


def main():
    receipt_path = HERE / 'SOURCE_RECEIPT.json'
    if receipt_path.exists():
        raise RuntimeError('Source snapshot already exists')
    ssh = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
           '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
           '-o', 'StrictHostKeyChecking=yes', LOGIN]
    command = shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', CODE])
    result = subprocess.run([*ssh, command], capture_output=True, text=True, timeout=30, check=True)
    value = json.loads(result.stdout)
    records = []
    for item in value['files']:
        data = item['text'].encode()
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise RuntimeError('Installed source transport differs')
        path = HERE / 'source' / item['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)
        records.append({k: item[k] for k in ('path', 'sha256', 'bytes')})
    receipt = dict(UTC=datetime.now(timezone.utc).isoformat(), ssh_destination=LOGIN,
                   GPU_uuid=value['GPU_uuid'], files=records, dataset_members_read=False,
                   GPU_compute=False, wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(files=len(records), bytes=sum(x['bytes'] for x in records), dataset_members_read=False)))


if __name__ == '__main__':
    main()
