"""Mirror compact closed resource metadata, never arrays or scientific scores."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REMOTE = r'''
import base64,hashlib,json,socket,subprocess
from pathlib import Path
assert socket.gethostname() == 'anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
phase=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
names=['massive_native_packet_20261012_v2/resources_v1/'+n for n in ('MANIFEST.json','RESOURCES.json','COST_PROJECTION.json')]
names += ['publication/Native_runtime_scope_and_resource_preparation_20261012_v1/PUSH_RECEIPT_SSH22_V1.json']
files=[]
for name in names:
    path=phase/name
    raw=path.read_bytes()
    assert len(raw)<2000000
    files.append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
print(json.dumps({'files':files}))
'''

def main():
    output = HERE / 'native_resource_closure_v1'
    assert not output.exists(), 'Preserve prior fetch receipts'
    output.mkdir()
    prefix = ['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
              '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15',
              '-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no',
              'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru']
    result = subprocess.run(prefix+[shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE])],
                            capture_output=True,text=True,timeout=60)
    (output/'TRANSPORT.json').write_text(json.dumps({'exit_code':result.returncode,'stderr':result.stderr},indent=2)+'\n')
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    receipts=[]
    for item in document['files']:
        raw=base64.b64decode(item.pop('data'))
        assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
        path=Path(item['path'])
        assert not path.is_absolute() and '..' not in path.parts
        target=output/path
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(raw)
        receipts.append(item)
    (output/'FILE_RECEIPTS.json').write_text(json.dumps(receipts,indent=2)+'\n')
    print(json.dumps({'files':len(receipts),'metadata_bytes':sum(x['bytes'] for x in receipts),'bulk_arrays_fetched':False}))

if __name__=='__main__':
    main()
