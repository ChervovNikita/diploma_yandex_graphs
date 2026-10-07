"""Stage exact source and optionally run authorized synthetic CPU checks.

Never launches a scientific fit; verifies hostname/UUID before repository I/O.
Remote source preparation is independent of all active queue sources.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parent
REMOTE_REPO='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
REMOTE_PHASE=REMOTE_REPO+'/experiments_iclr/postsubmission_20260930'
REMOTE_DIR=REMOTE_PHASE+'/'+ROOT.name
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-p','2222','-i',
     '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
     'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru']


def call(code,payload=None):
    result=subprocess.run(SSH+['/usr/bin/python3 -c '+shlex.quote(code)],input=payload,capture_output=True)
    if result.returncode:raise RuntimeError(result.stdout.decode()+result.stderr.decode())
    return json.loads(result.stdout)


GUARD="""import socket,subprocess,pathlib,hashlib,json,os,sys
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
"""


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cpu-check',action='store_true')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w') as archive:
        for name in ['MANIFEST.json','STATIC_CHECK.json']+[r['path'] for r in manifest['files']]:
            archive.add(ROOT/name,arcname=name,recursive=False)
    code=GUARD+"""import io,tarfile
root=pathlib.Path(sys.argv[1]);phase=root.parent
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:') as archive:
 members=archive.getmembers()
 assert all(x.isfile() and not pathlib.PurePosixPath(x.name).is_absolute() and '..' not in pathlib.PurePosixPath(x.name).parts for x in members)
 data={x.name:archive.extractfile(x).read() for x in members}
 manifest=json.loads(data['MANIFEST.json'])
 assert set(data)=={'MANIFEST.json','STATIC_CHECK.json'}|{r['path'] for r in manifest['files']}
 for row in manifest['files']:
  assert hashlib.sha256(data[row['path']]).hexdigest()==row['sha256'] and len(data[row['path']])==row['bytes']
 if root.exists():
  assert all((root/name).read_bytes()==value for name,value in data.items()),'Existing stage changed; no overwrite'
 else:
  root.mkdir()
  for name,value in data.items():
   path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
 dependencies=json.loads(data['DEPENDENCIES.json'])
 rows=[dependencies['polynormer'],dependencies['ncn']['model'],dependencies['ncn']['utils']]
 for row in rows:
  assert hashlib.sha256((phase/row['path']).read_bytes()).hexdigest()==row['sha256'],'External native dependency mismatch'
 print(json.dumps(dict(staged=True,hostname=socket.gethostname(),gpu_uuid='GPU-44039938-fd82-41d2-fefd-de71514e2fac',remote_directory=str(root),source_manifest_sha256=hashlib.sha256(data['MANIFEST.json']).hexdigest(),files=len(manifest['files']),external_dependencies_matched=len(rows),scientific_fits=0)))
"""
    # Pass only this freshly created source-directory path after quoted code.
    result=subprocess.run(SSH+['/usr/bin/python3 -c '+shlex.quote(code)+' '+shlex.quote(REMOTE_DIR)],
        input=buffer.getvalue(),capture_output=True)
    if result.returncode:raise RuntimeError(result.stdout.decode()+result.stderr.decode())
    receipt=json.loads(result.stdout)
    (ROOT/'STAGE_RECEIPT.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps(receipt))
    if args.cpu_check:
        check=GUARD+"""repo=pathlib.Path(sys.argv[1]);root=pathlib.Path(sys.argv[2]);phase=root.parent
env=dict(os.environ,PYTHONPATH=str(phase/'native_ncn_dependency_overlay_20261005_v1')+':'+str(repo/'.venv/lib/python3.11/site-packages'),CUDA_VISIBLE_DEVICES='')
result=subprocess.run([str(phase/'native_ncn_runtime_20261005_v1/.venv/bin/python'),'-B',str(root/'check_cpu.py'),'--output',str(root/'CPU_CHECK.json')],cwd=str(repo),env=env,text=True,capture_output=True,timeout=180)
if result.returncode:raise RuntimeError(result.stdout+result.stderr)
print((root/'CPU_CHECK.json').read_text())
"""
        result=subprocess.run(SSH+['/usr/bin/python3 -c '+shlex.quote(check)+' '+shlex.quote(REMOTE_REPO)+' '+shlex.quote(REMOTE_DIR)],capture_output=True,timeout=210)
        if result.returncode:
            failure={'kind':'synthetic_CPU_check_failure','stdout':result.stdout.decode(),'stderr':result.stderr.decode(),'source_manifest_sha256':receipt['source_manifest_sha256']}
            (ROOT/'CPU_FAILURE.json').write_text(json.dumps(failure,indent=2)+'\n')
            raise RuntimeError(failure['stdout']+failure['stderr'])
        checked=json.loads(result.stdout)
        (ROOT/'CPU_CHECK.json').write_text(json.dumps(checked,indent=2,sort_keys=True)+'\n')
        print(json.dumps({'cpu_check_passed':checked['passed'],'rows':len(checked['rows']),'seconds':checked['seconds']}))


if __name__=='__main__':main()
