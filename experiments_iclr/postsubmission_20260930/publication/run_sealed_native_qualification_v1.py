"""Deploy one exact text-only source packet and run its native qualification.

This helper uses only the authorized one-GPU route and writes a fresh packet/run
inside that repository. Original sources and ongoing training are untouched.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

PHASE = Path(__file__).resolve().parents[1]
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SSH = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
       '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
       '-o', 'StrictHostKeyChecking=yes', LOGIN]
REMOTE = r'''
import base64,hashlib,json,os,pathlib,subprocess,sys
repo=pathlib.Path(sys.argv[1]);uuid=sys.argv[2];request=json.load(sys.stdin)
assert repo.resolve()==repo
actual=subprocess.run(['git','rev-parse','--show-toplevel'],cwd=repo,capture_output=True,text=True,check=True).stdout.strip()
assert actual==str(repo)
inventory=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()
assert inventory==[uuid], 'Intended one-GPU route must match before deployment/execution'
phase=repo/'experiments_iclr/postsubmission_20260930'
target=phase/request['packet']
assert target.resolve().is_relative_to(phase) and not target.exists()
prepared=[]
for row in request['files']:
 rel=pathlib.PurePosixPath(row['path'])
 assert not rel.is_absolute() and '..' not in rel.parts
 data=base64.b64decode(row['data'])
 assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
 prepared.append((target.joinpath(*rel.parts),data))
target.mkdir()
for path,data in prepared:
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_bytes(data)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==next(r['sha256'] for r in request['files'] if r['path']==str(path.relative_to(target)))
manifest=(target/'MANIFEST.json').read_bytes()
assert hashlib.sha256(manifest).hexdigest()==request['manifest_sha256']
assert json.loads((target/'SEAL.json').read_text())['manifest_sha256']==request['manifest_sha256']
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES=uuid,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
argv=[str(repo/'.venv/bin/python'),'-B',str(target/'prototype/qualify_squirrel17.py'),'--run-name',request['run_name']]
if request['preflight_only']:argv.append('--preflight-only')
result=subprocess.run(argv,cwd=repo,env=env,capture_output=True,text=True,timeout=660)
qualification=target/'runs'/request['run_name']/'QUALIFICATION.json'
observed=json.loads(qualification.read_text()) if qualification.exists() else None
receipt=dict(ssh_destination=request['login'],route_uuid=uuid,packet=request['packet'],
             manifest_sha256=request['manifest_sha256'],argv=argv,exit_code=result.returncode,
             stdout=result.stdout,stderr=result.stderr,qualification=observed,
             original_scores_changed=False,other_jobs_modified=False)
print(json.dumps(receipt))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--receipt-dir', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    for value in (args.packet, args.run_name, args.receipt_dir):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
            raise ValueError('Simple new project identity required')
    packet = PHASE/args.packet
    receipt_dir = PHASE/args.receipt_dir
    receipt_dir.mkdir(exist_ok=False)
    manifest_bytes = (packet/'MANIFEST.json').read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    assert json.loads((packet/'SEAL.json').read_text())['manifest_sha256'] == manifest_hash
    files = []
    for path in sorted(packet.rglob('*')):
        if path.is_dir():
            continue
        assert path.is_file() and not path.is_symlink()
        assert path.suffix in ('.md', '.json', '.py', '.txt') and path.stat().st_size < 2_000_000
        data = path.read_bytes()
        files.append(dict(path=str(path.relative_to(packet)),bytes=len(data),
                          sha256=hashlib.sha256(data).hexdigest(),data=base64.b64encode(data).decode()))
    payload = dict(packet=args.packet,run_name=args.run_name,files=files,login=LOGIN,
                   manifest_sha256=manifest_hash,preflight_only=args.preflight_only)
    command = shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,REPO,UUID])
    (receipt_dir/'REMOTE_CODE.txt').write_text(REMOTE)
    summary = dict(packet=args.packet,manifest_sha256=manifest_hash,
                   files=[{k:r[k] for k in ('path','bytes','sha256')} for r in files])
    (receipt_dir/'DEPLOYMENT_INVENTORY.json').write_text(json.dumps(summary,indent=2)+'\n')
    result = subprocess.run([*SSH,command],input=json.dumps(payload),capture_output=True,text=True,timeout=700)
    receipt = dict(UTC=datetime.now(timezone.utc).isoformat(),ssh_destination=LOGIN,
                   helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)
    if result.returncode == 0:
        receipt['result'] = json.loads(result.stdout)
        q = receipt['result']['qualification']
        if q is not None:
            (receipt_dir/'QUALIFICATION.json').write_text(json.dumps(q,indent=2)+'\n')
        output = dict(transport_exit_code=0,native_exit_code=receipt['result']['exit_code'],
                      status=q.get('status') if q else None,
                      resource_preflight=q.get('resource_preflight') if q else None,
                      error=q.get('error') if q else None,
                      native_stderr=receipt['result']['stderr'])
    else:
        output = dict(transport_exit_code=result.returncode,stderr=result.stderr)
    (receipt_dir/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(output))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
