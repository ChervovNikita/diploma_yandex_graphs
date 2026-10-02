"""Deploy the reviewed Squirrel source and start its single twelve-fit study.

Only the explicitly authorized one-GPU repository is reached. Existing files
must match their recorded hashes and are never overwritten. Training is detached
from the SSH transport so a connection interruption does not stop it.
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
assert subprocess.run(['git','rev-parse','--show-toplevel'],cwd=repo,capture_output=True,text=True,check=True).stdout.strip()==str(repo)
inventory=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()
assert inventory==[uuid], 'Wrong allocation: no deployment or training permitted'
phase=repo/'experiments_iclr/postsubmission_20260930'
target=phase/request['packet']
assert not target.exists(), 'Fresh reviewed preparation required'
prepared=[]
for row in request['files']:
 rel=pathlib.PurePosixPath(row['path'])
 assert not rel.is_absolute() and '..' not in rel.parts
 path=phase.joinpath(*rel.parts)
 assert path.resolve().is_relative_to(phase)
 data=base64.b64decode(row['data'])
 assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
 if path.exists():
  assert path.is_file() and not path.is_symlink() and path.read_bytes()==data
 prepared.append((path,data))
for path,data in prepared:
 if not path.exists():
  path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as handle:handle.write(data)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==hashlib.sha256(data).hexdigest()
manifest=(target/'MANIFEST.json').read_bytes()
assert hashlib.sha256(manifest).hexdigest()==request['manifest_sha256']
assert json.loads((target/'SEAL.json').read_text())['manifest_sha256']==request['manifest_sha256']
for row in json.loads(manifest)['payload']:
 path=target/row['path']
 assert path.resolve().is_relative_to(target) and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'] and path.stat().st_size==row['bytes']
release=phase/request['release']
admission=json.loads(release.read_text())
assert admission==dict(execution_authorized=True,run_name=request['run_name'],prepared_manifest_sha256=request['manifest_sha256'],mode_donor_freeze_sha256=request['mode_freeze_sha256'])
launch=target/('launch_'+request['run_name']);launch.mkdir(exist_ok=False)
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES=uuid,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
argv=[str(repo/'.venv/bin/python'),'-B',str(target/'prototype/continue_squirrel.py'),'--run-name',request['run_name'],'--admission',str(release)]
with (launch/'execution.log').open('xb') as log:
 process=subprocess.Popen(argv,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(UTC=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),PID=process.pid,route_uuid=uuid,actual_git_root=str(repo),argv=argv,prepared_manifest_sha256=request['manifest_sha256'],mode_donor_freeze_sha256=request['mode_freeze_sha256'],release_sha256=hashlib.sha256(release.read_bytes()).hexdigest(),process_detached_from_transport=True,other_jobs_modified=False,original_scores_changed=False)
with (launch/'START.json').open('x') as handle:json.dump(receipt,handle,indent=2);handle.write('\n')
print(json.dumps(receipt))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--receipt-dir', required=True)
    args = parser.parse_args()
    for value in (args.packet, args.run_name, args.receipt_dir):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
            raise ValueError('Simple fresh project identity required')
    packet = PHASE/args.packet
    release = args.admission.resolve()
    assert release.is_relative_to(PHASE) and release.suffix == '.json'
    manifest = (packet/'MANIFEST.json').read_bytes()
    digest = hashlib.sha256(manifest).hexdigest()
    assert json.loads((packet/'SEAL.json').read_text())['manifest_sha256'] == digest
    admission = json.loads(release.read_text())
    assert admission['execution_authorized'] is True and admission['run_name'] == args.run_name
    assert admission['prepared_manifest_sha256'] == digest
    freeze = PHASE/'graph_paired_staged_root_admission_v1/MODE_DONOR_FREEZE.json'
    assert hashlib.sha256(freeze.read_bytes()).hexdigest() == admission['mode_donor_freeze_sha256']
    paths = [p for p in packet.rglob('*') if p.is_file()]
    paths += [freeze, release]
    files = []
    for path in sorted(set(paths)):
        assert path.is_relative_to(PHASE) and not path.is_symlink()
        assert path.suffix in ('.md', '.json', '.py', '.txt') and path.stat().st_size < 2_000_000
        data = path.read_bytes()
        files.append(dict(path=str(path.relative_to(PHASE)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),data=base64.b64encode(data).decode()))
    out = PHASE/args.receipt_dir
    out.mkdir(exist_ok=False)
    request = dict(packet=args.packet,run_name=args.run_name,manifest_sha256=digest,
                   mode_freeze_sha256=admission['mode_donor_freeze_sha256'],
                   release=str(release.relative_to(PHASE)),files=files)
    (out/'REMOTE_CODE.txt').write_text(REMOTE)
    (out/'DEPLOYMENT_INVENTORY.json').write_text(json.dumps({**{k:v for k,v in request.items() if k!='files'},'files':[{k:r[k] for k in ('path','bytes','sha256')} for r in files]},indent=2)+'\n')
    command = shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,REPO,UUID])
    result = subprocess.run([*SSH,command],input=json.dumps(request),capture_output=True,text=True,timeout=60)
    receipt = dict(UTC=datetime.now(timezone.utc).isoformat(),ssh_destination=LOGIN,
                   helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)
    if result.returncode == 0:
        receipt['start'] = json.loads(result.stdout)
    (out/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
