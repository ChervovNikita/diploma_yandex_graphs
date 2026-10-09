"""Read-only exact compact-report harvest through the reviewed normal77 route."""
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
WRAPPER = PHASE / 'gpu77_connection_recovery_v1/run_gpu77_v3.py'
COMMAND_ID = 'QK36_COMPLETED_SCIENTIFIC_COMPACT_HARVEST_20261010_v1'
REPO = '/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'
REMOTE = r'''
import base64,gzip,hashlib,json,os,socket,stat,subprocess
from pathlib import Path
repo=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
phase=repo/'experiments_iclr/postsubmission_20260930'
assert socket.gethostname()=='peptide'
assert Path.cwd().resolve()==repo
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()
assert gpu==['GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998','GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced']
roots=[phase/'pre_sigmoid_qk36_original_route_assembly_activation_root_20261010_v1/server_only_analysis_execution/compact',phase/'pre_sigmoid_qk36_original_route_assembly_supervision_root_20261010_v1']
files=[];omitted=[]
for root in roots:
 assert root.is_dir() and not root.is_symlink() and root.resolve().is_relative_to(phase)
 for folder,dirs,names in os.walk(root,followlinks=False):
  for name in dirs:
   assert not (Path(folder)/name).is_symlink()
  for name in sorted(names):
   path=Path(folder)/name
   assert stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
   relative=str(path.relative_to(phase))
   if path.suffix not in ('.json','.md','.csv','.tsv','.txt','.log'):
    omitted.append(relative);continue
   data=path.read_bytes()
   data.decode('utf-8')
   files.append(dict(path=relative,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),base64=base64.b64encode(data).decode('ascii')))
payload=dict(schema='qk36-exact-compact-text-harvest-v1',hostname=socket.gethostname(),cwd=str(Path.cwd()),GPU_inventory=gpu,roots=[str(r.relative_to(phase)) for r in roots],files=files,omitted=omitted,raw_arrays_or_checkpoints_read=False,scientific_execution=False)
packed=gzip.compress(json.dumps(payload,sort_keys=True,separators=(',',':')).encode(),mtime=0)
print(json.dumps(dict(schema='gzip-json-compact-harvest-v1',gzip_base64=base64.b64encode(packed).decode(),gzip_bytes=len(packed),file_count=len(files))))
'''


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data, 'Never overwrite differing evidence'
    else:
        path.write_bytes(data)


def main():
    assert hashlib.sha256(WRAPPER.read_bytes()).hexdigest() == '035b740ceb50eefcfa3cec2aef6dbd1294c769d133e2bc4cf88fb52b088421ff'
    command = 'cd ' + shlex.quote(REPO) + ' && /usr/bin/python3 -I -S -B -c ' + shlex.quote(REMOTE)
    assert len(command.encode()) < 120000
    command_file = WRAPPER.parent / (COMMAND_ID + '_COMMAND.txt')
    write_new(command_file, command.encode())
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(['python3','-B',str(WRAPPER),'--id',COMMAND_ID,'--command-file',str(command_file)],capture_output=True,text=True)
    record = dict(schema='qk36-compact-harvest-local-transport-v1',UTC_started=started,UTC_finished=datetime.now(timezone.utc).isoformat(),command_id=COMMAND_ID,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,credential_values_read_or_recorded=False)
    write_new(HERE/'HARVEST_TRANSPORT.json',(json.dumps(record,indent=2,sort_keys=True)+'\n').encode())
    assert result.returncode == 0, 'Retained transport failure; no automatic retry'
    outer = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')][-1]
    assert outer['exit_code'] == 0
    candidates = [json.loads(line) for line in outer['stdout'].splitlines() if line.startswith('{')]
    packed = next(value for value in candidates if value.get('schema')=='gzip-json-compact-harvest-v1')
    payload = json.loads(gzip.decompress(base64.b64decode(packed['gzip_base64'])))
    rows = []
    for row in payload['files']:
        relative = Path(row['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        data = base64.b64decode(row['base64'])
        assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
        write_new(HERE/'evidence'/relative,data)
        rows.append({k:v for k,v in row.items() if k != 'base64'})
    inventory = {k:v for k,v in payload.items() if k != 'files'}
    inventory.update(files=rows,total_bytes=sum(row['bytes'] for row in rows),file_count=len(rows),transport_file='HARVEST_TRANSPORT.json')
    write_new(HERE/'HARVEST_INVENTORY.json',(json.dumps(inventory,indent=2,sort_keys=True)+'\n').encode())
    print(json.dumps(dict(exit_code=result.returncode,file_count=len(rows),total_bytes=inventory['total_bytes'],omitted=inventory['omitted'],files=[row['path'] for row in rows]),indent=2))


if __name__ == '__main__':
    main()
