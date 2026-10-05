"""Run a separately recorded root metadata request on the authorized allocation."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
REMOTE = r'''
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,socket,subprocess,sys
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
packet=json.load(sys.stdin)
def path(rel):
 p=Path(rel);assert not p.is_absolute() and '..' not in p.parts
 q=phase/p;assert q.resolve().is_relative_to(phase)
 assert not any(x.is_symlink() for x in (q,*q.parents) if x.is_relative_to(phase))
 return q
for row in packet['files']:
 p=path(row['path']);data=base64.b64decode(row['data'])
 assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
 if p.exists():assert p.read_bytes()==data
 else:
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(data)
request=packet['request']
for ref in request['input_bindings']:
 assert hashlib.sha256(path(ref['path']).read_bytes()).hexdigest()==ref['sha256']
result=subprocess.run(request['argv'],cwd=repo,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=request['child_timeout_seconds'],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
value={'UTC':datetime.now(timezone.utc).isoformat(),'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'returned_metadata':[]}
for rel in request.get('return_metadata',[]):
 p=path(rel)
 assert p.name not in ('FREEZE.json','CONFIG.json','EXTERNAL_ANCHORS.json','VALID_HISTORY.jsonl') and p.suffix=='.json' and p.stat().st_size<2_000_000
 data=p.read_bytes();value['returned_metadata'].append({'path':rel,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'utf8':data.decode()})
print(json.dumps(value))
'''

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    request_path = HERE/args.request
    request = json.loads(request_path.read_text())
    files = []
    for rel in request['stage_files']:
        path = PHASE/rel
        assert path.resolve().is_relative_to(PHASE) and path.is_file() and not path.is_symlink()
        data = path.read_bytes()
        assert len(data)<2_000_000
        files.append({'path':rel,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'data':base64.b64encode(data).decode()})
    command = ['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=20','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','cd '+shlex.quote(REPO)+' && exec /usr/bin/python3 -I -S -B -c '+shlex.quote(REMOTE)]
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(command,input=json.dumps({'files':files,'request':request}),capture_output=True,text=True,timeout=min(request['child_timeout_seconds']+25,55))
    receipt = {'start_UTC':started,'terminal_UTC':datetime.now(timezone.utc).isoformat(),'exit_code':result.returncode,'stderr':result.stderr,'request_sha256':hashlib.sha256(request_path.read_bytes()).hexdigest(),'remote_source_sha256':hashlib.sha256(REMOTE.encode()).hexdigest(),'automatic_retry':False}
    if result.returncode == 0:
        receipt['result'] = json.loads(result.stdout)
        for row in receipt['result']['returned_metadata']:
            data = row['utf8'].encode()
            assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
            path = HERE/'observation'/row['path']
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as f:f.write(data)
    else:
        receipt['stdout']=result.stdout
    with (HERE/args.receipt).open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps({'transport_exit_code':result.returncode,'command_exit_code':receipt.get('result',{}).get('exit_code'),'stdout':receipt.get('result',{}).get('stdout'),'stderr':receipt.get('result',{}).get('stderr',result.stderr),'returned_metadata_count':len(receipt.get('result',{}).get('returned_metadata',[]))}))
    raise SystemExit(result.returncode or receipt.get('result',{}).get('exit_code',0))

if __name__ == '__main__':
    main()
