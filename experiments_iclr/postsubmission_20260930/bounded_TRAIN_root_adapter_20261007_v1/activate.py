"""Stage an explicit TRAIN activation inventory and detach its bound adapter once."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import zlib

PHASE = Path(__file__).resolve().parents[1]
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
REMOTE = r'''
from pathlib import Path
import base64,hashlib,json,os,socket,subprocess,sys,time,zlib
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd()==repo and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for row in payload['files']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 target=phase/rel;assert target.resolve().is_relative_to(phase)
 data=base64.b64decode(row['data']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
 if target.exists():assert target.is_file() and target.read_bytes()==data
 else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
config=phase/payload['config_relative'];assert sha(config)==payload['config_sha256']
cfg=json.loads(config.read_text())
for row in cfg['bindings']:assert sha(phase/row['path'])==row['sha256']
root=phase/cfg['execution_root_relative'];root.mkdir(exist_ok=True)
assert not (root/'owner').exists() and not (root/'DETACHED_LAUNCH.json').exists()
for entry in cfg['entries']:
 output=Path(entry['output_directory']);assert output.resolve().is_relative_to(root.resolve()) and not output.exists()
 output.parent.mkdir(parents=True,exist_ok=True)
argv=['/usr/bin/python3','-I','-S','-B',str(phase/'bounded_TRAIN_root_adapter_20261007_v1/execute.py'),'--config',str(config),'--config-sha256',payload['config_sha256']]
with (root/'owner.stdout.log').open('xb') as out,(root/'owner.stderr.log').open('xb') as err:
 p=subprocess.Popen(argv,cwd=repo,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
time.sleep(.5)
receipt={'PID':p.pid,'argv':argv,'exit_code':p.poll(),'config_sha256':payload['config_sha256'],'TRAIN_only':True,'retry':False}
if p.poll() is None:
 stat=(Path('/proc')/str(p.pid)/'stat').read_text();fields=stat[stat.rfind(')')+2:].split();receipt.update(start_ticks=int(fields[19]),pgid=int(fields[2]),sid=int(fields[3]))
with (root/'DETACHED_LAUNCH.json').open('x') as handle:json.dump(receipt,handle,indent=2);handle.write('\n')
print(json.dumps(receipt))
'''

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    assert args.inventory.resolve().is_relative_to(PHASE)
    assert args.receipt.resolve().is_relative_to(PHASE) and not args.receipt.exists()
    inventory = json.loads(args.inventory.read_text())
    rows = []
    for rel in inventory['files']:
        source = (PHASE / rel).resolve(strict=True)
        assert source.is_relative_to(PHASE) and source.is_file()
        data = source.read_bytes()
        assert len(data) < 2_000_000
        rows.append({'path': rel, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'data': base64.b64encode(data).decode()})
    payload = dict(inventory, files=rows)
    encoded = base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
    command = 'cd ' + shlex.quote(REPO) + ' && /usr/bin/python3 -I -S -B -c ' + shlex.quote('exec(' + repr(REMOTE) + ')')
    ssh = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt', '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=yes', '-o', 'UpdateHostKeys=no', 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru', command]
    result = subprocess.run(ssh, input=encoded, capture_output=True, text=True, timeout=60)
    record = {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr, 'inventory_sha256': hashlib.sha256(args.inventory.read_bytes()).hexdigest(), 'command_sha256': hashlib.sha256(command.encode()).hexdigest(), 'scientific_retry': False}
    args.receipt.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr if result.returncode else ''}))
    raise SystemExit(result.returncode)

if __name__ == '__main__':
    main()
