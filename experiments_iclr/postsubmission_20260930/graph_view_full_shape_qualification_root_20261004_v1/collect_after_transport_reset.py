"""Read saved QA receipts and exact matching children; do not retry or signal."""
from datetime import datetime, timezone
from pathlib import Path
import base64
import hashlib
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
CODE = r'''
from pathlib import Path
from datetime import datetime, timezone
import base64, hashlib, json, os, subprocess
repo = Path(REPO); root = repo/'experiments_iclr/postsubmission_20260930'/OUTPUT_NAME
assert Path.cwd() == repo
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],
    capture_output=True,text=True,check=True).stdout.splitlines() == [UUID]
pattern = str(root/'check_one.py')
found = subprocess.run(['pgrep','-f',pattern], capture_output=True, text=True)
assert found.returncode in (0,1)
owned = []
for pid in found.stdout.splitlines():
    try:
        proc = Path('/proc')/pid
        argv = [x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
        if len(argv) < 3 or argv[0] != str(repo/'.venv/bin/python') or argv[1:3] != ['-B',pattern]:
            continue
        raw = (proc/'stat').read_text(); fields = raw[raw.rfind(')')+2:].split()
        owned.append(dict(PID=int(pid), start_ticks=int(fields[19]), parent_PID=int(fields[1]),
            group=int(fields[2]), session=int(fields[3]), state=fields[0], argv=argv))
    except FileNotFoundError:
        continue
files = []
for path in sorted(root.glob('*.json')):
    if path.name == 'INPUT_BINDINGS.json': continue
    assert not path.is_symlink() and path.stat().st_size < 2000000
    raw=path.read_bytes()
    files.append(dict(name=path.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                     data=base64.b64encode(raw).decode()))
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(), exact_owned_children=owned,
    files=files, signals_sent=False, retry_or_resume=False, predictive_values_read=False)))
'''


def main():
    code = 'REPO='+repr(REPO)+'\nUUID='+repr(UUID)+'\nOUTPUT_NAME='+repr(HERE.name)+'\n'+CODE
    command = 'cd '+shlex.quote(REPO)+' && '+shlex.join(['/usr/bin/python3','-I','-S','-B','-c',code])
    ssh=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
         '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','UpdateHostKeys=no',
         '-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',LOGIN,command]
    result=subprocess.run(ssh,capture_output=True,text=True,timeout=30)
    assert result.returncode == 0,result.stderr
    value=json.loads(result.stdout); rows=value.pop('files'); value['fetched_files']=[]
    for row in rows:
        path=HERE/row['name']; assert path.parent==HERE
        raw=base64.b64decode(row['data'],validate=True)
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        if path.exists(): assert path.read_bytes()==raw
        else:
            with path.open('xb') as stream:stream.write(raw)
        value['fetched_files'].append({k:v for k,v in row.items() if k!='data'})
    stamp=datetime.now(timezone.utc).strftime('%H%M%S')
    with (HERE/('RECOVERY_'+stamp+'.json')).open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
