"""Guarded bounded transfer/commands for this expansion; no scientific alias."""
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
sys.path.insert(0, str(PHASE / 'reference_wrapper_redesign_20261013_v1/smoke_20261014'))
from remote_ops import SSH, GUARD, REMOTE_ROOT, call
REMOTE_PHASE = REMOTE_ROOT + '/experiments_iclr/postsubmission_20260930'
REMOTE_EXPANSION = REMOTE_PHASE + '/' + HERE.name


def upload():
    files = [(PHASE / 'AGENTS.md', 'AGENTS.md')]
    for path in sorted(HERE.rglob('*')):
        relative = path.relative_to(HERE)
        if path.is_file() and not ({'results', '__pycache__', '.cache'} & set(relative.parts)) \
                and path.suffix in {'.py', '.md', '.json', '.txt', '.xml'}:
            files.append((path, HERE.name + '/' + str(relative)))
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        for path, relative in files:
            archive.add(path, arcname=relative, recursive=False)
    data = stream.getvalue()
    code = GUARD + '''import io,sys,tarfile,hashlib,json,os
from pathlib import Path
root=Path(ROOT_LITERAL)
print('READY',flush=True)
data=sys.stdin.buffer.read(SIZE_LITERAL)
assert len(data)==SIZE_LITERAL and hashlib.sha256(data).hexdigest()==HASH_LITERAL
changed=[]
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
 members=archive.getmembers()
 for item in members:
  target=root/item.name
  assert item.isfile() and target.resolve().is_relative_to(root) and not target.is_symlink()
  content=archive.extractfile(item).read()
  if target.exists() and target.read_bytes()==content:continue
  if target.exists() and target.name!='AGENTS.md':
   raise ValueError('Refuse source replacement; freeze a repair identity: '+str(target))
  target.parent.mkdir(parents=True,exist_ok=True)
  temporary=target.with_name(target.name+'.upload-'+str(os.getpid()))
  with temporary.open('xb') as out:out.write(content)
  temporary.replace(target);changed.append(item.name)
print(json.dumps({'source_files':len(members),'packet_bytes':len(data),'changed':changed,'jobs_started':0}),flush=True)
'''
    code = code.replace('ROOT_LITERAL', repr(REMOTE_PHASE)).replace('SIZE_LITERAL', str(len(data))).replace('HASH_LITERAL', repr(hashlib.sha256(data).hexdigest()))
    child = subprocess.Popen(SSH + ['stty raw -echo; python3 -B -c '+shlex.quote(code)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    first = child.stdout.readline()
    if first.strip() != b'READY':
        child.terminate();out,err=child.communicate(timeout=15)
        raise RuntimeError((first+out+err).decode(errors='replace'))
    out,err=child.communicate(input=data,timeout=90)
    if child.returncode:raise RuntimeError((out+err).decode(errors='replace'))
    return json.loads(out.decode().strip())

if __name__=='__main__':
    result=upload()
    (HERE/'TRANSFER_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('source_files','packet_bytes','jobs_started')}))
