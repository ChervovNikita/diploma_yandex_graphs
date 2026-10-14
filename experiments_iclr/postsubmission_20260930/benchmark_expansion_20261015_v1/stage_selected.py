"""Stage an explicit stable source selection without overwriting prior files."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
from remote_control import SSH, GUARD, REMOTE_EXPANSION, HERE


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths',nargs='+');args=parser.parse_args()
    files=[]
    for name in args.paths:
        path=(HERE/name).resolve()
        if not path.is_relative_to(HERE):raise ValueError('Select only expansion source')
        files.extend([path] if path.is_file() else [p for p in sorted(path.rglob('*')) if p.is_file() and p.suffix in {'.py','.md','.json','.txt'} and '__pycache__' not in p.parts])
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as archive:
        for path in files:archive.add(path,arcname=str(path.relative_to(HERE)),recursive=False)
    packet=stream.getvalue()
    code=GUARD+'''import io,sys,tarfile,hashlib,json,os
from pathlib import Path
root=Path(ROOT_LITERAL)
print('READY',flush=True)
data=sys.stdin.buffer.read(SIZE_LITERAL)
assert len(data)==SIZE_LITERAL and hashlib.sha256(data).hexdigest()==HASH_LITERAL
added=[]
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
 for member in archive.getmembers():
  target=root/member.name
  assert member.isfile() and target.resolve().is_relative_to(root) and not target.is_symlink()
  content=archive.extractfile(member).read()
  if target.exists():
   assert target.read_bytes()==content,'Existing selected source differs: '+member.name
   continue
  target.parent.mkdir(parents=True,exist_ok=True)
  temporary=target.with_name(target.name+'.upload-'+str(os.getpid()))
  with temporary.open('xb') as stream:stream.write(content)
  temporary.replace(target);added.append(member.name)
print(json.dumps({'added':added,'packet_bytes':len(data),'scientific_jobs_started':0}),flush=True)
'''
    code=code.replace('ROOT_LITERAL',repr(REMOTE_EXPANSION)).replace('SIZE_LITERAL',str(len(packet))).replace('HASH_LITERAL',repr(hashlib.sha256(packet).hexdigest()))
    child=subprocess.Popen(SSH+['stty raw -echo; python3 -B -c '+shlex.quote(code)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    first=child.stdout.readline()
    if first.strip()!=b'READY':
        child.terminate();out,err=child.communicate(timeout=15);raise RuntimeError((first+out+err).decode(errors='replace'))
    out,err=child.communicate(input=packet,timeout=90)
    if child.returncode:raise RuntimeError((out+err).decode(errors='replace'))
    print(out.decode().strip())

if __name__=='__main__':main()
