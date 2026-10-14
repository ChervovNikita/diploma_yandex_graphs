"""Bounded source transfer and commands on the literal authorized allocation."""
import hashlib
import io
import json
import shlex
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE_ROOT = "/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs"
REMOTE_DESIGN = REMOTE_ROOT + "/experiments_iclr/postsubmission_20260930/reference_wrapper_redesign_20261013_v1"
LOGIN = "anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru"
SSH = ['ssh','-tt','-e','none','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
       '-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no',LOGIN]
GUARD = '''import socket,subprocess
assert socket.gethostname()=="anogena-2-0"
assert subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],text=True).splitlines()==["GPU-44039938-fd82-41d2-fefd-de71514e2fac"]
'''


def call(code, timeout=60):
    result = subprocess.run(SSH + ['python3 -B -c ' + shlex.quote(GUARD + code)],
                            capture_output=True,text=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout.strip()


def upload():
    files = [p for p in sorted(HERE.rglob('*')) if p.is_file()
             and not ({'results','transport','__pycache__','working_data'} & set(p.relative_to(HERE).parts))
             and p.suffix in ('.py','.md','.json','.sh','.txt')]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as tar:
        for path in files:
            tar.add(path,arcname=str(path.relative_to(HERE)),recursive=False)
    packet = buffer.getvalue()
    code = GUARD + '''import io,sys,tarfile,hashlib,json
from pathlib import Path
root=Path(ROOT_LITERAL)
root.mkdir(exist_ok=True)
print("READY",flush=True)
data=sys.stdin.buffer.read(SIZE_LITERAL)
assert len(data)==SIZE_LITERAL and hashlib.sha256(data).hexdigest()==HASH_LITERAL
with tarfile.open(fileobj=io.BytesIO(data),mode="r:gz") as archive:
 members=archive.getmembers()
 for item in members:
  target=root/item.name
  assert item.isfile() and target.resolve().is_relative_to(root) and not target.is_symlink()
  target.parent.mkdir(parents=True,exist_ok=True)
  temporary=target.with_name(target.name+'.upload-part')
  assert not temporary.exists()
  temporary.write_bytes(archive.extractfile(item).read())
  temporary.replace(target)
print(json.dumps({"source_files":len(members),"packet_bytes":len(data),"jobs_started":0}),flush=True)
'''
    code = code.replace('ROOT_LITERAL',repr(REMOTE_DESIGN+'/smoke_20261014')).replace('SIZE_LITERAL',str(len(packet))).replace('HASH_LITERAL',repr(hashlib.sha256(packet).hexdigest()))
    proc = subprocess.Popen(SSH + ['stty raw -echo; python3 -B -c '+shlex.quote(code)],
                            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    first = proc.stdout.readline()
    if first.strip()!=b'READY':
        proc.terminate()
        out,err=proc.communicate(timeout=15)
        raise RuntimeError(first.decode(errors='replace')+out.decode(errors='replace')+err.decode(errors='replace'))
    out,err=proc.communicate(input=packet,timeout=60)
    if proc.returncode:
        raise RuntimeError(out.decode(errors='replace')+err.decode(errors='replace'))
    return json.loads(out.decode().strip())


if __name__ == '__main__':
    print(json.dumps(upload()))
