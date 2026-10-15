"""Exact immutable source transport only; no qualifier execution or admission."""
from pathlib import Path
import hashlib
import importlib.util
import io
import json
import shlex
import subprocess
import tarfile
HERE=Path(__file__).resolve().parent
E=HERE.parent;PHASE=E.parent
spec=importlib.util.spec_from_file_location('_combined9_reviewed_route',E/'remote_control.py')
remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
files=[]
for name in('runtime_link_v8_combined_lifecycle_v1','runtime_link_v9_combined_lifecycle_v1'):
    folder=HERE/name;manifest=json.loads((folder/'MANIFEST.json').read_text());files.append(folder/'MANIFEST.json')
    for row in manifest['files']:
        p=folder/row['path'];data=p.read_bytes()
        assert len(data)==row['bytes']and hashlib.sha256(data).hexdigest()==row['sha256'];files.append(p)
files += [E/'benchmark_reporting_integration_v2/observe_exports.py',E/'benchmark_reporting_protocol_v1/EXPOSURE_LEDGER.json',Path(__file__)]
assert all(p.suffix in('.py','.json','.md','.txt')for p in files)
names=[str(p.relative_to(PHASE))for p in files]
stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w:gz')as archive:
    for p,n in zip(files,names):archive.add(p,arcname=n,recursive=False)
packet=stream.getvalue()
code=remote.GUARD+'''import io,sys,tarfile,hashlib,json
from pathlib import Path
root=Path(ROOT_LITERAL)
print('READY',flush=True)
data=sys.stdin.buffer.read(SIZE_LITERAL)
assert len(data)==SIZE_LITERAL and hashlib.sha256(data).hexdigest()==HASH_LITERAL
changed=[]
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz')as archive:
 members=archive.getmembers();assert {m.name for m in members}==set(NAMES_LITERAL)
 for m in members:
  assert m.isfile()and Path(m.name).suffix in('.py','.json','.md','.txt')
  target=root/m.name;assert target.resolve().is_relative_to(root)and not target.is_symlink()
  raw=archive.extractfile(m).read()
  if target.exists():
   assert target.read_bytes()==raw,'Preserve existing source generation: '+str(target)
   continue
  target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb')as handle:handle.write(raw)
  changed.append(m.name)
print(json.dumps({'files':len(members),'packet_bytes':len(data),'changed':changed,'native_jobs':0,'models':0,'forwards':0,'optimizer_updates':0,'TEST_accessed':False,'server77_accessed':False}),flush=True)
'''
for key,val in [('ROOT_LITERAL',repr(remote.REMOTE_PHASE)),('SIZE_LITERAL',str(len(packet))),('HASH_LITERAL',repr(hashlib.sha256(packet).hexdigest())),('NAMES_LITERAL',repr(names))]:code=code.replace(key,val)
child=subprocess.Popen(remote.SSH+['stty raw -echo; python3 -B -c '+shlex.quote(code)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
first=child.stdout.readline()
if first.strip()!=b'READY':
    out,err=child.communicate(timeout=20);raise RuntimeError((first+out+err).decode(errors='replace'))
out,err=child.communicate(input=packet,timeout=90)
if child.returncode:raise RuntimeError((out+err).decode(errors='replace'))
result=json.loads(out.decode().strip())
with(HERE/'COMBINED_V9_SOURCE_STAGE_RECEIPT.json').open('x')as handle:json.dump(result,handle,indent=2,sort_keys=True);handle.write('\n')
print(json.dumps(result,sort_keys=True))
