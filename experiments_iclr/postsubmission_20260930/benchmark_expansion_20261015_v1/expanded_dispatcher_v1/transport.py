"""Explicit compact source transfer and metadata preparation; no broad upload."""
import argparse,hashlib,importlib.util,io,json
from pathlib import Path
import shlex,subprocess,tarfile
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_dispatcher_remote',HERE.parent/'remote_control.py');remote=importlib.util.module_from_spec(s);s.loader.exec_module(remote)
REMOTE=remote.REMOTE_EXPANSION+'/'+HERE.name
NAMES=('contracts.py','engine.py','dispatcher.py','fake_worker.py','test_fake_dispatch.py','prepare_release.py','transport.py',
       'README.md','PARENT_SOURCE.json','DELTA.diff.txt','CHECKS.json','REVIEW_FIX.json','MANIFEST.json')


def stage():
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz')as archive:
        for name in NAMES:archive.add(HERE/name,arcname=name,recursive=False)
    packet=buffer.getvalue()
    code=remote.GUARD+'''import io,sys,tarfile,hashlib,json
from pathlib import Path
root=Path(ROOT_LITERAL)
print('READY',flush=True)
data=sys.stdin.buffer.read(SIZE_LITERAL)
assert len(data)==SIZE_LITERAL and hashlib.sha256(data).hexdigest()==HASH_LITERAL
changed=[]
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz')as archive:
 members=archive.getmembers();assert {m.name for m in members}==set(NAMES_LITERAL)
 for member in members:
  assert member.isfile()and Path(member.name).name==member.name
  target=root/member.name;assert target.resolve().is_relative_to(root)and not target.is_symlink()
  raw=archive.extractfile(member).read()
  if target.exists():assert target.read_bytes()==raw;continue
  root.mkdir(parents=True,exist_ok=True)
  with target.open('xb')as out:out.write(raw)
  changed.append(member.name)
print(json.dumps({'remote':str(root),'source_files':len(members),'changed':changed,'native_children_started':0}),flush=True)
'''.replace('ROOT_LITERAL',repr(REMOTE)).replace('SIZE_LITERAL',str(len(packet))).replace('HASH_LITERAL',repr(hashlib.sha256(packet).hexdigest())).replace('NAMES_LITERAL',repr(NAMES))
    p=subprocess.Popen(remote.SSH+['stty raw -echo; python3 -B -c '+shlex.quote(code)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    first=p.stdout.readline()
    if first.strip()!=b'READY':raise RuntimeError(first.decode(errors='replace'))
    out,err=p.communicate(input=packet,timeout=90)
    if p.returncode:raise RuntimeError((out+err).decode(errors='replace'))
    value=json.loads(out);path=HERE/'STAGE_RECEIPT.json';assert not path.exists();path.write_text(json.dumps(value,indent=2)+'\n');return value


def prepare():
    code='import subprocess,sys,json\nr=subprocess.run([sys.executable,"-B",'+repr(REMOTE+'/prepare_release.py')+'],capture_output=True,text=True)\nprint(json.dumps({"returncode":r.returncode,"stdout":r.stdout,"stderr":r.stderr}))'
    value=json.loads(remote.call(code,timeout=180));path=HERE/'PREPARATION_EXECUTION.json';assert not path.exists();path.write_text(json.dumps(value,indent=2)+'\n');return value


def collect():
    code='''from pathlib import Path
import json,hashlib
root=Path(ROOT_LITERAL)
files={}
for name in ['SUMMARY.json','PREFLIGHT.json']:
 raw=(root/name).read_bytes();files[name]={'text':raw.decode(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
print(json.dumps(files))
'''.replace('ROOT_LITERAL',repr(REMOTE+'/generated_v1'))
    values=json.loads(remote.call(code,timeout=60));folder=HERE/'compact_generated_v1';assert not folder.exists();folder.mkdir();pins={}
    for name,value in values.items():
        raw=value['text'].encode();assert len(raw)==value['bytes']and hashlib.sha256(raw).hexdigest()==value['sha256'];(folder/name).write_bytes(raw)
        pins[name]={k:value[k]for k in('bytes','sha256')}
    (folder/'COLLECTION_RECEIPT.json').write_text(json.dumps({'files':pins,'full_release_or_bindings_copied':False,'fixture_trees_copied':False,'tensors_copied':0},indent=2)+'\n')
    return {'compact_folder':str(folder),'files':pins,'full_release_copied':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','prepare','collect']);a=p.parse_args()
    print(json.dumps(stage()if a.action=='stage'else prepare()if a.action=='prepare'else collect(),sort_keys=True))
