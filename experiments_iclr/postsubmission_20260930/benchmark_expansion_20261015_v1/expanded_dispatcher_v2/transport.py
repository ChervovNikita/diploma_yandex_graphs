"""Exact allowlist source/JSON transport, one no-model probe, metadata compile."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
HERE=Path(__file__).resolve().parent
E=HERE.parent;PHASE=E.parent
spec=importlib.util.spec_from_file_location('_dispatcher2_reviewed_route',E/'remote_control.py')
remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
REMOTE=remote.REMOTE_EXPANSION+'/'+HERE.name


def stage():
    files=[p for folder in (HERE,E/'runtime_link_v7')for p in sorted(folder.iterdir())if p.is_file()and p.suffix in('.py','.json','.md','.txt')and p.name not in('STAGE_RECEIPT.json','COMPILE_RECEIPT.json','INTERPRETER_PROBE.json')]
    ledger=json.loads((HERE/'EVIDENCE_APPLICABILITY.json').read_text())
    for row in ledger['evidence']:
        files.append(PHASE/Path(row['receipt']['path']).relative_to(remote.REMOTE_PHASE))
    plans=json.loads((HERE/'MINIMAL_NATIVE_PROPOSALS.json').read_text())
    for value in plans['charge_authority'].values():files.append(PHASE/Path(value['path']).relative_to(remote.REMOTE_PHASE))
    files=list(dict.fromkeys(files));names=[str(p.relative_to(PHASE))for p in files]
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
   assert target.read_bytes()==raw,'Preserve existing generation: '+str(target)
   continue
  target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb')as out:out.write(raw)
  changed.append(m.name)
print(json.dumps({'files':len(members),'packet_bytes':len(data),'changed':changed,'native_jobs':0,'server77_accessed':False}),flush=True)
'''
    for key,val in [('ROOT_LITERAL',repr(remote.REMOTE_PHASE)),('SIZE_LITERAL',str(len(packet))),('HASH_LITERAL',repr(hashlib.sha256(packet).hexdigest())),('NAMES_LITERAL',repr(names))]:code=code.replace(key,val)
    child=subprocess.Popen(remote.SSH+['stty raw -echo; python3 -B -c '+shlex.quote(code)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    first=child.stdout.readline()
    if first.strip()!=b'READY':
        out,err=child.communicate(timeout=20);raise RuntimeError((first+out+err).decode(errors='replace'))
    out,err=child.communicate(input=packet,timeout=90)
    if child.returncode:raise RuntimeError((out+err).decode(errors='replace'))
    result=json.loads(out.decode().strip());exclusive('STAGE_RECEIPT.json',result);return result


def exclusive(name,value):
    with (HERE/name).open('x')as out:json.dump(value,out,indent=2,sort_keys=True);out.write('\n')


def probe():
    environment=json.loads((HERE/'PYTHON_ENVIRONMENT.json').read_text())
    env_sha=hashlib.sha256((HERE/'PYTHON_ENVIRONMENT.json').read_bytes()).hexdigest()
    code='''from pathlib import Path
import subprocess,json,hashlib
p=Path(ENTRY_LITERAL)
assert hashlib.sha256(p.read_bytes()).hexdigest()==ENTRY_SHA_LITERAL
result=subprocess.run(ARGV_LITERAL,capture_output=True,text=True,timeout=30)
assert result.returncode==0,result.stdout+result.stderr
value=json.loads(result.stdout);assert value['status']=='PASS'and value['environment']['Torch_imported']is False
raw=(json.dumps(value,indent=2,sort_keys=True)+'\\n').encode();out=Path(OUT_LITERAL)
assert not out.exists()
with out.open('xb')as handle:handle.write(raw)
print(json.dumps(value))
'''
    argv=[environment['lexical_executable'],'-B',REMOTE+'/qualified_python_entry.py','--environment',REMOTE+'/PYTHON_ENVIRONMENT.json','--environment-sha256',env_sha,'--probe-only']
    for key,val in [('ENTRY_LITERAL',repr(REMOTE+'/qualified_python_entry.py')),('ENTRY_SHA_LITERAL',repr(hashlib.sha256((HERE/'qualified_python_entry.py').read_bytes()).hexdigest())),('ARGV_LITERAL',repr(argv)),('OUT_LITERAL',repr(REMOTE+'/INTERPRETER_PROBE.json'))]:code=code.replace(key,val)
    result=json.loads(remote.call(code,timeout=60));exclusive('INTERPRETER_PROBE.json',result);return result


def compile_remote():
    code='''from pathlib import Path
import subprocess,json,sys,hashlib
p=Path(PREPARE_LITERAL)
assert hashlib.sha256(p.read_bytes()).hexdigest()==SHA_LITERAL
result=subprocess.run([sys.executable,'-B',str(p)],capture_output=True,text=True,timeout=180)
print(json.dumps({'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}))
'''.replace('PREPARE_LITERAL',repr(REMOTE+'/prepare_release_v2.py')).replace('SHA_LITERAL',repr(hashlib.sha256((HERE/'prepare_release_v2.py').read_bytes()).hexdigest()))
    result=json.loads(remote.call(code,timeout=240));exclusive('COMPILE_RECEIPT.json',result)
    if result['returncode']:raise RuntimeError(json.dumps(result))
    return json.loads(result['stdout'])


def collect():
    code='''from pathlib import Path
import hashlib,json
folder=Path(OUTPUT_LITERAL)
files={}
for name in('SUMMARY.json','PREFLIGHT.json'):
 raw=(folder/name).read_bytes();files[name]={'text':raw.decode(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
print(json.dumps(files))
'''.replace('OUTPUT_LITERAL',repr(REMOTE+'/generated_v1'))
    result=json.loads(remote.call(code,timeout=60));folder=HERE/'compact_generated_v1';folder.mkdir(exist_ok=False);pins={}
    for n,row in result.items():
        b=row['text'].encode();assert len(b)==row['bytes']and hashlib.sha256(b).hexdigest()==row['sha256']
        with (folder/n).open('xb')as out:out.write(b)
        pins[n]={k:row[k]for k in('bytes','sha256')}
    value={'remote_output':REMOTE+'/generated_v1','files':pins,'full_skeleton_release_or_candidates_copied':False,'data_tensors_copied':False,'native_jobs':0}
    (folder/'COLLECTION_RECEIPT.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('stage','probe','compile','collect'));a=p.parse_args()
    print(json.dumps({'stage':stage,'probe':probe,'compile':compile_remote,'collect':collect}[a.action](),sort_keys=True))
