"""Narrow source/metadata transport via reviewed expansion remote_control only."""
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
s=importlib.util.spec_from_file_location('_expanded_launch_remote',HERE.parent/'remote_control.py')
remote=importlib.util.module_from_spec(s);s.loader.exec_module(remote)
REMOTE=remote.REMOTE_EXPANSION+'/'+HERE.name
NAMES=('launch_contracts.py','build_plan.py','worker.py','transport.py','HOST_POLICY.json','ADMISSIONS_TEMPLATE.json',
       'MOLBACE24_COST_INPUT.json','REUSED_SUPERVISOR_PINS.json','MEASUREMENT_REQUIREMENTS.json','RESUME_STATE_SCHEMA.json',
       'README.md','ADAPTATION_PLAN.md','MANIFEST.json')


def stage():
    packet=io.BytesIO()
    with tarfile.open(fileobj=packet,mode='w:gz')as archive:
        for name in NAMES:archive.add(HERE/name,arcname=name,recursive=False)
    data=packet.getvalue()
    code=remote.GUARD+'''import io,sys,tarfile,hashlib,json
from pathlib import Path
root=Path(ROOT_LITERAL)
print('READY',flush=True)
data=sys.stdin.buffer.read(SIZE_LITERAL)
assert len(data)==SIZE_LITERAL and hashlib.sha256(data).hexdigest()==HASH_LITERAL
changed=[]
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz')as archive:
 members=archive.getmembers()
 assert {m.name for m in members}==set(NAMES_LITERAL)
 for member in members:
  assert member.isfile()and Path(member.name).name==member.name
  target=root/member.name
  assert target.resolve().is_relative_to(root)and not target.is_symlink()
  raw=archive.extractfile(member).read()
  if target.exists():assert target.read_bytes()==raw;continue
  root.mkdir(parents=True,exist_ok=True)
  with target.open('xb')as out:out.write(raw)
  changed.append(member.name)
print(json.dumps({'source_files':len(members),'changed':changed,'remote':str(root),'models_constructed':0,'scientific_fits_started':0}),flush=True)
'''.replace('ROOT_LITERAL',repr(REMOTE)).replace('SIZE_LITERAL',str(len(data))).replace('HASH_LITERAL',repr(hashlib.sha256(data).hexdigest())).replace('NAMES_LITERAL',repr(NAMES))
    p=subprocess.Popen(remote.SSH+['stty raw -echo; python3 -B -c '+shlex.quote(code)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    first=p.stdout.readline()
    if first.strip()!=b'READY':raise RuntimeError(first.decode(errors='replace'))
    out,err=p.communicate(input=data,timeout=90)
    if p.returncode:raise RuntimeError((out+err).decode(errors='replace'))
    value=json.loads(out);target=HERE/'STAGE_RECEIPT.json';assert not target.exists();target.write_text(json.dumps(value,indent=2)+'\n')
    return value


def run_metadata(action):
    script='build_plan.py'if action=='prepare'else'worker.py'
    argv=[REMOTE+'/'+script]
    if action=='dry-run':argv+=['dry-run','--manifest',REMOTE+'/generated_v1/PLAN.json','--worker','allocation/gpu0','--output',REMOTE+'/generated_v1/DRY_RUN_allocation_gpu0.json']
    code='import subprocess,sys,json\nr=subprocess.run([sys.executable,"-B",*'+repr(argv)+'],cwd='+repr(remote.REMOTE_PHASE)+',capture_output=True,text=True)\nprint(json.dumps({"returncode":r.returncode,"stdout":r.stdout,"stderr":r.stderr}))'
    value=json.loads(remote.call(code,timeout=180))
    target=HERE/(action.replace('-','_').upper()+'_EXECUTION.json');assert not target.exists();target.write_text(json.dumps(value,indent=2)+'\n')
    return value


def collect():
    code='''from pathlib import Path
import hashlib,json
root=Path(ROOT_LITERAL)
files={}
for name in ['SUMMARY.json','COST_CATALOGUE.json','DRY_RUN_allocation_gpu0.json']:
 raw=(root/name).read_bytes()
 files[name]={'text':raw.decode(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
print(json.dumps(files))
'''.replace('ROOT_LITERAL',repr(REMOTE+'/generated_v1'))
    files=json.loads(remote.call(code,timeout=90));target=HERE/'compact_generated_v1';assert not target.exists();target.mkdir();pins={}
    for name,row in files.items():
        raw=row['text'].encode();assert len(raw)==row['bytes']and hashlib.sha256(raw).hexdigest()==row['sha256'];(target/name).write_bytes(raw)
        pins[name]={k:row[k]for k in('bytes','sha256')}
    (target/'COLLECTION_RECEIPT.json').write_text(json.dumps({'files':pins,'full_manifest_copied':False,'full_placement_copied':False,'full_task_descriptors_copied':False,'models_or_data_tensors_copied':0},indent=2)+'\n')
    return {'local_compact_metadata':str(target),'files':pins,'full_manifest_copied':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','prepare','dry-run','collect']);a=p.parse_args()
    value=stage()if a.action=='stage'else collect()if a.action=='collect'else run_metadata(a.action)
    print(json.dumps(value,sort_keys=True))
