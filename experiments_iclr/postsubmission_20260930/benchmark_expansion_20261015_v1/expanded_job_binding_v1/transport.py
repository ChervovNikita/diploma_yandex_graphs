"""Narrow guarded transport for this metadata compiler package only."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parent
spec = importlib.util.spec_from_file_location('_expanded_compiler_remote_control', EXPANSION/'remote_control.py')
remote = importlib.util.module_from_spec(spec); spec.loader.exec_module(remote)
REMOTE = remote.REMOTE_EXPANSION+'/'+HERE.name
SOURCE_NAMES = ('README.md','build_registry.py','compile_descriptors.py','transport.py','REGISTRY_v1.json','MANIFEST.json')


def stage():
    # Never call remote.upload(): explicit own-package names only.
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for name in SOURCE_NAMES:
            archive.add(HERE/name, arcname=name, recursive=False)
    packet = buffer.getvalue()
    code = remote.GUARD + '''import io,sys,tarfile,hashlib,json
from pathlib import Path
root=Path(ROOT_LITERAL)
print('READY',flush=True)
packet=sys.stdin.buffer.read(SIZE_LITERAL)
assert len(packet)==SIZE_LITERAL and hashlib.sha256(packet).hexdigest()==HASH_LITERAL
changed=[]
with tarfile.open(fileobj=io.BytesIO(packet),mode='r:gz') as archive:
 members=archive.getmembers()
 assert {m.name for m in members}==set(NAMES_LITERAL)
 for member in members:
  assert member.isfile() and Path(member.name).name==member.name
  target=root/member.name
  assert target.resolve().is_relative_to(root) and not target.is_symlink()
  raw=archive.extractfile(member).read()
  if target.exists():
   assert target.read_bytes()==raw, 'Preserve staged source; use a new package/generation: '+str(target)
   continue
  root.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as out:out.write(raw)
  changed.append(member.name)
print(json.dumps({'remote_package':str(root),'source_files':len(members),'packet_bytes':len(packet),'changed':changed,'scientific_fits_started':0}),flush=True)
'''
    code = code.replace('ROOT_LITERAL',repr(REMOTE)).replace('SIZE_LITERAL',str(len(packet))).replace('HASH_LITERAL',repr(hashlib.sha256(packet).hexdigest())).replace('NAMES_LITERAL',repr(SOURCE_NAMES))
    proc = subprocess.Popen(remote.SSH+['stty raw -echo; python3 -B -c '+shlex.quote(code)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    first = proc.stdout.readline()
    if first.strip()!=b'READY':
        proc.terminate();out,err=proc.communicate(timeout=15)
        raise RuntimeError((first+out+err).decode(errors='replace'))
    out,err=proc.communicate(input=packet,timeout=90)
    if proc.returncode:raise RuntimeError((out+err).decode(errors='replace'))
    value=json.loads(out.decode().strip())
    target=HERE/'STAGE_RECEIPT.json'
    assert not target.exists()
    target.write_text(json.dumps(value,indent=2)+'\n')
    return value


def compile_remote(output_name, registry):
    assert output_name.replace('_','').isalnum() and '/' not in output_name
    assert Path(registry).name==registry and registry.endswith('.json')
    code = 'import subprocess,sys\nresult=subprocess.run([sys.executable,"-B",'+repr(REMOTE+'/compile_descriptors.py')+',"--output-name",'+repr(output_name)+',"--registry",'+repr(registry)+'],check=True,capture_output=True,text=True)\nprint(result.stdout)'
    return json.loads(remote.call(code,timeout=240))


def collect(output_name):
    assert output_name.replace('_','').isalnum() and '/' not in output_name
    code='''from pathlib import Path
import json,hashlib
output=Path(OUTPUT_LITERAL)
ready=json.loads((output/'READINESS.json').read_text())
index=json.loads((output/'DESCRIPTOR_INDEX.json').read_text())['fits']
compact={'schema':'expanded-compact-fit-index-v1','full_index':ready['artifacts']['index'],
 'fits':[[r['id'],r['status'],r['ancestor_contract_hash'],r['expanded_contract_hash'],
         None if r['descriptor'] is None else r['descriptor']['sha256']] for r in index],
 'columns':['fit_id','status','ancestor_contract_hash','expanded_contract_hash','descriptor_sha256']}
files={}
for name in ['READINESS.json','BLOCKED_FITS.json','SOURCE_RECEIPTS.json']:
 raw=(output/name).read_bytes()
 files[name]={'text':raw.decode(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
files['COMPACT_INDEX.json']={'value':compact}
print(json.dumps(files))
'''.replace('OUTPUT_LITERAL',repr(REMOTE+'/'+output_name))
    files=json.loads(remote.call(code,timeout=120))
    target=HERE/('compact_'+output_name)
    assert not target.exists()
    target.mkdir()
    receipts={}
    for name,row in files.items():
        if 'text' in row:
            raw=row['text'].encode();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        else:raw=(json.dumps(row['value'],separators=(',',':'))+'\n').encode()
        with (target/name).open('xb') as out:out.write(raw)
        receipts[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    (target/'COLLECTION_RECEIPT.json').write_text(json.dumps({'remote_output':REMOTE+'/'+output_name,'files':receipts,
        'full_fit_descriptors_copied':0,'data_or_tensor_files_copied':0},indent=2)+'\n')
    return {'local_compact_metadata':str(target),'files':receipts,'full_fit_descriptors_copied':0}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('stage','compile','collect'))
    parser.add_argument('--output-name',default='generated_v1')
    parser.add_argument('--registry',default='REGISTRY_v1.json')
    args=parser.parse_args()
    result=stage() if args.action=='stage' else compile_remote(args.output_name,args.registry) if args.action=='compile' else collect(args.output_name)
    print(json.dumps(result,sort_keys=True))
