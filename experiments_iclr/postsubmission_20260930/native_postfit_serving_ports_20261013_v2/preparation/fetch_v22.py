import base64,hashlib,json,shlex,subprocess,sys,zlib
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[1]
sys.path.insert(0,str(R/'retained_prototype_batch_20261012_v1/preparation'))
from read_allocation_metadata import PREFIX,GUARD
code=GUARD+r'''
import zlib
root=phase/'master_controller_source_20261013_v22/runtime';assert root.is_dir();rows=[]
for q in sorted(root.rglob('*.py')):
 b=q.read_bytes();rows.append({'path':str(q.relative_to(root)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'data':base64.b64encode(b).decode()})
raw=json.dumps(rows).encode();print(json.dumps({'data':base64.b64encode(zlib.compress(raw)).decode(),'sha256':hashlib.sha256(raw).hexdigest(),'tensor_reads':0,'GO':False}))
'''
p=subprocess.run(PREFIX+[shlex.join(['/usr/bin/python3','-I','-S','-B','-c',code])],capture_output=True,text=True,timeout=90)
(H/'V22_FETCH_TRANSPORT.json').write_text(json.dumps({'exit_code':p.returncode,'stderr':p.stderr,'stdout_sha256':hashlib.sha256(p.stdout.encode()).hexdigest(),'tensor_reads':0,'GO':False},indent=2)+'\n')
assert p.returncode==0,p.stdout
v=json.loads(next(x for x in p.stdout.splitlines() if x.startswith('{')));raw=zlib.decompress(base64.b64decode(v['data']));assert hashlib.sha256(raw).hexdigest()==v['sha256']
out=H/'actual_v22_base';out.mkdir(exist_ok=True);metadata=[]
for row in json.loads(raw):
 b=base64.b64decode(row.pop('data'));assert hashlib.sha256(b).hexdigest()==row['sha256'];q=out/row['path'];q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b);metadata.append(row)
(H/'ACTUAL_V22_METADATA.json').write_text(json.dumps({'files':metadata,'tensor_reads':0,'GO':False},indent=2)+'\n')
print(json.dumps({'files':len(metadata),'GO':False}))
