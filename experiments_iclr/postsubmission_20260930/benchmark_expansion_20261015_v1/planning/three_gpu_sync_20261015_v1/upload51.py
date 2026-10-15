"""Upload an explicit, small project source selection to51; no replacement."""
import base64,hashlib,io,json,tarfile,sys
from pathlib import Path
from remote51 import call,ROOT
HERE=Path(__file__).resolve().parent;W=HERE.parents[3];P=W/'postsubmission_research_20260930'
def upload(paths):
 files=[]
 for name in paths:
  p=(P/name).resolve();assert p.is_relative_to(P)
  files.extend([p] if p.is_file() else [x for x in sorted(p.rglob('*')) if x.is_file() and x.suffix in ('.py','.md','.json','.txt','.sh') and '__pycache__'not in x.parts])
 stream=io.BytesIO()
 with tarfile.open(fileobj=stream,mode='w:gz') as a:
  for p in files:a.add(p,arcname=str(p.relative_to(P)),recursive=False)
 data=stream.getvalue();assert len(data)<800000,'Use pinned server stream for large payload'
 code="""from pathlib import Path
import io,tarfile,base64,hashlib,json
R=Path(@ROOT@);P=R/'experiments_iclr/postsubmission_20260930';data=base64.b64decode(@DATA@);assert hashlib.sha256(data).hexdigest()==@HASH@
changed=[]
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as a:
 members=a.getmembers()
 for m in members:
  p=P/m.name;assert m.isfile()and p.resolve().is_relative_to(P)and not p.is_symlink();raw=a.extractfile(m).read()
  if p.exists():assert p.read_bytes()==raw,'Different existing source: '+m.name
  else:
   p.parent.mkdir(parents=True,exist_ok=True)
   with p.open('xb')as f:f.write(raw)
   changed.append(m.name)
print(json.dumps({'files':len(members),'changed':changed,'bytes':len(data),'training_started':False}))
"""
 for k,v in {'ROOT':ROOT,'DATA':base64.b64encode(data).decode(),'HASH':hashlib.sha256(data).hexdigest()}.items():code=code.replace('@'+k+'@',repr(v))
 return json.loads(call(code))
if __name__=='__main__':print(json.dumps(upload(sys.argv[1:])))
