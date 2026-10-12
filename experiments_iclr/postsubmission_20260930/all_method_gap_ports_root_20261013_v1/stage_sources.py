"""Stage this root-owned source proposal on the verified allocation only."""
import base64,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
P=HERE.parent
sys.path.insert(0,str(P/'training_pipeline_v2_20261012/preparation'))
from prepare_molecular_packets_allocation import GUARD,invoke
REMOTE=GUARD+r'''
payload=json.loads(base64.b64decode(sys.argv[1]));out=phase/'all_method_gap_ports_root_20261013_v1'
assert out.is_dir()
for row in payload:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 raw=base64.b64decode(row['data']);assert hashlib.sha256(raw).hexdigest()==row['sha256'];p=out/rel;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==raw,'Preserve different prior source'
 else:p.write_bytes(raw)
 for q in out.rglob('*.py'):compile(q.read_text(),str(q),'exec')
inventory=[{k:v for k,v in r.items() if k!='data'} for r in payload]
(out/'SOURCE_PROPOSAL_EXTRA_HOP_02.json').write_text(json.dumps({'files':inventory,'scientific_fits':0,'models':0,'status':'SOURCE_PROPOSAL_NOT_YET_ADMITTED','GO':False},indent=2)+'\n')
print(json.dumps({'files':len(inventory),'source_compilation':True,'scientific_fits':0}))
'''
def main():
 out=HERE/'stage_02';out.mkdir();payload=[]
 for q in sorted((HERE/'runtime').rglob('*.py')):
  b=q.read_bytes();payload.append({'path':str(q.relative_to(HERE)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'data':base64.b64encode(b).decode()})
 r=invoke(REMOTE,[base64.b64encode(json.dumps(payload).encode()).decode()],out,'STAGE',90);assert r['exit_code']==0;print(r['stdout'])
if __name__=='__main__':main()
