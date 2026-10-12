"""Read the guarded source and metadata once; no training or target tensors."""
import base64
import hashlib
import json
from pathlib import Path
from prepare_molecular_packets_allocation import GUARD, invoke

OUT = Path(__file__).resolve().parent / 'final_audit_root_20261013_v1'
REMOTE = GUARD + r'''
def receipt(p):
 raw=p.read_bytes();return {'path':str(p.relative_to(phase)), 'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest()}
manifest_path=phase/'master_controller_source_20261012_v19/MANIFEST.json'
manifest=json.loads(manifest_path.read_text())
files=[]
for row in manifest['files']:
 p=phase/row['path'];assert receipt(p)==row
 files.append(dict(row,data=base64.b64encode(p.read_bytes()).decode()))
metadata={}
for name in ['practical_master_inputs_20261012_v22/INPUT_INDEX.json',
 'practical_master_inputs_20261012_v22/RESOURCES.json',
 'massive_practical_master_20261012_v22/COMPACT_SUMMARY.json',
 'massive_practical_master_20261012_v22/SUMMARY.json',
 'master_controller_source_20261012_v19/MANIFEST.json']:
 p=phase/name
 if p.exists():metadata[name]={'receipt':receipt(p),'data':json.loads(p.read_text())}
print(json.dumps({'files':files,'metadata':metadata,'route_verified':True,'fits':0}))
'''

def main():
 OUT.mkdir()
 (OUT/'SOURCE_SCRIPT.py').write_bytes(Path(__file__).read_bytes())
 result=invoke(REMOTE,[],OUT,'SOURCE_AND_METADATA_SNAPSHOT',120)
 assert result['exit_code']==0
 value=json.loads(result['stdout'])
 for row in value['files']:
  raw=base64.b64decode(row['data']);assert hashlib.sha256(raw).hexdigest()==row['sha256']
  p=OUT/'controller_v19'/Path(row['path']).name;p.parent.mkdir(exist_ok=True);p.write_bytes(raw)
 for name,row in value['metadata'].items():
  p=OUT/'metadata'/name;p.parent.mkdir(parents=True,exist_ok=True)
  p.write_text(json.dumps(row['data'],indent=2)+'\n')
 (OUT/'SNAPSHOT_RECEIPTS.json').write_text(json.dumps({'files':[{k:v for k,v in r.items() if k!='data'} for r in value['files']],
  'metadata':{k:r['receipt'] for k,r in value['metadata'].items()},'fits':0},indent=2)+'\n')
 print(json.dumps({'source_files':len(value['files']),'metadata_files':list(value['metadata']),'fits':0}))

if __name__=='__main__':main()
