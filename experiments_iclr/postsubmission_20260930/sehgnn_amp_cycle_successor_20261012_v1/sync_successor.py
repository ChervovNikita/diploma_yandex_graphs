"""Confined immutable source sync only; root schedules the CUDA fixture."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'training_pipeline_v2_20261012/preparation'))
from prepare_molecular_packets_allocation import GUARD,invoke

TRANSFER=GUARD+r'''
transport=phase/'sehgnn_amp_cycle_successor_20261012_v1_transport'
target=transport/'payload.b64';offset=int(sys.argv[1])
if offset==0:
 assert not transport.exists();transport.mkdir()
else:assert target.stat().st_size==offset
with target.open('ab') as f:f.write(sys.argv[2].encode())
print(json.dumps({'transport_bytes':target.stat().st_size,'fits':0,'GO':False}))
'''
MATERIALIZE=GUARD+r'''
transport=phase/'sehgnn_amp_cycle_successor_20261012_v1_transport'
encoded=(transport/'payload.b64').read_text();assert hashlib.sha256(encoded.encode()).hexdigest()==sys.argv[1]
payload=json.loads(zlib.decompress(base64.b64decode(encoded)))
stage=phase/'sehgnn_amp_cycle_successor_20261012_v1';assert not stage.exists();stage.mkdir()
rows=[]
for row in payload:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 path=stage/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
 rows.append({'path':str(path.relative_to(phase)),'bytes':len(raw),'sha256':row['sha256']})
for path in stage.rglob('*.py'):compile(path.read_text(),str(path),'exec')
(stage/'SYNC_MANIFEST.json').write_text(json.dumps({'schema':'native-SeHGNN-AMP-source-sync-v1','files':rows,'scientific_fits':0,'GO':False},indent=2)+'\n')
print(json.dumps({'status':'source_synced','files':len(rows),'fixture':[r for r in rows if r['path'].endswith('qualify_synthetic_cuda_amp.py')],'runtime_basic':[r for r in rows if r['path'].endswith('runtime/adapters/sehgnn_native.py')],'scientific_fits':0,'CUDA_executions':0,'GO':False}))
'''

def main():
    out=HERE/'transport';out.mkdir()
    files=[]
    for path in sorted(HERE.rglob('*')):
        if path.is_file() and 'transport' not in path.relative_to(HERE).parts and '__pycache__' not in path.parts and path.suffix in ('.py','.sh','.patch','.json'):
            raw=path.read_bytes();files.append({'path':str(path.relative_to(HERE)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
    encoded=base64.b64encode(zlib.compress(json.dumps(files,separators=(',',':')).encode(),9)).decode()
    for offset in range(0,len(encoded),48000):
        r=invoke(TRANSFER,[str(offset),encoded[offset:offset+48000]],out,'TRANSFER_'+str(offset),90)
        assert r['exit_code']==0,'Preserve failed transfer receipt'
    r=invoke(MATERIALIZE,[hashlib.sha256(encoded.encode()).hexdigest()],out,'MATERIALIZE',90)
    assert r['exit_code']==0,'Preserve failed materialization receipt'
    print(r['stdout'])


if __name__=='__main__':main()
