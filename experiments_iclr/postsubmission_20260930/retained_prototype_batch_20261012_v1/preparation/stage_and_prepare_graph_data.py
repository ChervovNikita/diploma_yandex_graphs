"""Owned guarded source staging and GRAPH/TRAIN-only view/walk preprocessing."""
import base64,hashlib,json,shlex,subprocess,zlib
from pathlib import Path
from read_allocation_metadata import PREFIX,GUARD
HERE=Path(__file__).resolve().parent;PACKET=HERE.parent;NAME=PACKET.name;TRANSPORT=NAME+'_source_transport'
APPEND=GUARD+r'''
p=phase/sys.argv[1];offset=int(sys.argv[2])
if not offset:assert not p.exists();p.mkdir()
f=p/'payload.b64'
if offset:assert f.stat().st_size==offset
with f.open('ab') as s:s.write(sys.argv[3].encode())
print(json.dumps({'transport_bytes':f.stat().st_size,'route_guard':True,'scientific_fits':0}))
'''
STAGE=GUARD+r'''
f=phase/sys.argv[1]/'payload.b64';raw=f.read_bytes();assert hashlib.sha256(raw).hexdigest()==sys.argv[2]
payload=json.loads(zlib.decompress(base64.b64decode(raw)));packet=phase/payload['name'];assert not packet.exists();packet.mkdir()
for row in payload['assets']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts;raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 target=packet/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
(packet/'STAGED_SOURCE_RECEIPTS.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='data'} for r in payload['assets']],indent=2)+'\n')
print(json.dumps({'source_assets':len(payload['assets']),'route_guard':True,'scientific_fits':0,'GO':False}))
'''
PREPARE=GUARD+r'''
packet=phase/sys.argv[1];python=repo/'.venv/bin/python';env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
p=subprocess.run([str(python),'-B',str(packet/'runtime/prepare_prototype_graphs.py'),'--phase',str(phase),'--packet',str(packet)],cwd=packet,env=env,text=True,timeout=1800)
assert p.returncode==0
print(json.dumps({'GRAPH_TRAIN_preparation_exit':p.returncode,'scientific_fits':0,'heldout_targets_read':0,'GO':False}))
'''
def invoke(code,args,name,timeout=90):
 p=subprocess.run(PREFIX+[shlex.join(['/usr/bin/python3','-I','-S','-B','-c',code,*args])],capture_output=True,text=True,timeout=timeout)
 (HERE/(name+'_TRANSPORT.json')).write_text(json.dumps({'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'route_guard_before_project_operations':True,'scientific_fits':0,'GO':False},indent=2)+'\n')
 if p.returncode:raise RuntimeError('Guarded operation failed, receipt preserved: '+name)
 print(p.stdout,flush=True)
def main():
 rows=[]
 for path in sorted(PACKET.rglob('*')):
  if path.is_file() and not path.name.endswith('_TRANSPORT.json') and '__pycache__' not in path.parts:
   raw=path.read_bytes();rows.append({'path':str(path.relative_to(PACKET)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
 payload={'name':NAME,'assets':rows};encoded=base64.b64encode(zlib.compress(json.dumps(payload,separators=(',',':')).encode()))
 (HERE/'STAGE_PAYLOAD_RECEIPT.json').write_text(json.dumps({'assets':len(rows),'encoded_bytes':len(encoded),'sha256':hashlib.sha256(encoded).hexdigest()},indent=2)+'\n')
 for offset in range(0,len(encoded),50000):invoke(APPEND,[TRANSPORT,str(offset),encoded[offset:offset+50000].decode()],'SOURCE_TRANSFER_'+str(offset//50000))
 invoke(STAGE,[TRANSPORT,hashlib.sha256(encoded).hexdigest()],'SOURCE_STAGE')
 invoke(PREPARE,[NAME],'GRAPH_TRAIN_PREPARATION',1860)
if __name__=='__main__':main()
