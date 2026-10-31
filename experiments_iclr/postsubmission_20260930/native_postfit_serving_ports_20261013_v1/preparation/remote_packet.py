"""Guarded idempotent source-only stage / bounded synthetic bridge / metadata read."""
import base64,hashlib,json,shlex,subprocess,sys,zlib
from pathlib import Path
H=Path(__file__).resolve().parent;S=H.parent;R=S.parent;sys.path.insert(0,str(R/'retained_prototype_batch_20261012_v1/preparation'));from read_allocation_metadata import PREFIX,GUARD
mode=sys.argv[1];assert mode in ('stage','qa','metadata','freeze');attempt=sys.argv[2] if len(sys.argv)>2 else '1'
rows=[]
if mode=='stage':
 for q in sorted(S.rglob('*')):
  if q.is_file() and q.suffix!='.pyc' and '__pycache__' not in q.parts and 'synthetic_bridge_qa_01' not in q.parts:
   b=q.read_bytes();rows.append({'path':str(q.relative_to(R)),'sha256':hashlib.sha256(b).hexdigest(),'data':base64.b64encode(b).decode()})
payload=base64.b64encode(zlib.compress(json.dumps(rows).encode())).decode()
transfer=str((H/('stage_payload_attempt'+attempt+'.b64')).relative_to(R))
if mode=='stage':
 for start in range(0,len(payload),45000):
  chunkcode=GUARD+"\np=phase/sys.argv[1];p.parent.mkdir(parents=True,exist_ok=True)\nwith p.open('a') as f:f.write(sys.argv[2])\n"
  v=subprocess.run(PREFIX+[shlex.join(['/usr/bin/python3','-I','-S','-B','-c',chunkcode,transfer,payload[start:start+45000]])],capture_output=True,text=True,timeout=60);assert v.returncode==0,v.stdout
code=GUARD+r'''
import zlib
mode=sys.argv[1];packet=phase/'native_postfit_serving_ports_20261013_v1';value={'mode':mode,'route_guard':True,'GO':False}
if mode=='stage':
 rows=json.loads(zlib.decompress(base64.b64decode((phase/sys.argv[2]).read_text())))
 for row in rows:
  path=phase/row['path'];assert path.resolve().is_relative_to(packet.resolve());b=base64.b64decode(row['data']);assert hashlib.sha256(b).hexdigest()==row['sha256'];path.parent.mkdir(parents=True,exist_ok=True)
  if path.exists() and path.read_bytes()!=b:
   assert path.stat().st_mode&0o200,'Immutable artifact collision';path.write_bytes(b)
  elif not path.exists():path.write_bytes(b)
 value['files']=len(rows)
elif mode in ('qa','metadata'):
 script=packet/('check_bridge_CPU.py' if mode=='qa' else 'check_metadata_only.py');env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');cmd=[str(repo/'.venv/bin/python'),'-B',str(script)] if mode=='qa' else ['/usr/bin/python3','-I','-S','-B',str(script)];p=subprocess.run(cmd,cwd=packet,env=env,capture_output=True,text=True,timeout=90);value.update(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
 if p.returncode==0:
  report=packet/('BRIDGE_CPU_QA.json' if mode=='qa' else 'METADATA_INGEST.json');b=report.read_bytes();value['report']={'data':base64.b64encode(b).decode(),'sha256':hashlib.sha256(b).hexdigest(),'name':report.name}
elif mode=='freeze':
 assert (packet/'CLOSED_HANDOFF.json').is_file() and (packet/'BRIDGE_CPU_QA.json').is_file() and (packet/'METADATA_INGEST.json').is_file()
 rows=[]
 for q in sorted(packet.rglob('*')):
  if q.is_file() and q.suffix!='.pyc':q.chmod(0o444);b=q.read_bytes();rows.append({'path':str(q.relative_to(phase)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
 value['files']=rows
print(json.dumps(value))
'''
p=subprocess.run(PREFIX+[shlex.join(['/usr/bin/python3','-I','-S','-B','-c',code,mode,transfer])],capture_output=True,text=True,timeout=120);transport={'mode':mode,'exit_code':p.returncode,'stderr':p.stderr,'stdout_sha256':hashlib.sha256(p.stdout.encode()).hexdigest(),'route_guard':True,'GO':False};(H/(mode+'_transport_attempt'+attempt+'.json')).write_text(json.dumps(transport,indent=2)+'\n');assert p.returncode==0,p.stdout;v=json.loads(next(x for x in p.stdout.splitlines() if x.startswith('{')))
if 'report' in v:
 row=v.pop('report');b=base64.b64decode(row['data']);assert hashlib.sha256(b).hexdigest()==row['sha256'];(S/row['name']).write_bytes(b)
(H/(mode+'_receipt_attempt'+attempt+'.json')).write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({k:v for k,v in v.items() if k!='files'}));assert v.get('exit_code',0)==0,v.get('stderr')
