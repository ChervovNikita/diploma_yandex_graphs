"""Owned new Single8 pipeline fixtures on literal allocation; synthetic CPU only."""
import base64,hashlib,json,shlex,subprocess,sys,zlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;P=HERE.parent;ROOT=P.parent;S=P/'single8_successor_v1'
sys.path.insert(0,str(ROOT/'retained_prototype_batch_20261012_v1/preparation'))
from read_allocation_metadata import PREFIX,GUARD
rows=[]
for q in sorted(P.rglob('*')):
 if q.is_file() and q.suffix!='.pyc' and 'CPU_cases' not in q.parts:
  raw=q.read_bytes();rows.append({'path':str(q.relative_to(ROOT)),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
payload=base64.b64encode(zlib.compress(json.dumps(rows).encode())).decode()
attempt=1+len(list(HERE.glob('SINGLE8_CPU_attempt*.json')));transfer='retained_final_five_batch_20261012_v1/preparation/single8_CPU_transport_attempt'+str(attempt)
for start in range(0,len(payload),45000):
 code=GUARD+"\np=phase/sys.argv[1];p.parent.mkdir(parents=True,exist_ok=True)\nwith p.open('a') as f:f.write(sys.argv[2])\n"
 r=subprocess.run(PREFIX+[shlex.join(['/usr/bin/python3','-I','-S','-B','-c',code,transfer,payload[start:start+45000]])],capture_output=True,text=True,timeout=60);assert r.returncode==0,r.stdout
code=GUARD+r'''
import zlib
for row in json.loads(zlib.decompress(base64.b64decode((phase/sys.argv[1]).read_text()))):
 path=phase/row['path'];raw=base64.b64decode(row['data']);assert hashlib.sha256(raw).hexdigest()==row['sha256'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
packet=phase/'retained_final_five_batch_20261012_v1/single8_successor_v1'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
p=subprocess.run([str(repo/'.venv/bin/python'),'-B',str(packet/'check_single8_pipeline_CPU.py')],cwd=packet/'runtime',env=env,capture_output=True,text=True,timeout=240)
value=dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,route_guard=True,actual_input_arrays_opened=0,scientific_fits=0,GO=False)
(packet/('CPU_transport_attempt'+sys.argv[2]+'.json')).write_text(json.dumps(value,indent=2)+'\n')
if p.returncode==0:
 raw=(packet/'SINGLE8_PIPELINE_CPU_QA.json').read_bytes();value['report']=dict(data=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest())
print(json.dumps(value))
'''
p=subprocess.run(PREFIX+[shlex.join(['/usr/bin/python3','-I','-S','-B','-c',code,transfer,str(attempt)])],capture_output=True,text=True,timeout=270)
(HERE/('SINGLE8_CPU_attempt'+str(attempt)+'.json')).write_text(json.dumps(dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,GO=False),indent=2)+'\n');assert p.returncode==0,p.stdout
v=json.loads(next(x for x in p.stdout.splitlines() if x.startswith('{')))
if v['exit_code']==0:
 row=v.pop('report');raw=base64.b64decode(row['data']);assert hashlib.sha256(raw).hexdigest()==row['sha256'];(S/'SINGLE8_PIPELINE_CPU_QA.json').write_bytes(raw)
print(json.dumps(v,indent=2));assert v['exit_code']==0
