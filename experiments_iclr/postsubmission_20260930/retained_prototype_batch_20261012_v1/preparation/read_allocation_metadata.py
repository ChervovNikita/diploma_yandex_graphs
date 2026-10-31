"""Guarded current controller/provider custody reads, no tensor/outcome access."""
import base64,hashlib,json,shlex,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
PREFIX=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru']
GUARD=r'''
import base64,hashlib,json,os,socket,subprocess,sys
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');phase=repo/'experiments_iclr/postsubmission_20260930'
'''
READ=GUARD+r'''
root=phase/'training_pipeline_v2_20261012/prepared_data';providers=[]
for directory in sorted(root.iterdir()):
 if directory.name in {'roman-empire','amazon-ratings','minesweeper','questions','tolokers'}:continue
 path=directory/'CUSTODY.json'
 if path.is_file():
  raw=path.read_bytes();value=json.loads(raw);providers.append({'name':directory.name,'receipt':{'path':str(path.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},'metadata':value})
controller=phase/'master_controller_source_20261012_v8/runtime/compile_master_plan.py';raw=controller.read_bytes()
print(json.dumps({'providers':providers,'controller':{'path':str(controller.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()},'route_guard':True,'tensor_reads':0,'scientific_fits':0,'GO':False}))
'''
p=subprocess.run(PREFIX+[shlex.join(['/usr/bin/python3','-I','-S','-B','-c',READ])],capture_output=True,text=True,timeout=60)
(HERE/'METADATA_READ_TRANSPORT.json').write_text(json.dumps({'exit_code':p.returncode,'stderr':p.stderr,'stdout_sha256':hashlib.sha256(p.stdout.encode()).hexdigest(),'route_guard_before_project_operations':True,'tensor_reads':0,'GO':False},indent=2)+'\n');assert p.returncode==0,p.stdout
value=json.loads([line for line in p.stdout.splitlines() if line.startswith('{')][-1]);raw=base64.b64decode(value['controller'].pop('data'));assert hashlib.sha256(raw).hexdigest()==value['controller']['sha256']
(HERE/'ACTUAL_CONTROLLER_V8_SOURCE.py').write_bytes(raw);(HERE/'CURRENT_ALLOCATION_PROVIDER_METADATA.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({'providers':[{'name':r['name'],'dataset_id':r['metadata'].get('dataset_id'),'schema':r['metadata'].get('schema'),'model_graph':r['metadata'].get('model_graph'),'splits':len(r['metadata'].get('splits',[]))} for r in value['providers']],'controller':value['controller'],'tensor_reads':0,'GO':False},indent=2))
