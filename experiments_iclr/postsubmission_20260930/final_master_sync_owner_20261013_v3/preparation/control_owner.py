"""Literal allocation guard for finite final-source owner staging/start/status."""
import argparse,base64,hashlib,importlib.util,json,re,zlib
from pathlib import Path
OUT=Path(__file__).resolve().parents[1];PHASE=OUT.parent
spec=importlib.util.spec_from_file_location('guarded_allocation_transport',PHASE/'training_pipeline_v2_20261012/preparation/prepare_molecular_packets_allocation.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
def invoke(code,args,label,timeout=120):
 out=OUT/'transport';out.mkdir(exist_ok=True)
 if (out/(label+'.json')).exists():raise FileExistsError('Prior attempt retained; fresh label required')
 result=helper.invoke(helper.GUARD+code,args,out,label,timeout)
 if result['exit_code']:raise RuntimeError('Guarded owner operation failed; exact receipt retained: '+label)
 return json.loads(result['stdout'].strip())
STAGE=r'''
payload=json.loads(zlib.decompress(base64.b64decode(sys.argv[1])));stage=phase/payload['destination'];assert not stage.exists();stage.mkdir()
for row in payload['files']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts;raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'];p=stage/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
inventory=json.loads((stage/'SYNC_TOOLING_INVENTORY.json').read_text());assert isinstance(inventory,list) and len(inventory)==7
for row in inventory:
 raw=(phase/row['path']).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');tick=time.monotonic();child=subprocess.run([str(repo/'.venv/bin/python'),'-B',str(stage/'tests/check_owner.py')],cwd=stage,env=env,capture_output=True,text=True,timeout=30);qa={'exit_code':child.returncode,'stdout':child.stdout,'stderr':child.stderr,'wall_seconds':time.monotonic()-tick,'models':0,'fits':0,'GO':False};(stage/'ALLOCATION_ENGINEERING_QA.json').write_text(json.dumps(qa,indent=2,sort_keys=True)+'\n')
if child.returncode:print(json.dumps({'qa':qa}));sys.exit(1)
def receipt(p):
 raw=p.read_bytes();return {'path':str(p.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
rows=[receipt(p) for p in sorted(stage.rglob('*')) if p.is_file()];(stage/'SOURCE_INVENTORY.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
ready={'schema':'finite-final-master-sync-owner-ready-v1','status':'STAGED_PROVEN_AWAIT_EXACT_FINAL_MASTER_BINDING','runtime':receipt(stage/'runtime/finite_owner.py'),'source_inventory':receipt(stage/'SOURCE_INVENTORY.json'),'engineering_QA':receipt(stage/'ALLOCATION_ENGINEERING_QA.json'),'tooling_inventory_sha256':'c0c48ed05c7c452d107f2b0d830446287c2d5e3115962de97d80881d5e6e2892','modes':['manifest_audit','bundle'],'old_array_copies':0,'final_master_runs':0,'package_installs':0,'server77_access':False,'models':0,'fits':0,'GO':False};(stage/'ALLOCATION_HANDOFF.json').write_text(json.dumps(ready,indent=2,sort_keys=True)+'\n')
for p in stage.rglob('*'):
 if p.is_file():p.chmod(0o444)
artifacts=[]
for n in ('SOURCE_INVENTORY.json','ALLOCATION_ENGINEERING_QA.json','ALLOCATION_HANDOFF.json'):
 raw=(stage/n).read_bytes();artifacts.append({'path':n,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
print(json.dumps({'ready':ready,'artifacts':artifacts}))
'''
START=r'''
import re
request=json.loads(zlib.decompress(base64.b64decode(sys.argv[1])));assert request.pop('final_master_declared_by_root') is True;stage=phase/request.pop('owner_tooling');expected=request.pop('owner_inventory_sha256');inventory=stage/'SOURCE_INVENTORY.json';assert hashlib.sha256(inventory.read_bytes()).hexdigest()==expected
for row in json.loads(inventory.read_text()):
 p=phase/row['path'];raw=p.read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
label=request.pop('label');assert re.fullmatch('[A-Za-z0-9_]+',label);mode=request.pop('mode');assert mode in ('manifest_audit','bundle');launch=stage/'launches'/label;assert not launch.exists();launch.mkdir(parents=True)
command=[str(repo/'.venv/bin/python'),'-B',str(stage/'runtime/finite_owner.py'),mode,'--project-root',str(phase)]
for key,value in request.items():command += ['--'+key.replace('_','-'),str(value)]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with (launch/'STDOUT.log').open('x') as stdout,(launch/'STDERR.log').open('x') as stderr:child=subprocess.Popen(command,cwd=stage,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
fields=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split();owner={'pid':child.pid,'start_ticks':int(fields[19]),'pgid':int(fields[2])};value={'schema':'exact-final-source-owner-launch-v1','owner':owner,'command':command,'owner_directory':request['owner_directory'],'launch_logs':str(launch.relative_to(phase)),'mode':mode,'scientific_fits':0,'GO':False};p=launch/'LAUNCH_RECEIPT.json';p.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');p.chmod(0o444);print(json.dumps({'launch':value,'launch_receipt':str(p.relative_to(phase))}))
'''
STATUS=r'''
p=phase/sys.argv[1];launch=json.loads(p.read_text());owner=launch['owner'];stat=Path('/proc')/str(owner['pid'])/'stat';alive=False
if stat.exists():
 fields=stat.read_text().rsplit(')',1)[1].split();alive=int(fields[19])==owner['start_ticks'] and int(fields[2])==owner['pgid'] and fields[0]!='Z'
folder=phase/launch['owner_directory'];result=folder/'OWNER_RESULT.json';value={'owner_alive':alive,'owner':owner,'owner_result':json.loads(result.read_text()) if result.is_file() else None,'scientific_fits':0,'GO':False}
if not alive and not result.is_file():value['startup_or_owner_failure_logs']={n:(p.parent/n).read_text()[-8000:] for n in ('STDOUT.log','STDERR.log')}
print(json.dumps(value))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('operation',choices=('stage','start','status'));p.add_argument('--label',required=True);p.add_argument('--request-json');p.add_argument('--launch-receipt');a=p.parse_args()
 if not re.fullmatch('[A-Za-z0-9_]+',a.label):raise ValueError('Fresh safe label required')
 if a.operation=='stage':
  files=[]
  for folder in ('runtime','preparation','tests'):
   for path in sorted((OUT/folder).rglob('*.py')):
    raw=path.read_bytes();files.append({'path':str(path.relative_to(OUT)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
  raw=(OUT/'SYNC_TOOLING_INVENTORY.json').read_bytes();files.append({'path':'SYNC_TOOLING_INVENTORY.json','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
  value=invoke(STAGE,[base64.b64encode(zlib.compress(json.dumps({'destination':OUT.name,'files':files}).encode(),9)).decode()],a.label,120)
  for row in value.pop('artifacts',[]):
   raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'];path=OUT/row['path'];assert not path.exists();path.write_bytes(raw);path.chmod(0o444)
 elif a.operation=='start':
  if not a.request_json:raise ValueError('Exact final-master request JSON required')
  request=json.loads(Path(a.request_json).read_text());value=invoke(START,[base64.b64encode(zlib.compress(json.dumps(request).encode(),9)).decode()],a.label,120)
 else:
  if not a.launch_receipt:raise ValueError('Exact launch receipt path required')
  rel=Path(a.launch_receipt)
  if rel.is_absolute() or '..' in rel.parts:raise ValueError('Project-relative receipt required')
  value=invoke(STATUS,[a.launch_receipt],a.label,60)
 print(json.dumps(value,sort_keys=True))
if __name__=='__main__':main()
