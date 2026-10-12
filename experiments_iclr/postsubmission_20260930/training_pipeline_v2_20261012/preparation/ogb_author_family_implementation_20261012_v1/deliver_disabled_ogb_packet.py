"""Deliver reviewed stdlib source and raw-only disabled packet on the allocation."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import zlib

P=Path('/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/training_pipeline_v2_20261012')
O=P/'preparation/ogb_author_family_implementation_20261012_v1'
source_names=['ogb_author_cli_spec.py','run_ogb_author_cli.py','ogb_declared_seed_entry.py','compile_ogb_author_study.py','author_cli_spec.py','run_author_cli.py','tests/test_ogb_author_cli.py']
sources={name:base64.b64encode((P/name).read_bytes()).decode() for name in source_names}
baseline=json.loads((P/'docs/OGB_BATCHNORM_AUTHOR_REFERENCE_PORT_PLAN_20261012.json').read_text())['existing_author_runtime_bindings']
old_hashes={row['path']:row['sha256'] for row in baseline}
remote=r'''
import base64,hashlib,json,os,pathlib,subprocess,sys,zlib
from datetime import datetime,timezone
assert subprocess.check_output(['hostname'],text=True).strip()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
r=pathlib.Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
p=r/'training_pipeline_v2_20261012';delivery=r/'ogb_author_family_implementation_delivery_20261012_v1';assert not delivery.exists()
delivery.mkdir();(delivery/'previous_live_dispatch').mkdir();(delivery/'engineering').mkdir()
sources=SOURCE_PAYLOAD;old=OLD_HASHES_PAYLOAD
def sha(raw):return hashlib.sha256(raw).hexdigest()
def bound(path):return {'path':str(path.relative_to(r)),'bytes':path.stat().st_size,'sha256':sha(path.read_bytes())}
def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
changed=[]
for name,encoded in sources.items():
 raw=base64.b64decode(encoded);path=p/name
 if name in old:
  previous=path.read_bytes();assert sha(previous) in {old[name],sha(raw)},'LIVE source drift; no overwrite'
  (delivery/'previous_live_dispatch'/path.name).write_bytes(previous)
 elif path.exists():assert path.read_bytes()==raw,'Existing dedicated file differs'
 path.parent.mkdir(parents=True,exist_ok=True)
 if not path.exists() or path.read_bytes()!=raw:
  temp=path.with_name(path.name+'.ogb-delivery-tmp');temp.write_bytes(raw);temp.replace(path);changed.append(name)
sys.path.insert(0,str(p))
from compile_ogb_author_study import materialize_ogb_author_study
from ogb_author_cli_spec import bind_ogb_author_files,load_ogb_author_spec,ogb_author_counts
from spec import canonical,fingerprint,validate
profiles={ds+'-'+model:{'reservation_mib':65536,'wall_seconds':86400,'estimated_seconds':86400,
    'resource_evidence':{'status':'PENDING_ACTUAL_SOURCE_RESOURCE_PROTOCOL_ARCHITECTURE_QUALIFICATION','measurement':False,
    'scope':'Scheduling placeholder only; native help/import and synthetic parser fixtures are not model memory/training-time evidence'}}
    for ds in ['ogbg-molhiv','ogbg-molbace'] for model in ['gin','gin-virtual','gcn','gcn-virtual']}
write(delivery/'RESOURCE_PROFILES_DISABLED.json',profiles)
value,plan,summary=materialize_ogb_author_study(r,'ogb_batchnorm_author_reference_preparation_20261012_v2','training_pipeline_v2_20261012','ogb_batchnorm_author_cli_packet_20261012_v1',profiles)
packet=r/'ogb_batchnorm_author_cli_packet_20261012_v1'
validate(value);counts=ogb_author_counts(value);assert counts['author_cli_groups']==80 and counts['author_registered_maximum_optimizer_updates']==4268000
assert len(plan['groups'])==80 and not plan['scientific_launch_enabled']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
python=r.parents[1]/'.venv/bin/python'
results=[]
for name,args in [('WORKER_HELP',['run_ogb_author_cli.py','--help']),('COMPILER_HELP',['compile_ogb_author_study.py','--help']),('SEED_WRAPPER_HELP',['ogb_declared_seed_entry.py','--help'])]:
 proc=subprocess.run([str(python),'-B',*args],cwd=p,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=30)
 (delivery/'engineering'/(name+'.stdout.txt')).write_text(proc.stdout);(delivery/'engineering'/(name+'.stderr.txt')).write_text(proc.stderr)
 assert proc.returncode==0 and not proc.stderr
 results.append({'name':name,'argv':[str(python),'-B',*args],'exit_code':proc.returncode,'CUDA_VISIBLE_DEVICES':'',
   'stdout':bound(delivery/'engineering'/(name+'.stdout.txt')),'stderr':bound(delivery/'engineering'/(name+'.stderr.txt'))})
script="import importlib,sys,json;[importlib.import_module(n) for n in ['ogb_author_cli_spec','run_ogb_author_cli','compile_ogb_author_study','author_cli_spec','run_author_cli']];assert all(n not in sys.modules for n in ['torch','numpy','torch_geometric','ogb']);print(json.dumps({'stdlib_only':True,'numerical_imports':0}))"
proc=subprocess.run([str(python),'-B','-c',script],cwd=p,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=30)
assert proc.returncode==0 and not proc.stderr
results.append({'name':'DEDICATED_AND_DISPATCH_STDLIB_IMPORT','exit_code':0,'CUDA_VISIBLE_DEVICES':'','stdout':proc.stdout.strip(),'numerical_imports':0})
# Metadata-only preflight of one bound process verifies versions/files/fresh
# cache and custody; no model/provider import or author worker execution.
from run_ogb_author_cli import preflight_ogb_group
admission=preflight_ogb_group(r,value,value['runs'][0]);assert admission[1].is_dir()
checks={'schema':'ogb-author-family-disabled-allocation-engineering-v1','source_files':[bound(p/name) for name in sources],
 'changed_live_files':changed,'frozen_controller_changed':False,'native_CLI_modified':False,'standalone_plan':bound(packet/'PLAN.json'),
 'standalone_plan_hash':fingerprint(plan),'study_hash':fingerprint(value),'counts':counts,'results':results,
 'one_process_metadata_preflight_passed':True,'all80_source_raw_bindings_rehashed':True,
 'actual_author_fits':0,'model_constructions':0,'model_forwards':0,'optimizer_steps':0,'GO_created':False,
 'native_source_help_import_inherited_receipt':value['source']['files'][1],
 'quarantine_files_created':len(list((packet/'quarantine').iterdir())),
 'fresh_processed_directories':sum(1 for run in value['runs'] if not list((r/run['cli']['fresh_processed_dir']).iterdir()))}
assert checks['fresh_processed_directories']==80 and checks['quarantine_files_created']==0
write(delivery/'ALLOCATION_ENGINEERING_CHECK.json',checks)
write(packet/'ALLOCATION_ENGINEERING_CHECK_BINDING.json',bound(delivery/'ALLOCATION_ENGINEERING_CHECK.json'))
files=['PLAN.json','SUMMARY.json','ALLOCATION_ENGINEERING_CHECK_BINDING.json','studies/'+value['study_id']+'.json.gz']
payload={'packet_files':{name:base64.b64encode((packet/name).read_bytes()).decode() for name in files},
 'engineering_check':checks,'engineering_receipt':bound(delivery/'ALLOCATION_ENGINEERING_CHECK.json'),
 'packet_bindings':[bound(packet/name) for name in files]}
print(base64.b64encode(zlib.compress(canonical(payload).encode(),9)).decode())
'''
remote=remote.replace('SOURCE_PAYLOAD',repr(sources)).replace('OLD_HASHES_PAYLOAD',repr(old_hashes))
encoded_remote=base64.b64encode(zlib.compress(remote.encode(),9)).decode()
loader="import base64,zlib;exec(zlib.decompress(base64.b64decode("+repr(encoded_remote)+")).decode())"
wrapper="import subprocess;assert subprocess.check_output(['hostname'],text=True).strip()=='anogena-2-0';assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac'];raise SystemExit(subprocess.run(['/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/.venv/bin/python','-B','-c',"+repr(loader)+"]).returncode)"
args=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','StrictHostKeyChecking=yes','-o','BatchMode=yes','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','python3 -c '+shlex.quote(wrapper)]
start=datetime.now(timezone.utc).isoformat();proc=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=120)
(O/'DELIVERY_TRANSPORT.json').write_text(json.dumps({'start_UTC':start,'exit_code':proc.returncode,'stderr':proc.stderr,'stdout_bytes':len(proc.stdout)},indent=2)+'\n')
if proc.returncode:
 (O/'FAILED_DELIVERY_OUTPUT.txt').write_text(proc.stdout)
 raise SystemExit(proc.stdout[-4000:])
payload=json.loads(zlib.decompress(base64.b64decode(proc.stdout.strip())))
mirror=O/'allocation_artifacts/ogb_batchnorm_author_cli_packet_20261012_v1'
for name,encoded in payload['packet_files'].items():
 destination=mirror/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(base64.b64decode(encoded))
(O/'ALLOCATION_ENGINEERING_CHECK.json').write_text(json.dumps(payload['engineering_check'],indent=2,sort_keys=True)+'\n')
(O/'ALLOCATION_BINDINGS.json').write_text(json.dumps({k:payload[k] for k in ['engineering_receipt','packet_bindings']},indent=2,sort_keys=True)+'\n')
print(json.dumps({'plan_hash':payload['engineering_check']['standalone_plan_hash'],'study_hash':payload['engineering_check']['study_hash'],'counts':payload['engineering_check']['counts'],'engineering_receipt':payload['engineering_receipt'],'frozen_controller_changed':False,'actual_fits':0}))
