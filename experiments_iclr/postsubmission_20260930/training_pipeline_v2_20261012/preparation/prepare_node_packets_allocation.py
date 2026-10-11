"""Immutable no-fit/no-score preparation driver for eight new-dataset packets."""
import base64,hashlib,json,shlex,subprocess,zlib
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
PHASE=PACKAGE.parent
DATASETS=('cora','citeseer','pubmed','wikics','amazon-computers','amazon-photo','coauthor-cs','coauthor-physics')
exposure={'schema':'prospective-node-development-exposure-v1','datasets':{}}
for name in DATASETS:
 known=name!='amazon-computers'
 exposure['datasets'][name]={'previously_evaluated':True if known else None,
    'evaluation_access_status':'encountered_development' if known else 'unresolved_never_claim_unused',
    'whole_project_access_audit_complete':False,
    'evidence':['research_dataset_experiment_census_20261011_v1/REPORT.md' if name not in ('cora','amazon-computers') else 'training_pipeline_v2_20261012/preparation/EXPOSURE_DECISION.json'],
    'interpretation':'New to the original five-dataset benchmark; no protected-confirmation claim.'}
files=[]
for rel in ('prepare_node_data.py','docs/NODE_PROVIDER_ROLE_CATALOGUE.json'):
 raw=(PACKAGE/rel).read_bytes();files.append({'path':rel,'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
raw=json.dumps(exposure,indent=2).encode();files.append({'path':'EXPOSURE_DECISION.json','sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
(HERE/'EXPOSURE_DECISION.json').write_bytes(raw+b'\n')
payload=base64.b64encode(zlib.compress(json.dumps({'files':files,'datasets':DATASETS}).encode())).decode()
REMOTE=r'''
import base64,hashlib,json,os,socket,subprocess,sys,time,zlib
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
package=phase/'training_pipeline_v2_20261012'
stage=package/'preparation/node_data_stage_v1'
assert not stage.exists();stage.mkdir(parents=True)
v=json.loads(zlib.decompress(base64.b64decode(sys.argv[1])))
for row in v['files']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 raw=base64.b64decode(row['data']);assert hashlib.sha256(raw).hexdigest()==row['sha256']
 p=stage/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
(stage/'SOURCE_INVENTORY.json').write_text(json.dumps([{k:x for k,x in r.items() if k!='data'} for r in v['files']],indent=2)+'\n')
records=[]
for dataset in v['datasets']:
 command=[str(repo/'.venv/bin/python'),'-B','-u',str(stage/'prepare_node_data.py'),'--dataset',dataset,
  '--catalogue',str(stage/'docs/NODE_PROVIDER_ROLE_CATALOGUE.json'),'--cache-root',str(package/'provider_cache'),
  '--output',str(package/'prepared_data'/dataset),'--exposure-audit',str(stage/'EXPOSURE_DECISION.json'),
  '--expected-pyg-version','2.7.0','--download']
 tick=time.monotonic()
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
 p=subprocess.run(command,cwd=phase,env=env,capture_output=True,text=True,timeout=900)
 row={'dataset':dataset,'exit_code':p.returncode,'wall_seconds':time.monotonic()-tick,'stdout':p.stdout,'stderr':p.stderr,'fits':0,'scores':0}
 (stage/(dataset+'.TRANSPORT.json')).write_text(json.dumps(row,indent=2)+'\n')
 records.append({k:x for k,x in row.items() if k not in ('stdout','stderr')})
 (stage/'PROGRESS.json').write_text(json.dumps(records,indent=2)+'\n')
 print(json.dumps(row),flush=True)
 if p.returncode:break
(stage/'SUMMARY.json').write_text(json.dumps({'records':records,'complete':len(records)==len(v['datasets']) and all(r['exit_code']==0 for r in records),'scientific_training_runs':0,'predictive_scores':0},indent=2)+'\n')
'''
cmd=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,payload])]
assert not (HERE/'NODE_DATA_TRANSPORT_V1.json').exists()
r=subprocess.run(cmd,capture_output=True,text=True,timeout=7500)
(HERE/'NODE_DATA_TRANSPORT_V1.json').write_text(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2)+'\n')
print(r.stdout)
if r.returncode:print(r.stderr)
