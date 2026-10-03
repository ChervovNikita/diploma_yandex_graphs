"""Publish root-admitted resources and fixed fit releases; keep queue disabled."""
from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import json
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PHASE=ROOT.parent
CANDIDATE=ROOT/'v6_measured_resource_admission_candidate_20261003_v1'
TEXT={'.json','.jsonl','.py','.txt','.md','.log','.sh','.html','.patch'}
OPAQUE={'.pt','.npz','.npy','.pth'}


def desc(p):
    raw=p.read_bytes();raw.decode('utf8')
    return dict(path=p.relative_to(PHASE).as_posix(),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def collect_descriptors(obj):
    if isinstance(obj,dict):
        if all(k in obj for k in ('path','sha256','bytes')) and isinstance(obj['sha256'],str) and len(obj['sha256'])==64 and set(obj['sha256']) <= set('0123456789abcdef'):
            yield {k:obj[k] for k in ('path','sha256','bytes')}
        for v in obj.values():yield from collect_descriptors(v)
    elif isinstance(obj,list):
        for v in obj:yield from collect_descriptors(v)


files={}
pending=[p for p in CANDIDATE.rglob('*') if p.is_file()]
pending += [PHASE/r['path'] for r in json.loads((CANDIDATE/'disabled_fit_releases/split0_gnnm_boundary_4_seed17.json').read_text())['custody_inputs'] if Path(r['path']).suffix in TEXT]
source_dir=PHASE/'amazon_polynormer_paired_family_source_preparation_20261003_v6'
pending += [source_dir/r['path'] for r in json.loads((source_dir/'MANIFEST.json').read_text())['payload']]
pending.append(ROOT/'v6_runtime_execution_receipts_20261003_v1/LOCAL_METADATA_VERIFICATION.json')
while pending:
    p=pending.pop()
    assert p.resolve().is_relative_to(PHASE) and p.suffix in TEXT and not p.is_symlink()
    row=desc(p)
    if row['path'] in files:
        assert files[row['path']]['descriptor']==row
        continue
    raw=p.read_bytes()
    files[row['path']]=dict(descriptor=row,base64=base64.b64encode(raw).decode())
payload=dict(files=list(files.values()))
inventory=dict(files=[e['descriptor'] for e in payload['files']],text_bytes=sum(e['descriptor']['bytes'] for e in payload['files']))
with (HERE/'STAGE_INVENTORY.json').open('x') as h:json.dump(inventory,h,indent=2);h.write('\n')
with (HERE/'STAGE_PAYLOAD.json').open('x') as h:json.dump(payload,h)
CODE='''from pathlib import Path
from datetime import datetime,timezone
import base64,copy,hashlib,json,os,subprocess,sys
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
root=phase/'amazon_polynormer_paired_family_execution_root_20261003_v3'
candidate=root/'v6_measured_resource_admission_candidate_20261003_v1'
packet=root/'v6_admitted_resource_fit_release_preparation_20261003_v1'
resource_path=root/'V6_RESOURCE_ADMISSION_v1.json'
os.chdir(repo)
g=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
assert Path.cwd()==repo and g.returncode==0 and len(g.stdout.strip().splitlines())==1
gpu=[x.strip() for x in g.stdout.strip().split(',')];assert gpu[0]=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
route=dict(repository=str(repo),GPU_UUID=gpu[0])
memory={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
cg={}
for n in ('memory.max','memory.current','memory.high'):
 p=Path('/sys/fs/cgroup')/n
 if p.is_file():cg[n]=p.read_text().strip()
headroom=int(cg['memory.max'])-int(cg['memory.current']) if cg.get('memory.max','max').isdigit() and cg.get('memory.current','').isdigit() else None
fs=os.statvfs(phase)
resources=dict(UTC=datetime.now(timezone.utc).isoformat(),GPU_free_bytes=int(gpu[3])*2**20,GPU_used_bytes=int(gpu[4])*2**20,GPU_utilization_percent=int(gpu[5]),host_MemAvailable_bytes=memory['MemAvailable'],cgroup=cg,cgroup_available_bytes=headroom,filesystem_available_bytes=fs.f_bavail*fs.f_frsize,quota_status='UNKNOWN_UNCERTIFIED',physical_allocation_expiry_status='UNKNOWN_UNCERTIFIED')
assert resources['GPU_free_bytes']>=75*2**30 and memory['MemAvailable']>=32*2**30 and (headroom is None or headroom>=32*2**30)
assert resources['filesystem_available_bytes']>=32*2**30
def confined(value):
 p=phase/value;assert not Path(value).is_absolute() and '..' not in p.parts and p.is_relative_to(phase)
 assert not any(q.is_symlink() for q in (p,*p.parents) if q.is_relative_to(phase));return p
def desc(p):
 assert p.suffix in ('.json','.jsonl','.py','.txt','.md','.log','.sh','.html','.patch')
 b=p.read_bytes();b.decode('utf8');return dict(path=str(p.relative_to(phase)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def verify(row):
 p=confined(row['path']);assert desc(p)==row;return p
def unique(rows):
 result={}
 for r in rows:
  assert r['path'] not in result or result[r['path']]==r
  result[r['path']]=r
 return list(result.values())
created=[]
def write(p,value):
 assert p.is_relative_to(root) and not p.exists() and not any(q.is_symlink() for q in (p,*p.parents) if q.is_relative_to(phase))
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as h:json.dump(value,h,indent=2,allow_nan=False);h.write('\\n');h.flush();os.fsync(h.fileno())
 row=desc(p);created.append(row);return row
transport=json.load(sys.stdin);stage_created=[];stage_existing=[];to_create=[]
for entry in transport['files']:
 row=entry['descriptor'];p=confined(row['path']);b=base64.b64decode(entry['base64'],validate=True);b.decode('utf8')
 assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
 if p.exists():assert desc(p)==row;stage_existing.append(row)
 else:to_create.append((p,b,row))
assert not resource_path.exists() and not packet.exists()
registry=json.loads((root/'registry/REGISTRY.json').read_text());assert len(registry['physical_fits'])==15 and len(registry['families'])==9
assert not any(confined(r[k]).exists() for r in registry['physical_fits'] for k in ('output','claim_path','release_path'))
assert not (root/'v6_releases/closure_v1.json').exists() and not (root/'v6_closure_v1').exists()
assert not (root/'v6_releases/all_v1.json').exists() and not (root/'v6_full_schedule_v1').exists()
for p,b,row in to_create:
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as h:h.write(b);h.flush();os.fsync(h.fileno())
 assert desc(p)==row;stage_created.append(row)
plan=json.loads((candidate/'PLAN.json').read_text())
resource=json.loads(verify(plan['resource_candidate']).read_text());assert desc(candidate/'RESOURCE_ADMISSION_CANDIDATE.json')['sha256']=='c3427a052c4bf96a718e06147c9ad034e04156c64e3e858d196db28797d8d735'
assert resource['execution_authorized'] is False and resource['full_15_fit_schedule_authorized'] is False
assert resource['source']==plan['source'] and resource['registry']==plan['original_registry']
assert resource['qualification_freeze']==plan['qualification_freeze'] and resource['prospective_queue_time_budget_seconds']==691200 and resource['prospective_storage_budget_bytes']==32*2**30
assert resource['quota_status']==resource['physical_allocation_expiry_status']=='UNKNOWN_UNCERTIFIED' and resource['observed_physical_allocation_expiry_UTC'] is None
verify(resource['qualification_freeze']);verify(resource['registry']);verify(resource['attempt_registry']);verify(resource['forecast_evidence'])
qdir=verify(resource['qualification_freeze']).parent
q=json.loads((qdir/'RESULT.json').read_text());t=json.loads((qdir/'TERMINAL.json').read_text())
assert q['status']=='passed' and q['source']==resource['source'] and len(q['forms'])==5 and t['status']=='success' and t['physical_exit_code']==0
assert all(f['local_replay']['bitwise_full_next_step'] and f['global_replay']['bitwise_full_next_step'] and f['retirement_probe']['live_model_Adam_grad_modes_stage_and_RNG_bitwise_unchanged'] and f['retirement_probe']['selected_local_and_global_probe_still_available'] for f in q['forms'])
source_dir=verify(resource['source']['manifest']).parent
assert json.loads(verify(resource['source']['seal']).read_text())['manifest']==resource['source']['manifest']
for r in json.loads((source_dir/'MANIFEST.json').read_text())['payload']:
 actual=desc(source_dir/r['path']);assert (actual['sha256'],actual['bytes'])==(r['sha256'],r['bytes'])
assert [r['row'] for r in registry['physical_fits']]==json.loads((source_dir/'DESIGN.json').read_text())['physical_fit_schedule']
decision=dict(UTC=datetime.now(timezone.utc).isoformat(),authority='Root explicit inspection and admission of exact original15 V6 study.',candidate=plan['resource_candidate'],original_registry=resource['registry'],original_master_source_claim=plan['original_master_source_claim'],qualification_freeze=resource['qualification_freeze'],prospective_queue_time_budget_seconds=691200,operational_storage_budget_bytes=32*2**30,physical_allocation_expiry_and_quota='UNKNOWN_UNCERTIFIED',scope='Admit resource body and publish exact fixed15fit and closure releases. Prepare disabled queue for final root inspection; no queue launch or scoring.',automatic_retry=False,new_registry_or_replacement_cohort=False)
admitted=copy.deepcopy(resource)
admitted.update(status='ADMITTED_ROOT_ORIGINAL15_MEASURED_RESOURCE_ENVELOPE',execution_authorized=True,full_15_fit_schedule_authorized=True,root_inspection_and_release_required=False,root_admission_decision=decision)
resource_changed=[k for k in admitted if admitted[k]!=resource.get(k)]
assert set(resource_changed)=={'status','execution_authorized','full_15_fit_schedule_authorized','root_inspection_and_release_required','root_admission_decision'}
actual_resource=write(resource_path,admitted)
fit_rows=[]
packet.mkdir()
for registered,item in zip(registry['physical_fits'],plan['fit_candidates']):
 before=json.loads(verify(item['candidate']).read_text())
 assert before['fit_id']==registered['id'] and before['execution_authorized'] is False
 assert before['self_path']==registered['release_path'] and before['output']==registered['output'] and before['registered_claim_path']==registered['claim_path']
 assert before['resource_admission']==plan['resource_candidate'] and before['attempt_registry']==resource['attempt_registry'] and before['caps']==resource['caps']
 assert before['qualification_freeze']==resource['qualification_freeze'] and before['source']==resource['source']
 for row in before['custody_inputs']:
  if Path(row['path']).suffix in ('.pt','.npz','.npy','.pth'):
   p=confined(row['path']);assert p.is_file() and p.stat().st_size==row['bytes']
  else:verify(row)
 rebound=copy.deepcopy(before);rebound.update(resource_admission=actual_resource,custody_inputs=unique([*before['custody_inputs'],actual_resource]))
 assert {k:v for k,v in rebound.items() if k not in ('resource_admission','custody_inputs')}=={k:v for k,v in before.items() if k not in ('resource_admission','custody_inputs')}
 disabled=write(packet/'disabled_fit_releases'/(registered['id']+'.json'),rebound)
 actual=write(confined(registered['release_path']),dict(rebound,execution_authorized=True))
 fit_rows.append(dict(fit_id=registered['id'],original_disabled_candidate=item['candidate'],rebound_disabled_candidate=disabled,published=actual,changed_fields=['resource_admission','custody_inputs','execution_authorized']))
before=json.loads(verify(plan['closure_release_disabled']).read_text())
assert before['execution_authorized'] is False and before['kind']=='close' and before['resource_admission']==plan['resource_candidate'] and before['attempt_registry']==resource['attempt_registry']
rebound=copy.deepcopy(before);rebound.update(resource_admission=actual_resource,custody_inputs=unique([*before['custody_inputs'],actual_resource]))
disabled_close=write(packet/'CLOSURE_RELEASE_DISABLED.json',rebound)
actual_close=write(confined(rebound['self_path']),dict(rebound,execution_authorized=True))
before=json.loads(verify(plan['all_release_disabled']).read_text())
queue=copy.deepcopy(before);queue.update(resource_admission=actual_resource,fit_releases=[r['published'] for r in fit_rows],closure_release=actual_close,custody_inputs=unique([*before['custody_inputs'],actual_resource,*[r['published'] for r in fit_rows],actual_close]))
assert queue['execution_authorized'] is False and queue['attempt_registry']==resource['attempt_registry'] and queue['caps']==before['caps'] and queue['caps']['wall_seconds']==673200
assert {k:v for k,v in queue.items() if k not in ('resource_admission','fit_releases','closure_release','custody_inputs')}=={k:v for k,v in before.items() if k not in ('resource_admission','fit_releases','closure_release','custody_inputs')}
for row in queue['fit_releases']:assert json.loads(verify(row).read_text())['execution_authorized'] is True
assert json.loads(verify(actual_close).read_text())['execution_authorized'] is True
disabled_queue=write(packet/'ALL_FITS_RELEASE_DISABLED.json',queue)
argv=['/usr/bin/python3','-B',str(source_dir/'supervise.py'),'--kind','all','--release',str(confined(queue['self_path'])),'--output',str(confined(queue['output']))]
packet_plan=write(packet/'PLAN.json',dict(schema='amazon_polynormer_root_admitted_fixed15_publication_plan_v1',UTC=datetime.now(timezone.utc).isoformat(),source=resource['source'],original_registry=resource['registry'],original_master_source_claim=plan['original_master_source_claim'],resource_original_candidate=plan['resource_candidate'],resource_admitted=actual_resource,resource_changed_fields=resource_changed,root_decision=decision,current_resources=resources,published_fits=fit_rows,closure_original_candidate=plan['closure_release_disabled'],closure_rebound_disabled=disabled_close,closure_published=actual_close,queue_original_candidate=plan['all_release_disabled'],queue_disabled_candidate=disabled_queue,queue_actual_release_path=queue['self_path'],queue_output=queue['output'],queue_execution_authorized=False,queue_release_output_absent=True,exact_after_root_queue_authorization_argv=argv,argv_cwd=str(phase),all15_fit_outputs_and_claims_absent=True,neural_recipes_seeds_selectors_registered_paths_and_attempt_evidence_unchanged=True,predictive_fit_TEST_control_scoring_or_new_registration_launched=False,automatic_retry=False))
manifest=write(packet/'MANIFEST.json',dict(schema='amazon_polynormer_admitted_fixed15_preparation_manifest_v1',UTC=datetime.now(timezone.utc).isoformat(),execution_authorized=False,payload=[dict(relative=p.relative_to(packet).as_posix(),descriptor=desc(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
seal=write(packet/'SEAL.json',dict(schema='amazon_polynormer_admitted_fixed15_preparation_seal_v1',UTC=datetime.now(timezone.utc).isoformat(),manifest=manifest,execution_authorized=False))
new_metadata=[]
for row in created:
 p=verify(row);new_metadata.append(dict(descriptor=row,base64=base64.b64encode(p.read_bytes()).decode()))
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),route=route,current_resources=resources,staging=dict(created_exact_files=stage_created,existing_identical_files=stage_existing,files_overwritten=False),resource_original_candidate=plan['resource_candidate'],resource_admitted=actual_resource,resource_changed_fields=resource_changed,published_fits=fit_rows,closure_published=actual_close,disabled_queue=disabled_queue,packet=dict(plan=packet_plan,manifest=manifest,seal=seal),argv=argv,argv_cwd=str(phase),queue_release_output_absent=True,all15_fit_outputs_claims_absent=True,queue_or_fit_TEST_control_scoring_or_new_registration_launched=False,automatic_retry=False,new_metadata=new_metadata)))
'''
with (HERE/'PUBLICATION_REMOTE_CODE.py.txt').open('x') as h:h.write(CODE)
ssh=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','UpdateHostKeys=no','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru']
started=datetime.now(timezone.utc).isoformat()
r=subprocess.run([*ssh,shlex.join(['/usr/bin/python3','-I','-S','-B','-c',CODE])],input=json.dumps(payload),capture_output=True,text=True)
transport=dict(start_UTC=started,terminal_UTC=datetime.now(timezone.utc).isoformat(),exit_code=r.returncode,stderr=r.stderr,stdout_sha256=hashlib.sha256(r.stdout.encode()).hexdigest(),client_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),remote_code_sha256=hashlib.sha256(CODE.encode()).hexdigest(),private_key_contents_read=False,automatic_retry=False,predictive_fit_queue_or_scoring_launched=False)
with (HERE/'PUBLICATION_TRANSPORT.json').open('x') as h:json.dump(transport,h,indent=2);h.write('\n')
if r.stdout:
    value=json.loads(r.stdout);fetched=value.pop('new_metadata');value['fetched_descriptors']=[]
    for entry in fetched:
        row=entry['descriptor'];raw=base64.b64decode(entry['base64'],validate=True)
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        p=PHASE/row['path'];assert p.resolve().is_relative_to(PHASE)
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as h:h.write(raw)
        value['fetched_descriptors'].append(row)
    with (HERE/'PUBLICATION_RESULT.json').open('x') as h:json.dump(value,h,indent=2);h.write('\n')
    print(json.dumps({k:v for k,v in value.items() if k not in ('staging','published_fits','fetched_descriptors')},indent=2))
    print(json.dumps(dict(staged_created=len(value['staging']['created_exact_files']),staged_existing=len(value['staging']['existing_identical_files']),published_fit_count=len(value['published_fits']),fetched_metadata_count=len(value['fetched_descriptors'])),indent=2))
else:print(json.dumps(transport,indent=2))
raise SystemExit(r.returncode)
