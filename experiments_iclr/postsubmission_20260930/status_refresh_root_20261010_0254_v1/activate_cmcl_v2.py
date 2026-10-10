"""Stage/commit exact source, qualify all six full-input interfaces, then run18."""
from pathlib import Path
from datetime import datetime,timezone
import base64,json,subprocess
H=Path(__file__).resolve().parent;P=H.parent
S=P/'private_cmcl_polyformer_source_20261010_v1'
O=P/'private_cmcl_polyformer_root_preparation_20261010_v2'
selected=set()
for folder in (S,O):
    selected.update(q for q in folder.rglob('*') if q.is_file() and q.suffix in ('.py','.json','.md'))
bindings=json.loads((S/'SOURCE_BINDINGS.json').read_text())
for row in list(bindings['dependencies'].values())+bindings['ancestry']:
    q=P/row['path'];assert q.resolve().is_relative_to(P.resolve());selected.add(q)
files={str(q.relative_to(P)):base64.b64encode(q.read_bytes()).decode() for q in sorted(selected)}
remote=r'''
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,socket,subprocess
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=5).splitlines()==[GPU]
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
O=P/'private_cmcl_polyformer_root_preparation_20261010_v2'
FILES=__FILES__
for name,b in FILES.items():
    q=P/name;assert q.resolve().is_relative_to(P);data=base64.b64decode(b)
    if q.exists():assert q.read_bytes()==data,name+' conflicts with frozen source'
    else:q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data)
def readiness(name):
    inv=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,memory.total','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
    with (O/name).open('x') as f:json.dump(dict(UTC=datetime.now(timezone.utc).isoformat(),normal_host_execution=True,GPU_inventory=inv),f,indent=2);f.write('\n')
readiness('READINESS.json')
def git(*args):return subprocess.check_output(['git','-C',str(R),*args],text=True)
prior=git('rev-parse','HEAD').strip();assert prior=='80a566e194f88a719541cf06191f54f4d1844b3d'
assert git('branch','--show-current').strip()=='codex/postsubmission-research-20260930' and not git('diff','--cached','--name-only')
paths=['experiments_iclr/postsubmission_20260930/'+name for name in FILES]+['experiments_iclr/postsubmission_20260930/private_cmcl_polyformer_root_preparation_20261010_v1/READINESS.json']
subprocess.run(['git','-C',str(R),'add','-f','--',*paths],check=True)
staged=git('diff','--cached','--name-only').splitlines();assert staged and set(staged)<=set(paths)
subprocess.run(['git','-C',str(R),'commit','-m','Prepare private specialist-credit learning with full graph controls'],capture_output=True,check=True)
commit=dict(commit=git('rev-parse','HEAD').strip(),prior_head=prior,branch='codex/postsubmission-research-20260930',changed_paths=staged,UTC=datetime.now(timezone.utc).isoformat(),binary_artifacts_included=False,push_verified=False)
q=P/'publication/private_CMCL_safe_path_qualification_and_science_20261010_v2/COMMIT_RECEIPT.json';q.parent.mkdir(parents=True,exist_ok=True)
with q.open('x') as f:json.dump(commit,f,indent=2);f.write('\n')
print(json.dumps(dict(committed=commit)),flush=True)
subprocess.run(['python3','-B',str(O/'render.py'),'--purpose','engineering','--source-review',str(O/'ACTUAL_ROOT_SOURCE_REVIEW_V2.json'),'--owner-review',str(O/'ACTUAL_OWNER_REVIEW_V2.json')],cwd=R,check=True,timeout=30)
plan=O/'ENGINEERING_OWNER_PLAN.json'
res=subprocess.run(['python3','-B',str(O/'queue.py'),'--plan',str(plan),'--plan-sha256',hashlib.sha256(plan.read_bytes()).hexdigest()],cwd=R,check=False,timeout=3690)
fetched={}
for q in [O/'READINESS.json',O/'ENGINEERING_OWNER_PLAN.json',O/'engineering/FAMILY_COMPLETE.json',O/'engineering/FAMILY_FAILURE.json']:
    if q.is_file():fetched[str(q.relative_to(P))]=json.loads(q.read_text())
for condition in ('own_floor','private_cmcl','all_block_cmcl','vanilla_cmcl','private_uniform','private_constant_credit'):
    record='seed9101__'+condition
    for q in [O/'engineering/cells'/record/'COMPLETE.json',O/'engineering/cells'/record/'FAILURE.json',O/'owners'/('engineering__'+record)/'RAW_OWNER_TERMINAL.json']:
        if q.is_file():fetched[str(q.relative_to(P))]=json.loads(q.read_text())
    log=O/'owners'/('engineering__'+record)/'WORKER.log'
    if res.returncode and log.is_file():fetched[str(log.relative_to(P))]=log.read_text()[-5000:]
print(json.dumps(dict(engineering_exit=res.returncode,fetched=fetched)),flush=True)
if res.returncode:raise SystemExit(res.returncode)
readiness('SCIENCE_READINESS.json')
subprocess.run(['python3','-B',str(O/'render.py'),'--purpose','science','--readiness',str(O/'SCIENCE_READINESS.json'),'--source-review',str(O/'ACTUAL_ROOT_SOURCE_REVIEW_V2.json'),'--owner-review',str(O/'ACTUAL_OWNER_REVIEW_V2.json')],cwd=R,check=True,timeout=30)
plan=O/'SCIENCE_OWNER_PLAN.json';digest=hashlib.sha256(plan.read_bytes()).hexdigest()
with (O/'SCIENCE_OWNER.log').open('xb') as log:
    owner=subprocess.Popen(['python3','-B',str(O/'queue.py'),'--plan',str(plan),'--plan-sha256',digest],cwd=R,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
fields=Path('/proc',str(owner.pid),'stat').read_text().rsplit(')',1)[1].split()
launch=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=owner.pid,start_ticks=int(fields[19]),pgid=int(fields[2]),sid=int(fields[3]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),owner_plan_sha256=digest,source_commit=commit['commit'],new_groups=18,TEST_access=False,automatic_retry=False,qualification_VALID_access=True,qualification_not_accuracy_evidence=True)
with (O/'ACTUAL_SCIENCE_LAUNCH.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n')
print(json.dumps(dict(science_launched=launch,owner_plan=json.loads(plan.read_text()),root_admission=json.loads((O/'ROOT_ADOPTION.json').read_text()))),flush=True)
'''.replace('__FILES__',repr(files))
argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','python3 -']
res=subprocess.run(argv,input=remote,capture_output=True,text=True,timeout=3800)
(H/'CMCL_V2_ACTIVATION_TRANSPORT.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=res.returncode,stdout=res.stdout,stderr=res.stderr),indent=2)+'\n')
for line in res.stdout.splitlines():
    if not line.startswith('{'):continue
    item=json.loads(line)
    if 'committed' in item:
        q=P/'publication/private_CMCL_safe_path_qualification_and_science_20261010_v2/COMMIT_RECEIPT.json';q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(item['committed'],indent=2)+'\n')
    for name,data in item.get('fetched',{}).items():
        q=H/'cmcl_v2_metadata'/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(data,indent=2)+'\n')
    if 'science_launched' in item:(H/'CMCL_V2_SCIENCE_LAUNCH.json').write_text(json.dumps(item,indent=2)+'\n')
print(json.dumps(dict(exit_code=res.returncode,science_launched=(H/'CMCL_V2_SCIENCE_LAUNCH.json').exists(),stderr_tail=res.stderr[-2500:])))
raise SystemExit(res.returncode)
